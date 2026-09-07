"""Sparse CP implementation of the PerFed tensor-factorization baseline.

This module is an independent reconstruction of Li et al., *A Personalized
Federated Tensor Factorization Framework for Distributed IoT Services QoS
Prediction From Heterogeneous Data* (IEEE IoT-J, 2022).  The paper does not
release an implementation and does not specify the learning rate, local epoch
count, random seed, or factor initialization.  The executable reconstruction
therefore freezes the following choices in this module: rank five, one local
epoch by default, ``learning_rate=0.001``, ``gamma=0.01``, deterministic
``numpy`` seeding, and independent positive-uniform factor initialization
scaled by ``rank**(-1/3)``.  The scale avoids a numerically dormant product of
three very small Gaussian factors; it is a reconstruction choice, not a claim
about the paper's hidden implementation.

The model follows equations (12)--(18) and Algorithm 1 of the paper:

* a client/edge server keeps its user factor rows private;
* service and time factors have a client-local personalized copy and a master
  global center;
* local CP gradients read only supplied sparse support tuples;
* equation (17)'s printed ``+`` is interpreted as the standard negative
  gradient step, while equation (18) is implemented as a support-weighted
  elastic center update ``A += alpha * gamma * sum_k p_k(A_k-A)``.

No query labels are accepted by the prediction API.  Only public service/time
  factors occur in a :class:`PerFedClientUpdate` payload, so the private user
  factors cannot be uploaded accidentally through the normal API.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


FP32_BYTES = 4
PARAMETER_BYTES_FP32 = FP32_BYTES


def _as_float32(values: Any) -> np.ndarray:
    """Return a writable float32 copy, keeping model state detached."""

    return np.asarray(values, dtype=np.float32).copy()


def sparse_cp_predict(
    user_factors: np.ndarray,
    service_factors: np.ndarray,
    time_factors: np.ndarray,
    users: Any,
    services: Any,
    times: Any,
) -> np.ndarray:
    """Evaluate a CP model on sparse ``(user, service, time)`` tuples.

    The calculation is the rank-``R`` CP reconstruction
    ``sum_r U[u,r] * S[s,r] * T[t,r]``.  This helper is pure: all inputs are
    read and the returned array is newly allocated.
    """

    user_factors = np.asarray(user_factors, dtype=np.float32)
    service_factors = np.asarray(service_factors, dtype=np.float32)
    time_factors = np.asarray(time_factors, dtype=np.float32)
    users = np.asarray(users, dtype=np.int64)
    services = np.asarray(services, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    if user_factors.ndim != 2 or service_factors.ndim != 2 or time_factors.ndim != 2:
        raise ValueError("all factor matrices must be two-dimensional")
    if not (user_factors.shape[1] == service_factors.shape[1] == time_factors.shape[1]):
        raise ValueError("factor ranks must match")
    if users.shape != services.shape or users.shape != times.shape:
        raise ValueError("prediction indices must have identical shapes")
    if np.any(users < 0) or np.any(users >= user_factors.shape[0]):
        raise IndexError("user index out of range")
    if np.any(services < 0) or np.any(services >= service_factors.shape[0]):
        raise IndexError("service index out of range")
    if np.any(times < 0) or np.any(times >= time_factors.shape[0]):
        raise IndexError("time index out of range")
    return np.sum(
        user_factors[users] * service_factors[services] * time_factors[times],
        axis=-1,
        dtype=np.float32,
    )


def cp_sparse_loss(
    user_factors: np.ndarray,
    service_factors: np.ndarray,
    time_factors: np.ndarray,
    users: Any,
    services: Any,
    times: Any,
    targets: Any,
) -> float:
    """Return mean squared sparse reconstruction loss for a pure audit."""

    targets = np.asarray(targets, dtype=np.float32)
    predictions = sparse_cp_predict(
        user_factors, service_factors, time_factors, users, services, times
    )
    if predictions.shape != targets.shape:
        raise ValueError("targets must align with sparse indices")
    if targets.size == 0:
        raise ValueError("loss requires at least one sparse tuple")
    if not np.all(np.isfinite(targets)):
        raise ValueError("targets must be finite")
    residual = predictions.astype(np.float64) - targets.astype(np.float64)
    return float(np.mean(residual * residual))


@dataclass(frozen=True)
class PerFedState:
    """Deep snapshot of all model state used by read-only audits."""

    private_user_factors: np.ndarray
    local_service_factors: np.ndarray
    local_time_factors: np.ndarray
    global_service_factors: np.ndarray
    global_time_factors: np.ndarray

    def copy(self) -> "PerFedState":
        return PerFedState(
            self.private_user_factors.copy(),
            self.local_service_factors.copy(),
            self.local_time_factors.copy(),
            self.global_service_factors.copy(),
            self.global_time_factors.copy(),
        )


@dataclass(frozen=True)
class PerFedClientUpdate:
    """Public client update returned after sparse local optimization.

    Deliberately absent from this object are user indices and user factors.
    ``service_factors`` and ``time_factors`` are complete public factor
    matrices, matching the paper's public-component exchange contract.  The
    arrays are detached copies, so mutating a returned update cannot mutate a
    client or server model.
    """

    client_id: int
    service_factors: np.ndarray
    time_factors: np.ndarray
    support_entries: int
    uploaded_bytes: int
    downloaded_bytes: int

    @property
    def local_service_factors(self) -> np.ndarray:
        """Compatibility name for the public service factor payload."""

        return self.service_factors

    @property
    def local_time_factors(self) -> np.ndarray:
        """Compatibility name for the public time factor payload."""

        return self.time_factors

    def public_payload(self) -> dict[str, Any]:
        """Return a serialization-ready payload containing public state only."""

        return {
            "client_id": int(self.client_id),
            "service_factors": self.service_factors.copy(),
            "time_factors": self.time_factors.copy(),
            "support_entries": int(self.support_entries),
        }

    def as_dict(self) -> dict[str, Any]:
        """Return accounting plus the public payload for legacy runners."""

        payload = self.public_payload()
        payload.update({
            "uploaded_bytes": int(self.uploaded_bytes),
            "downloaded_bytes": int(self.downloaded_bytes),
        })
        return payload

    def __getitem__(self, key: str) -> Any:
        """Support the old ``result[\"uploaded_bytes\"]`` runner idiom."""

        return self.as_dict()[key]


class PersonalizedFederatedTensorFactorization:
    """Personalized sparse CP model with private user factors.

    Parameters use global user/service/time indices.  ``user_clients`` maps a
    global user to its owning edge server; if omitted, users are assigned to a
    deterministic equal-size IID partition using ``seed``.  Local public
    factors are independently initialized per client and are never replaced by
    the global center automatically; this preserves personalization between
    rounds.  ``global_update``/``aggregate`` only move the master center.

    The legacy methods ``update_client``, ``aggregate`` and ``predict`` are
    retained for the existing temporal runner.  New code can use
    :meth:`sync_round` for an explicit local-update/global-update transaction.
    """

    def __init__(
        self,
        num_users: int,
        num_services: int,
        num_times: int,
        num_clients: int = 6,
        rank: int = 5,
        learning_rate: float = 0.001,
        gamma: float = 0.01,
        seed: int = 0,
        user_clients: Any | None = None,
        *,
        l2: float = 0.0,
        regularization: float | None = None,
    ) -> None:
        if regularization is not None:
            # The runner historically calls the PerFed coupling coefficient
            # ``regularization``.  Accept it as a deliberate alias for gamma.
            gamma = regularization
        dimensions = {
            "num_users": num_users,
            "num_services": num_services,
            "num_times": num_times,
            "num_clients": num_clients,
            "rank": rank,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(float(gamma)) or float(gamma) < 0.0:
            raise ValueError("gamma must be finite and non-negative")
        if not np.isfinite(float(l2)) or float(l2) < 0.0:
            raise ValueError("l2 must be finite and non-negative")

        self.num_users = int(num_users)
        self.num_services = int(num_services)
        self.num_times = int(num_times)
        self.num_clients = int(num_clients)
        self.rank = int(rank)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.l2 = float(l2)
        self.seed = seed

        rng = np.random.default_rng(seed)
        if user_clients is None:
            mapping = np.empty(self.num_users, dtype=np.int64)
            permutation = rng.permutation(self.num_users)
            for client, users_for_client in enumerate(
                np.array_split(permutation, self.num_clients)
            ):
                mapping[users_for_client] = client
            user_clients = mapping
        else:
            supplied = np.asarray(user_clients)
            if supplied.shape != (self.num_users,):
                raise ValueError("user_clients must have one entry per global user")
            if not np.all(np.isfinite(supplied.astype(np.float64))):
                raise ValueError("user_clients must be finite")
            if not np.all(supplied == supplied.astype(np.int64)):
                raise ValueError("user_clients must contain integer client ids")
            user_clients = supplied.astype(np.int64)
        if np.any(user_clients < 0) or np.any(user_clients >= self.num_clients):
            raise ValueError("user_clients contains an out-of-range client id")
        self.user_clients = np.asarray(user_clients, dtype=np.int64).copy()

        # A positive rank^(-1/3)-scaled initialization is an explicit
        # reconstruction choice; the paper does not report its initializer.
        scale = float(self.rank ** (-1.0 / 3.0))
        self.private_user_factors = rng.uniform(
            0.0, scale, size=(self.num_users, self.rank)
        ).astype(np.float32)
        self.global_service_factors = rng.uniform(
            0.0, scale, size=(self.num_services, self.rank)
        ).astype(np.float32)
        self.global_time_factors = rng.uniform(
            0.0, scale, size=(self.num_times, self.rank)
        ).astype(np.float32)
        # Each server has its own personalized public copy.  The copies are
        # independent, as the paper says that personalized models are
        # separately initialized; they are coupled to the center by gamma.
        self.local_service_factors = rng.uniform(
            0.0, scale, size=(self.num_clients, self.num_services, self.rank)
        ).astype(np.float32)
        self.local_time_factors = rng.uniform(
            0.0, scale, size=(self.num_clients, self.num_times, self.rank)
        ).astype(np.float32)

        self._last_support_counts = np.zeros(self.num_clients, dtype=np.int64)
        self._last_global_update: dict[str, Any] | None = None

    @property
    def parameter_count(self) -> int:
        """Count all model parameters, including private client state."""

        return int(
            self.private_user_factors.size
            + self.local_service_factors.size
            + self.local_time_factors.size
            + self.global_service_factors.size
            + self.global_time_factors.size
        )

    @property
    def public_parameter_count(self) -> int:
        """Count one public service/time payload sent per client."""

        return int(self.global_service_factors.size + self.global_time_factors.size)

    @property
    def public_parameter_bytes(self) -> int:
        """FP32 bytes in one public payload; private user rows are excluded."""

        return int(self.public_parameter_count * FP32_BYTES)

    @property
    def private_parameter_count(self) -> int:
        return int(self.private_user_factors.size)

    @property
    def dimension(self) -> int:
        """Compatibility alias for the CP rank used by older runners."""

        return int(self.rank)

    @classmethod
    def random(cls, num_users: int, num_services: int, num_times: int, **kwargs: Any) -> "PersonalizedFederatedTensorFactorization":
        """Construct a seeded model; mirrors ``FullAMFModel.random``."""

        return cls(num_users, num_services, num_times, **kwargs)

    def copy(self) -> "PersonalizedFederatedTensorFactorization":
        """Return a deep model copy, including local/global/private state."""

        clone = type(self)(
            self.num_users,
            self.num_services,
            self.num_times,
            num_clients=self.num_clients,
            rank=self.rank,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            seed=self.seed,
            user_clients=self.user_clients.copy(),
            l2=self.l2,
        )
        clone.private_user_factors = self.private_user_factors.copy()
        clone.local_service_factors = self.local_service_factors.copy()
        clone.local_time_factors = self.local_time_factors.copy()
        clone.global_service_factors = self.global_service_factors.copy()
        clone.global_time_factors = self.global_time_factors.copy()
        clone._last_support_counts = self._last_support_counts.copy()
        clone._last_global_update = None if self._last_global_update is None else dict(self._last_global_update)
        return clone

    def communication_bytes(self, participating_clients: int = 1) -> dict[str, int]:
        """Return public-only FP32 communication for a round.

        Each participating client downloads the current public center and
        uploads its public personalized factors.  User factors never enter
        this accounting.  Index/header protocol bytes are intentionally not
        estimated because the paper reports factor transmission only.
        """

        if isinstance(participating_clients, bool) or int(participating_clients) != participating_clients:
            raise ValueError("participating_clients must be an integer")
        participating_clients = int(participating_clients)
        if participating_clients < 0:
            raise ValueError("participating_clients must be non-negative")
        total = int(participating_clients * self.public_parameter_bytes)
        return {
            "uplink_bytes": total,
            "downlink_bytes": total,
            "public_parameter_bytes_per_client": int(self.public_parameter_bytes),
            "private_parameter_bytes_per_client": int(self.private_parameter_count * FP32_BYTES),
        }

    def _validate_client(self, client_id: Any) -> int:
        if isinstance(client_id, bool) or int(client_id) != client_id:
            raise ValueError("client_id must be an integer")
        client_id = int(client_id)
        if client_id < 0 or client_id >= self.num_clients:
            raise IndexError("client_id out of range")
        return client_id

    def _validate_support(
        self,
        users: Any,
        services: Any,
        times: Any,
        targets: Any,
        client_id: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        times = np.asarray(times, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float32)
        if users.ndim != 1 or services.ndim != 1 or times.ndim != 1 or targets.ndim != 1:
            raise ValueError("support indices and targets must be one-dimensional")
        if not (users.shape == services.shape == times.shape == targets.shape):
            raise ValueError("support indices and targets must have identical shapes")
        if users.size == 0:
            raise ValueError("a participating client must have support entries")
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise IndexError("user index out of range")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("service index out of range")
        if np.any(times < 0) or np.any(times >= self.num_times):
            raise IndexError("time index out of range")
        if not np.all(np.isfinite(targets)):
            raise ValueError("targets must be finite")
        if client_id is not None and np.any(self.user_clients[users] != int(client_id)):
            raise ValueError("support contains a user owned by another client")
        return users.copy(), services.copy(), times.copy(), targets.copy()

    def _client_update_from_state(self, client_id: int, support_entries: int) -> PerFedClientUpdate:
        return PerFedClientUpdate(
            client_id=int(client_id),
            service_factors=self.local_service_factors[client_id].copy(),
            time_factors=self.local_time_factors[client_id].copy(),
            support_entries=int(support_entries),
            uploaded_bytes=int(self.public_parameter_bytes),
            downloaded_bytes=int(self.public_parameter_bytes),
        )

    def update_client(
        self,
        client: int,
        users: Any,
        services: Any,
        times: Any,
        values: Any,
        epochs: int = 1,
        learning_rate: float | None = None,
    ) -> PerFedClientUpdate:
        """Run local sparse SGD for one edge server and return public state.

        ``values`` are support-only QoS targets (typically normalized before
        this call).  The query set is intentionally not an argument.  Tuples
        are processed in canonical ``(user, service, time)`` order for
        deterministic replay, and each tuple uses simultaneous gradients from
        the pre-update factors.  The local public update is
        ``-alpha*(reconstruction_gradient + gamma*(local-global))``; this is
        the mathematically consistent negative-gradient interpretation of Eq.
        (17).  The private user factors are updated in place but excluded from
        the returned payload.
        """

        client_id = self._validate_client(client)
        if isinstance(epochs, bool) or int(epochs) != epochs or int(epochs) <= 0:
            raise ValueError("epochs must be a positive integer")
        epochs = int(epochs)
        eta = self.learning_rate if learning_rate is None else float(learning_rate)
        if not np.isfinite(eta) or eta <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        users, services, times, targets = self._validate_support(
            users, services, times, values, client_id
        )
        # Stable canonical order makes the sequential sparse-SGD path
        # independent of the order in which a caller materializes a mask.
        order = np.lexsort((times, services, users))
        eta32 = np.float32(eta)
        gamma32 = np.float32(self.gamma)
        l232 = np.float32(self.l2)
        for _ in range(epochs):
            for position in order:
                user = int(users[position])
                service = int(services[position])
                time_slice = int(times[position])
                target = np.float32(targets[position])
                before_user = self.private_user_factors[user].copy()
                before_service = self.local_service_factors[client_id, service].copy()
                before_time = self.local_time_factors[client_id, time_slice].copy()
                prediction = np.float32(
                    np.sum(before_user * before_service * before_time, dtype=np.float32)
                )
                error = np.float32(prediction - target)
                user_gradient = error * before_service * before_time + l232 * before_user
                service_gradient = (
                    error * before_user * before_time
                    + gamma32 * (before_service - self.global_service_factors[service])
                    + l232 * before_service
                )
                time_gradient = (
                    error * before_user * before_service
                    + gamma32 * (before_time - self.global_time_factors[time_slice])
                    + l232 * before_time
                )
                self.private_user_factors[user] = (
                    before_user - eta32 * user_gradient
                ).astype(np.float32)
                self.local_service_factors[client_id, service] = (
                    before_service - eta32 * service_gradient
                ).astype(np.float32)
                self.local_time_factors[client_id, time_slice] = (
                    before_time - eta32 * time_gradient
                ).astype(np.float32)
        self._last_support_counts[client_id] = int(users.size)
        return self._client_update_from_state(client_id, int(users.size))

    def _coerce_update(self, update: Any) -> PerFedClientUpdate:
        if isinstance(update, PerFedClientUpdate):
            result = update
        elif isinstance(update, Mapping):
            required = ("client_id", "service_factors", "time_factors")
            missing = [key for key in required if key not in update]
            if missing:
                raise ValueError("client update is missing: " + ", ".join(missing))
            client_id = self._validate_client(update["client_id"])
            result = PerFedClientUpdate(
                client_id=client_id,
                service_factors=_as_float32(update["service_factors"]),
                time_factors=_as_float32(update["time_factors"]),
                support_entries=int(update.get("support_entries", 1)),
                uploaded_bytes=int(update.get("uploaded_bytes", self.public_parameter_bytes)),
                downloaded_bytes=int(update.get("downloaded_bytes", self.public_parameter_bytes)),
            )
        else:
            raise TypeError("updates must contain PerFedClientUpdate objects or public mappings")
        client_id = self._validate_client(result.client_id)
        service = np.asarray(result.service_factors, dtype=np.float32)
        time = np.asarray(result.time_factors, dtype=np.float32)
        if service.shape != (self.num_services, self.rank):
            raise ValueError("client service factor shape mismatch")
        if time.shape != (self.num_times, self.rank):
            raise ValueError("client time factor shape mismatch")
        if not np.all(np.isfinite(service)) or not np.all(np.isfinite(time)):
            raise ValueError("client public factors must be finite")
        if int(result.support_entries) <= 0:
            raise ValueError("support_entries must be positive")
        return PerFedClientUpdate(
            client_id=client_id,
            service_factors=service.copy(),
            time_factors=time.copy(),
            support_entries=int(result.support_entries),
            uploaded_bytes=int(result.uploaded_bytes),
            downloaded_bytes=int(result.downloaded_bytes),
        )

    def _updates_for_aggregate(
        self, updates: Iterable[Any] | Mapping[Any, Any] | None
    ) -> list[PerFedClientUpdate]:
        if updates is None:
            participating = np.flatnonzero(self._last_support_counts > 0)
            if participating.size == 0:
                participating = np.arange(self.num_clients, dtype=np.int64)
            return [
                self._client_update_from_state(int(client), max(1, int(self._last_support_counts[client])))
                for client in participating
            ]
        if isinstance(updates, Mapping):
            values = []
            for key, value in updates.items():
                if isinstance(value, PerFedClientUpdate):
                    values.append(value)
                elif isinstance(value, Mapping):
                    item = dict(value)
                    item.setdefault("client_id", key)
                    values.append(item)
                else:
                    raise TypeError("mapping update values must be public update mappings")
        else:
            values = list(updates)
        if not values:
            raise ValueError("at least one client update is required")
        coerced = [self._coerce_update(item) for item in values]
        clients = [item.client_id for item in coerced]
        if len(set(clients)) != len(clients):
            raise ValueError("client updates must contain unique client ids")
        return coerced

    def global_update(
        self,
        updates: Iterable[Any] | Mapping[Any, Any] | None = None,
        weights: Any | None = None,
        learning_rate: float | None = None,
    ) -> dict[str, Any]:
        """Move global public factors toward a support-weighted local center.

        With local public factors ``S_k,T_k`` and normalized client weights
        ``p_k``, the update is
        ``S <- S + alpha*gamma*sum_k p_k(S_k-S)`` and the analogous time
        update.  This is the explicit client-weighted interpretation of Eq.
        (18).  The operation never modifies private factors or local factors.
        ``updates=None`` aggregates the clients updated since the last call,
        which preserves the legacy ``aggregate()`` runner API.
        """

        client_updates = self._updates_for_aggregate(updates)
        if weights is None:
            raw_weights = np.asarray(
                [item.support_entries for item in client_updates], dtype=np.float64
            )
        elif isinstance(weights, Mapping):
            try:
                raw_weights = np.asarray(
                    [float(weights[item.client_id]) for item in client_updates], dtype=np.float64
                )
            except KeyError as exc:
                raise ValueError("weights mapping is missing a client id") from exc
        else:
            raw_weights = np.asarray(weights, dtype=np.float64)
        if raw_weights.shape != (len(client_updates),):
            raise ValueError("weights must align with client updates")
        if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights <= 0.0):
            raise ValueError("client weights must be finite and positive")
        normalized = raw_weights / np.sum(raw_weights)
        alpha = self.learning_rate if learning_rate is None else float(learning_rate)
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("global learning_rate must be finite and positive")

        service_before = self.global_service_factors.copy()
        time_before = self.global_time_factors.copy()
        service_center = np.asarray(
            sum(
                weight * update.service_factors
                for weight, update in zip(normalized, client_updates)
            ),
            dtype=np.float32,
        )
        time_center = np.asarray(
            sum(
                weight * update.time_factors
                for weight, update in zip(normalized, client_updates)
            ),
            dtype=np.float32,
        )
        service_shift = (service_center - service_before).astype(np.float32)
        time_shift = (time_center - time_before).astype(np.float32)
        coupling_step = np.float32(alpha * self.gamma)
        self.global_service_factors = (
            service_before + coupling_step * service_shift
        ).astype(np.float32)
        self.global_time_factors = (
            time_before + coupling_step * time_shift
        ).astype(np.float32)
        result = {
            "client_ids": tuple(int(update.client_id) for update in client_updates),
            "weights": normalized.copy(),
            "support_entries": int(sum(update.support_entries for update in client_updates)),
            "coupling_step": float(alpha * self.gamma),
            "service_center_shift_norm": float(np.linalg.norm(service_shift)),
            "time_center_shift_norm": float(np.linalg.norm(time_shift)),
            "uplink_bytes": int(sum(update.uploaded_bytes for update in client_updates)),
            "downlink_bytes": int(sum(update.downloaded_bytes for update in client_updates)),
        }
        self._last_global_update = result
        return result

    def aggregate(
        self,
        updates: Iterable[Any] | Mapping[Any, Any] | None = None,
        weights: Any | None = None,
        learning_rate: float | None = None,
    ) -> dict[str, Any]:
        """Compatibility alias for :meth:`global_update` (equation 18)."""

        return self.global_update(updates, weights=weights, learning_rate=learning_rate)

    def local_update(self, *args: Any, **kwargs: Any) -> PerFedClientUpdate:
        """Explicit-name alias for :meth:`update_client`."""

        return self.update_client(*args, **kwargs)

    def global_center_update(
        self,
        updates: Iterable[Any] | Mapping[Any, Any] | None = None,
        weights: Any | None = None,
        learning_rate: float | None = None,
    ) -> dict[str, Any]:
        """Explicit-name alias for :meth:`global_update`."""

        return self.global_update(updates, weights=weights, learning_rate=learning_rate)

    @staticmethod
    def _batch_parts(batch: Any) -> tuple[Any, Any, Any, Any]:
        if isinstance(batch, Mapping):
            aliases = {
                "users": ("users", "user_indices"),
                "services": ("services", "service_indices"),
                "times": ("times", "time_indices"),
                "targets": ("targets", "values", "support_values"),
            }
            resolved = {}
            for name, names in aliases.items():
                for candidate in names:
                    if candidate in batch:
                        resolved[name] = batch[candidate]
                        break
                if name not in resolved:
                    raise ValueError(f"batch is missing {name}")
            return (
                resolved["users"], resolved["services"],
                resolved["times"], resolved["targets"],
            )
        try:
            users, services, times, targets = batch
        except (TypeError, ValueError) as exc:
            raise ValueError("each batch must be (users, services, times, targets)") from exc
        return users, services, times, targets

    def sync_round(
        self,
        batches: Mapping[Any, Any] | Iterable[Any],
        *,
        epochs: int = 1,
        local_learning_rate: float | None = None,
        global_learning_rate: float | None = None,
        weights: Any | None = None,
    ) -> dict[str, Any]:
        """Run local support-only updates followed by one global center update.

        ``batches`` may be a mapping ``client_id -> (users, services, times,
        targets)`` or an iterable of ``(client_id, users, services, times,
        targets)`` records.  The returned result contains only public update
        objects and communication/accounting metadata.
        """

        if isinstance(batches, Mapping):
            records = [(client, batch) for client, batch in batches.items()]
        else:
            records = []
            for record in batches:
                try:
                    client, users, services, times, targets = record
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "iterable batches must contain (client, users, services, times, targets)"
                    ) from exc
                records.append((client, (users, services, times, targets)))
        if not records:
            raise ValueError("sync_round requires at least one client batch")
        updates = []
        for client, batch in records:
            users, services, times, targets = self._batch_parts(batch)
            updates.append(
                self.update_client(
                    client, users, services, times, targets,
                    epochs=epochs, learning_rate=local_learning_rate,
                )
            )
        global_result = self.global_update(
            updates, weights=weights, learning_rate=global_learning_rate
        )
        return {
            "updates": tuple(updates),
            "global": global_result,
            "participating_clients": len(updates),
            "uplink_bytes": int(sum(item.uploaded_bytes for item in updates)),
            "downlink_bytes": int(sum(item.downloaded_bytes for item in updates)),
        }

    def predict(
        self,
        users: Any,
        services: Any,
        times: Any,
        client_ids: Any | None = None,
    ) -> np.ndarray:
        """Predict sparse entries using private user and local public factors.

        This method accepts no targets or labels and performs no state writes,
        making it suitable for a query-only read-only audit.
        """

        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        times = np.asarray(times, dtype=np.int64)
        if users.shape != services.shape or users.shape != times.shape:
            raise ValueError("prediction indices must have identical shapes")
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise IndexError("user index out of range")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("service index out of range")
        if np.any(times < 0) or np.any(times >= self.num_times):
            raise IndexError("time index out of range")
        if client_ids is None:
            clients = self.user_clients[users]
        else:
            clients = np.asarray(client_ids, dtype=np.int64)
            if clients.shape != users.shape:
                raise ValueError("client_ids must align with prediction indices")
            if np.any(clients < 0) or np.any(clients >= self.num_clients):
                raise IndexError("client_id out of range")
        return np.sum(
            self.private_user_factors[users]
            * self.local_service_factors[clients, services]
            * self.local_time_factors[clients, times],
            axis=-1,
            dtype=np.float32,
        )

    def reconstruction_loss(
        self,
        users: Any,
        services: Any,
        times: Any,
        targets: Any,
        client_ids: Any | None = None,
    ) -> float:
        """Return support/query reconstruction loss without changing state."""

        targets = np.asarray(targets, dtype=np.float32)
        predictions = self.predict(users, services, times, client_ids=client_ids)
        if predictions.shape != targets.shape:
            raise ValueError("targets must align with prediction indices")
        if targets.size == 0 or not np.all(np.isfinite(targets)):
            raise ValueError("targets must be finite and non-empty")
        residual = predictions.astype(np.float64) - targets.astype(np.float64)
        return float(np.mean(residual * residual))

    def state_copy(self) -> PerFedState:
        """Return a deep snapshot for causal/read-only audits."""

        return PerFedState(
            self.private_user_factors.copy(),
            self.local_service_factors.copy(),
            self.local_time_factors.copy(),
            self.global_service_factors.copy(),
            self.global_time_factors.copy(),
        )

    def state_arrays(self) -> tuple[np.ndarray, ...]:
        """Return deep-copied arrays in a stable order for audit code."""

        snapshot = self.state_copy()
        return (
            snapshot.private_user_factors,
            snapshot.local_service_factors,
            snapshot.local_time_factors,
            snapshot.global_service_factors,
            snapshot.global_time_factors,
        )

    def state_unchanged(self, snapshot: PerFedState | tuple[np.ndarray, ...]) -> bool:
        """Compare current state with a prior :meth:`state_copy`/array tuple."""

        if isinstance(snapshot, PerFedState):
            arrays = (
                snapshot.private_user_factors,
                snapshot.local_service_factors,
                snapshot.local_time_factors,
                snapshot.global_service_factors,
                snapshot.global_time_factors,
            )
        else:
            arrays = tuple(snapshot)
        current = self.state_arrays()
        return len(arrays) == len(current) and all(
            np.array_equal(before, after) for before, after in zip(arrays, current)
        )

    def read_only_audit(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """Run a prediction/loss operation and report whether state stayed fixed."""

        before = self.state_copy()
        operation(*args, **kwargs)
        return self.state_unchanged(before)

    def client_public_state(self, client: int) -> dict[str, Any]:
        """Return one client's public state without exposing private factors."""

        client_id = self._validate_client(client)
        return {
            "client_id": client_id,
            "service_factors": self.local_service_factors[client_id].copy(),
            "time_factors": self.local_time_factors[client_id].copy(),
        }


# Names used by earlier runners and by downstream baseline code.
PerFedModel = PersonalizedFederatedTensorFactorization
PerFedTensorFactorization = PersonalizedFederatedTensorFactorization
PersonalizedCPModel = PersonalizedFederatedTensorFactorization


__all__ = [
    "FP32_BYTES",
    "PARAMETER_BYTES_FP32",
    "PerFedState",
    "PerFedClientUpdate",
    "PersonalizedFederatedTensorFactorization",
    "PerFedModel",
    "PerFedTensorFactorization",
    "PersonalizedCPModel",
    "sparse_cp_predict",
    "cp_sparse_loss",
]
