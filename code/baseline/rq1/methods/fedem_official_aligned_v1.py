"""Independent official-mechanism-first FedEM adaptation for causal QoS.

The reference implementation uses a user-level FCF recommender, a private
MLP adapter, and a similarity/sample-size aggregation matrix.  This module
keeps those mechanisms while replacing the recommendation objective with a
fixed-log1p QoS regression model and a causal one-slice support/query API.

For a service factor, each client adapter receives only
``[q_local; q_server - q_local]`` where ``q`` is the rank-dimensional service
factor.  The default rank-16 adapter is the reference-shaped MLP
``32 -> 16 -> 8 -> 1`` with ReLU/ReLU/Sigmoid activations.  The same scalar
gate is applied to the service bias, but the bias is not an adapter input.
Adapter and embedding updates use separate Adam steps in the same order as the
reference implementation.  Adapter parameters stay client-specific: a copy
may be present in the server participant state, but it is never mixed across
clients.

Only current support indices and values enter local updates, similarity, and
aggregation.  Query labels, future values, and query masks are never accepted
by the update API.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


FP32_BYTES = 4
INDEX_BYTES = 4


def _as_float32(values: Any) -> np.ndarray:
    """Return a detached writable finite float32 array."""

    result = np.asarray(values, dtype=np.float32).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("values must be finite")
    return result


def sigmoid(values: Any) -> np.ndarray:
    """Numerically stable sigmoid returning float32 values."""

    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, -30.0, 30.0)
    result = 1.0 / (1.0 + np.exp(-clipped))
    return result.astype(np.float32)


def cosine_similarity(left: Any, right: Any) -> float:
    """Return a non-negative cosine similarity for finite vectors."""

    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("similarity vectors must have the same shape")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("similarity vectors must be finite")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.clip(np.dot(left, right) / (left_norm * right_norm), 0.0, 1.0))


def simplex_projection(values: Any) -> np.ndarray:
    """Project a vector onto the non-negative unit simplex.

    The reference FedEM QP has objective ``||x||^2 + c^T x`` under
    ``x >= 0, sum(x)=1``.  Its exact solution is the Euclidean simplex
    projection of ``-c/2``; no CVXPY dependency is needed.
    """

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("simplex projection requires a non-empty finite vector")
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values)
    candidates = sorted_values - (cumulative - 1.0) / np.arange(1, values.size + 1)
    valid = np.flatnonzero(candidates > 0.0)
    rho = int(valid[-1]) if valid.size else 0
    theta = float((cumulative[rho] - 1.0) / (rho + 1))
    result = np.maximum(values - theta, 0.0).astype(np.float32)
    total = float(np.sum(result, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("simplex projection produced an invalid vector")
    result /= np.float32(total)
    return result


@dataclass(frozen=True)
class FedEMClientUpdate:
    """Dense public participant state returned by one user-level client.

    ``user`` parameters are private and absent.  The adapter arrays are copied
    into the participant state for official-mechanism accounting, but the
    server never averages them; each client retains its own adapter.
    """

    client_id: int
    service_indices: np.ndarray
    service_factors: np.ndarray
    service_biases: np.ndarray
    support_counts: np.ndarray
    support_entries: int
    support_signature: np.ndarray
    adapter_weights: tuple[np.ndarray, ...]
    adapter_biases: tuple[np.ndarray, ...]
    service_indices_bytes: int
    adapter_bytes: int
    uploaded_bytes: int
    downloaded_bytes: int

    def public_payload(self) -> dict[str, Any]:
        """Return detached public participant state without user parameters."""

        return {
            "client_id": int(self.client_id),
            "service_indices": self.service_indices.copy(),
            "service_factors": self.service_factors.copy(),
            "service_biases": self.service_biases.copy(),
            "support_counts": self.support_counts.copy(),
            "support_entries": int(self.support_entries),
            "support_signature": self.support_signature.copy(),
            "adapter_weights": tuple(array.copy() for array in self.adapter_weights),
            "adapter_biases": tuple(array.copy() for array in self.adapter_biases),
            "service_indices_bytes": int(self.service_indices_bytes),
            "adapter_bytes": int(self.adapter_bytes),
        }

    def as_dict(self) -> dict[str, Any]:
        """Return detached participant state and wire-accounting metadata."""

        payload = self.public_payload()
        payload.update({
            "uploaded_bytes": int(self.uploaded_bytes),
            "downloaded_bytes": int(self.downloaded_bytes),
        })
        return payload


class FedEMQoSModel:
    """FedEM-style personalized model merging for online QoS prediction."""

    def __init__(
        self,
        num_users: int,
        num_services: int,
        num_times: int,
        rank: int = 16,
        learning_rate: float = 0.1,
        regularization: float = 3e-4,
        clients_sample_ratio: float = 1.0,
        agg_clients_ratio: float = 0.1,
        agg_mode: str = "both",
        sim_mode: str = "l2",
        alpha: float = 0.9,
        adapter_learning_rate: float = 0.1,
        adapter_regularization: float = 3e-4,
        adapter_architecture: Any | None = None,
        seed: int = 0,
    ):
        dimensions = (num_users, num_services, num_times, rank)
        if any(isinstance(value, bool) or int(value) != value for value in dimensions):
            raise ValueError("model dimensions must be integers")
        if min(dimensions) <= 0:
            raise ValueError("model dimensions must be positive")
        if not np.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(float(regularization)) or float(regularization) < 0.0:
            raise ValueError("regularization must be finite and non-negative")
        if not np.isfinite(float(clients_sample_ratio)) or not 0.0 < float(clients_sample_ratio) <= 1.0:
            raise ValueError("clients_sample_ratio must be in (0, 1]")
        if not np.isfinite(float(agg_clients_ratio)) or not 0.0 < float(agg_clients_ratio) <= 1.0:
            raise ValueError("agg_clients_ratio must be in (0, 1]")
        if agg_mode not in {"avg", "sim", "both"}:
            raise ValueError("agg_mode must be one of avg, sim, both")
        if sim_mode not in {"l2", "cos"}:
            raise ValueError("sim_mode must be one of l2, cos")
        if not np.isfinite(float(alpha)) or float(alpha) < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if not np.isfinite(float(adapter_learning_rate)) or float(adapter_learning_rate) <= 0.0:
            raise ValueError("adapter_learning_rate must be finite and positive")
        if not np.isfinite(float(adapter_regularization)) or float(adapter_regularization) < 0.0:
            raise ValueError("adapter_regularization must be finite and non-negative")

        self.num_users = int(num_users)
        self.num_services = int(num_services)
        self.num_times = int(num_times)
        self.rank = int(rank)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.clients_sample_ratio = float(clients_sample_ratio)
        self.agg_clients_ratio = float(agg_clients_ratio)
        self.agg_mode = str(agg_mode)
        self.sim_mode = str(sim_mode)
        self.alpha = float(alpha)
        self.adapter_learning_rate = float(adapter_learning_rate)
        self.adapter_regularization = float(adapter_regularization)
        self.rng = np.random.default_rng(int(seed))

        if adapter_architecture is None:
            adapter_architecture = (2 * self.rank, self.rank, max(1, self.rank // 2), 1)
        adapter_architecture = tuple(int(value) for value in adapter_architecture)
        if len(adapter_architecture) < 2 or adapter_architecture[0] != 2 * self.rank or adapter_architecture[-1] != 1:
            raise ValueError("adapter_architecture must start at 2*rank and end at 1")
        if any(value <= 0 for value in adapter_architecture):
            raise ValueError("adapter_architecture widths must be positive")
        self.adapter_layer_sizes = adapter_architecture
        self.adapter_input_dim = int(adapter_architecture[0])
        self.adapter_output_dim = 1

        scale = np.float32(0.05)
        self.private_user_factors = self.rng.normal(
            0.0, float(scale), (self.num_users, self.rank)
        ).astype(np.float32)
        self.private_user_biases = np.zeros(self.num_users, dtype=np.float32)
        self.server_service_factors = self.rng.normal(
            0.0, float(scale), (self.num_services, self.rank)
        ).astype(np.float32)
        self.server_service_biases = np.zeros(self.num_services, dtype=np.float32)
        self.local_service_factors = np.broadcast_to(
            self.server_service_factors[None, :, :],
            (self.num_users, self.num_services, self.rank),
        ).copy()
        self.local_service_biases = np.broadcast_to(
            self.server_service_biases[None, :],
            (self.num_users, self.num_services),
        ).copy()

        # Reference PyTorch Linear uses a Kaiming-uniform bound of 1/sqrt(in).
        # One initial adapter is copied to every client; subsequent updates are
        # private because each client owns its own first dimension.
        initial_weights: list[np.ndarray] = []
        initial_biases: list[np.ndarray] = []
        for input_dim, output_dim in zip(adapter_architecture[:-1], adapter_architecture[1:]):
            bound = 1.0 / np.sqrt(float(input_dim))
            initial_weights.append(
                self.rng.uniform(-bound, bound, (input_dim, output_dim)).astype(np.float32)
            )
            initial_biases.append(
                self.rng.uniform(-bound, bound, output_dim).astype(np.float32)
            )
        self.adapter_weights = [
            np.broadcast_to(weight[None, :, :], (self.num_users,) + weight.shape).copy()
            for weight in initial_weights
        ]
        self.adapter_biases = [
            np.broadcast_to(bias[None, :], (self.num_users,) + bias.shape).copy()
            for bias in initial_biases
        ]

        # A per-client personalized server model exists only for clients in the
        # current aggregation subset.  Adapter copies model the reference
        # participant state but never participate in aggregation.
        self.server_service_factors_by_client = np.broadcast_to(
            self.server_service_factors[None, :, :],
            (self.num_users, self.num_services, self.rank),
        ).copy()
        self.server_service_biases_by_client = np.broadcast_to(
            self.server_service_biases[None, :],
            (self.num_users, self.num_services),
        ).copy()
        self.server_available = np.zeros(self.num_users, dtype=bool)
        self.server_adapter_weights = [np.zeros_like(array) for array in self.adapter_weights]
        self.server_adapter_biases = [np.zeros_like(array) for array in self.adapter_biases]
        self.server_adapter_available = np.zeros(self.num_users, dtype=bool)

        self.last_similarity_matrix = np.zeros((self.num_users, self.num_users), dtype=np.float32)
        self.last_aggregation_matrix = np.zeros((self.num_users, self.num_users), dtype=np.float32)
        self.last_support_counts = np.zeros(self.num_users, dtype=np.int32)
        self.last_service_support_counts = np.zeros(
            (self.num_users, self.num_services), dtype=np.int32
        )
        self.last_participating_clients: tuple[int, ...] = ()
        self.last_aggregation_clients: tuple[int, ...] = ()
        self.last_update_trace: tuple[str, ...] = ()
        self.round_index = 0

    @property
    def num_clients(self) -> int:
        return self.num_users

    @property
    def public_parameter_count(self) -> int:
        return int(self.num_services * (self.rank + 1))

    @property
    def public_parameter_bytes(self) -> int:
        return int(self.public_parameter_count * FP32_BYTES)

    @property
    def adapter_parameter_count(self) -> int:
        return int(
            sum(array[0].size for array in self.adapter_weights)
            + sum(array[0].size for array in self.adapter_biases)
        )

    @property
    def adapter_parameter_bytes(self) -> int:
        return int(self.adapter_parameter_count * FP32_BYTES)

    @property
    def private_parameter_count(self) -> int:
        return int(
            self.private_user_factors.size
            + self.private_user_biases.size
            + sum(array.size for array in self.adapter_weights)
            + sum(array.size for array in self.adapter_biases)
        )

    def state_arrays(self) -> tuple[np.ndarray, ...]:
        """Return every mutable numeric state array in a stable order."""

        arrays: list[np.ndarray] = [
            self.private_user_factors,
            self.private_user_biases,
            self.local_service_factors,
            self.local_service_biases,
            self.server_service_factors,
            self.server_service_biases,
            self.server_service_factors_by_client,
            self.server_service_biases_by_client,
        ]
        arrays.extend(self.adapter_weights)
        arrays.extend(self.adapter_biases)
        arrays.extend(self.server_adapter_weights)
        arrays.extend(self.server_adapter_biases)
        arrays.extend([
            self.last_similarity_matrix,
            self.last_aggregation_matrix,
            self.last_support_counts,
            self.last_service_support_counts,
        ])
        return tuple(arrays)

    def state_copy(self) -> dict[str, Any]:
        """Return arrays and scalar/flag state for exact read-only audits."""

        return {
            "arrays": tuple(array.copy() for array in self.state_arrays()),
            "server_available": self.server_available.copy(),
            "server_adapter_available": self.server_adapter_available.copy(),
            "round_index": int(self.round_index),
            "last_participating_clients": tuple(self.last_participating_clients),
            "last_aggregation_clients": tuple(self.last_aggregation_clients),
            "last_update_trace": tuple(self.last_update_trace),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def state_unchanged(self, snapshot: Any) -> bool:
        """Compare query state, including adapters, flags, and round index."""

        current = self.state_arrays()
        if isinstance(snapshot, dict):
            arrays = tuple(snapshot["arrays"])
            if int(self.round_index) != int(snapshot["round_index"]):
                return False
            if not np.array_equal(self.server_available, snapshot["server_available"]):
                return False
            if not np.array_equal(self.server_adapter_available, snapshot["server_adapter_available"]):
                return False
            if tuple(self.last_participating_clients) != tuple(snapshot["last_participating_clients"]):
                return False
            if tuple(self.last_aggregation_clients) != tuple(snapshot["last_aggregation_clients"]):
                return False
            if tuple(self.last_update_trace) != tuple(snapshot["last_update_trace"]):
                return False
            if repr(self.rng.bit_generator.state) != repr(snapshot["rng_state"]):
                return False
        else:
            arrays = tuple(snapshot)
        return len(current) == len(arrays) and all(
            np.array_equal(before, after) for before, after in zip(arrays, current)
        )

    def _adapter_features(self, q_local: Any, q_server: Any) -> np.ndarray:
        """Build factor-only ``[q_local; q_server-q_local]`` input."""

        q_local = np.asarray(q_local, dtype=np.float32).reshape(-1)
        q_server = np.asarray(q_server, dtype=np.float32).reshape(-1)
        if q_local.shape != (self.rank,) or q_server.shape != q_local.shape:
            raise ValueError("adapter quality factors must have rank entries")
        features = np.concatenate((q_local, q_server - q_local)).astype(np.float32)
        if not np.all(np.isfinite(features)):
            raise ValueError("adapter features must be finite")
        return features

    def _adapter_forward(
        self,
        client_id: int,
        q_local: Any,
        q_server: Any,
        *,
        with_cache: bool = False,
    ) -> tuple[float, tuple[list[np.ndarray], list[np.ndarray]] | None]:
        """Run one client's private MLP and optionally return backprop cache."""

        client_id = self._validate_client(client_id)
        activation = self._adapter_features(q_local, q_server)
        activations = [activation]
        preactivations: list[np.ndarray] = []
        for layer_index in range(len(self.adapter_weights)):
            weight = self.adapter_weights[layer_index][client_id]
            bias = self.adapter_biases[layer_index][client_id]
            preactivation = (activation @ weight + bias).astype(np.float32)
            preactivations.append(preactivation)
            if layer_index == len(self.adapter_weights) - 1:
                activation = sigmoid(preactivation)
            else:
                activation = np.maximum(preactivation, 0.0).astype(np.float32)
            activations.append(activation)
        omega = float(np.clip(activation[0], 0.0, 1.0))
        cache = (activations, preactivations) if with_cache else None
        return omega, cache

    def _adapter_forward_batch(
        self,
        client_ids: np.ndarray,
        q_local: np.ndarray,
        q_server: np.ndarray,
    ) -> np.ndarray:
        """Vectorized read-only MLP forward for query prediction."""

        client_ids = np.asarray(client_ids, dtype=np.int64)
        q_local = np.asarray(q_local, dtype=np.float32)
        q_server = np.asarray(q_server, dtype=np.float32)
        if q_local.ndim != 2 or q_local.shape[1] != self.rank or q_server.shape != q_local.shape:
            raise ValueError("batched adapter factors have an invalid shape")
        activation = np.concatenate((q_local, q_server - q_local), axis=1).astype(np.float32)
        for layer_index in range(len(self.adapter_weights)):
            weights = self.adapter_weights[layer_index][client_ids]
            biases = self.adapter_biases[layer_index][client_ids]
            activation = np.einsum("ni,nio->no", activation, weights).astype(np.float32) + biases
            if layer_index == len(self.adapter_weights) - 1:
                activation = sigmoid(activation)
            else:
                activation = np.maximum(activation, 0.0).astype(np.float32)
        return np.clip(activation[:, 0], 0.0, 1.0).astype(np.float32)

    def adapter_omega(self, client_id: int, q_local: Any, q_server: Any) -> float:
        """Return one client's private MLP merge gate."""

        return self._adapter_forward(client_id, q_local, q_server)[0]

    @staticmethod
    def adaptive_local_server_merge(local: Any, target: Any, omega: float) -> np.ndarray:
        """Merge local and server rows using a bounded adapter gate."""

        local = np.asarray(local, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if local.shape != target.shape:
            raise ValueError("local and target states must have equal shapes")
        if not np.isfinite(float(omega)) or not 0.0 <= float(omega) <= 1.0:
            raise ValueError("omega must be in [0, 1]")
        result = ((1.0 - np.float32(omega)) * local + np.float32(omega) * target).astype(np.float32)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("adaptive merge produced non-finite values")
        return result

    @staticmethod
    def _adam_update(
        parameter: np.ndarray,
        gradient: np.ndarray,
        first_moment: np.ndarray,
        second_moment: np.ndarray,
        step: int,
        learning_rate: float,
    ) -> None:
        """Apply one standard Adam update in-place."""

        gradient = np.asarray(gradient, dtype=np.float32)
        first_moment *= np.float32(0.9)
        first_moment += np.float32(0.1) * gradient
        second_moment *= np.float32(0.999)
        second_moment += np.float32(0.001) * gradient * gradient
        bias_first = 1.0 - 0.9 ** int(step)
        bias_second = 1.0 - 0.999 ** int(step)
        first_hat = first_moment.astype(np.float64) / bias_first
        second_hat = second_moment.astype(np.float64) / bias_second
        parameter -= np.float32(learning_rate) * (
            first_hat / (np.sqrt(second_hat) + 1e-8)
        ).astype(np.float32)

    def _adapter_backward(
        self,
        client_id: int,
        cache: tuple[list[np.ndarray], list[np.ndarray]],
        dloss_domega: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Backpropagate a scalar support loss through the private MLP."""

        activations, preactivations = cache
        grad = np.asarray([
            np.float32(dloss_domega) * np.float32(
                activations[-1][0] * (1.0 - activations[-1][0])
            )
        ], dtype=np.float32)
        weight_gradients: list[np.ndarray] = [
            np.zeros_like(array[client_id]) for array in self.adapter_weights
        ]
        bias_gradients: list[np.ndarray] = [
            np.zeros_like(array[client_id]) for array in self.adapter_biases
        ]
        for layer_index in range(len(self.adapter_weights) - 1, -1, -1):
            weight_gradients[layer_index] = np.outer(
                activations[layer_index], grad
            ).astype(np.float32)
            bias_gradients[layer_index] = grad.copy()
            if layer_index > 0:
                grad = (grad @ self.adapter_weights[layer_index][client_id].T).astype(np.float32)
                grad *= (preactivations[layer_index - 1] > 0.0).astype(np.float32)
        return weight_gradients, bias_gradients

    def _validate_client(self, client_id: Any) -> int:
        if isinstance(client_id, bool) or int(client_id) != client_id:
            raise ValueError("client_id must be an integer")
        client_id = int(client_id)
        if client_id < 0 or client_id >= self.num_users:
            raise IndexError("client_id out of range")
        return client_id

    def _validate_support(
        self,
        client_id: int,
        users: Any,
        services: Any,
        times: Any,
        targets: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        times = np.asarray(times, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float32)
        if any(array.ndim != 1 for array in (users, services, times, targets)):
            raise ValueError("support arrays must be one-dimensional")
        if not (users.shape == services.shape == times.shape == targets.shape):
            raise ValueError("support arrays must have identical shapes")
        if users.size == 0:
            raise ValueError("a participating client must have support entries")
        if np.any(users != int(client_id)):
            raise ValueError("user-level client support must contain only its own user")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("service index out of range")
        if np.any(times < 0) or np.any(times >= self.num_times):
            raise IndexError("time index out of range")
        if np.any(times != times[0]):
            raise ValueError("update_client batches must contain exactly one time slice")
        if len(np.unique(services)) != services.size:
            raise ValueError("support must contain at most one value per service per slice")
        if not np.all(np.isfinite(targets)) or np.any(targets < 0.0) or np.any(targets > 1.0):
            raise ValueError("normalized support targets must lie in [0, 1]")
        return users, services, times, targets

    def _server_factor_for(self, client_id: int, service: int) -> np.ndarray:
        return self.server_service_factors_by_client[client_id, service]

    def _server_bias_for(self, client_id: int, service: int) -> np.float32:
        return np.float32(self.server_service_biases_by_client[client_id, service])

    def _effective_one(
        self,
        client_id: int,
        service: int,
        local_factor: Any | None = None,
        local_bias: Any | None = None,
    ) -> tuple[np.ndarray, np.float32, float]:
        """Return effective factor, bias, and gate without mutating state."""

        if local_factor is None:
            local_factor = self.local_service_factors[client_id, service]
        if local_bias is None:
            local_bias = self.local_service_biases[client_id, service]
        local_factor = np.asarray(local_factor, dtype=np.float32).copy()
        local_bias = np.float32(local_bias)
        if not self.server_available[client_id]:
            return local_factor, local_bias, 0.0
        server_factor = self._server_factor_for(client_id, service)
        omega = self.adapter_omega(client_id, local_factor, server_factor)
        effective_factor = self.adaptive_local_server_merge(local_factor, server_factor, omega)
        effective_bias = np.float32(
            (1.0 - np.float32(omega)) * local_bias
            + np.float32(omega) * self._server_bias_for(client_id, service)
        )
        return effective_factor, effective_bias, omega

    def _predict_one(self, client_id: int, service: int) -> np.float32:
        effective_factor, effective_bias, _ = self._effective_one(client_id, service)
        logit = np.dot(self.private_user_factors[client_id], effective_factor)
        logit += self.private_user_biases[client_id] + effective_bias
        return np.float32(sigmoid(np.asarray([logit]))[0])

    def update_client(
        self,
        client: int,
        users: Any,
        services: Any,
        times: Any,
        targets: Any,
        epochs: int = 1,
    ) -> FedEMClientUpdate:
        """Run official-order adapter then embedding Adam on one support slice."""

        client_id = self._validate_client(client)
        if isinstance(epochs, bool) or int(epochs) != epochs or int(epochs) <= 0:
            raise ValueError("epochs must be a positive integer")
        epochs = int(epochs)
        users, services, times, targets = self._validate_support(
            client_id, users, services, times, targets
        )
        order = np.argsort(services, kind="mergesort")
        signature = np.zeros(self.rank + 1, dtype=np.float32)
        trace: list[str] = []

        # The reference creates a fresh Adam optimizer for each local client
        # round.  Moments therefore persist across support steps/epochs in this
        # call but reset at the next call.
        adapter_first = [np.zeros_like(array[client_id]) for array in self.adapter_weights]
        adapter_second = [np.zeros_like(array[client_id]) for array in self.adapter_weights]
        adapter_bias_first = [np.zeros_like(array[client_id]) for array in self.adapter_biases]
        adapter_bias_second = [np.zeros_like(array[client_id]) for array in self.adapter_biases]
        user_first = np.zeros_like(self.private_user_factors[client_id])
        user_second = np.zeros_like(self.private_user_factors[client_id])
        user_bias_first = np.zeros((), dtype=np.float32)
        user_bias_second = np.zeros((), dtype=np.float32)
        service_first = np.zeros_like(self.local_service_factors[client_id, 0])
        service_second = np.zeros_like(self.local_service_factors[client_id, 0])
        service_bias_first = np.zeros((), dtype=np.float32)
        service_bias_second = np.zeros((), dtype=np.float32)
        adapter_step = 0
        embedding_step = 0

        for _ in range(epochs):
            for position in order:
                service = int(services[position])
                target = np.float32(targets[position])
                before_user = self.private_user_factors[client_id].copy()
                before_user_bias = np.float32(self.private_user_biases[client_id])
                before_service = self.local_service_factors[client_id, service].copy()
                before_service_bias = np.float32(self.local_service_biases[client_id, service])
                has_server = bool(self.server_available[client_id])
                q_server = self._server_factor_for(client_id, service).copy() if has_server else None

                # Stage 1: private adapter Adam step through the support loss.
                if has_server:
                    q_local = before_service
                    omega_before, cache = self._adapter_forward(
                        client_id, q_local, q_server, with_cache=True
                    )
                    effective_before = self.adaptive_local_server_merge(
                        q_local, q_server, omega_before
                    )
                    effective_bias_before = np.float32(
                        (1.0 - np.float32(omega_before)) * before_service_bias
                        + np.float32(omega_before) * self._server_bias_for(client_id, service)
                    )
                    prediction_before = np.float32(sigmoid(np.asarray([
                        np.dot(before_user, effective_before)
                        + before_user_bias
                        + effective_bias_before
                    ]))[0])
                    residual_before = np.float32(prediction_before - target)
                    gradient_before = np.float32(
                        residual_before * prediction_before * (1.0 - prediction_before)
                    )
                    delta_factor = q_server - q_local
                    dlogit_domega = np.float32(np.dot(before_user, delta_factor))
                    dloss_domega = np.float32(gradient_before * dlogit_domega)
                    weight_gradients, bias_gradients = self._adapter_backward(
                        client_id, cache, dloss_domega  # type: ignore[arg-type]
                    )
                    for layer_index in range(len(self.adapter_weights)):
                        weight_gradients[layer_index] += np.float32(self.adapter_regularization) * self.adapter_weights[layer_index][client_id]
                        bias_gradients[layer_index] += np.float32(self.adapter_regularization) * self.adapter_biases[layer_index][client_id]
                    adapter_step += 1
                    trace.append("adapter")
                    for layer_index in range(len(self.adapter_weights)):
                        self._adam_update(
                            self.adapter_weights[layer_index][client_id],
                            weight_gradients[layer_index],
                            adapter_first[layer_index],
                            adapter_second[layer_index],
                            adapter_step,
                            self.adapter_learning_rate,
                        )
                        self._adam_update(
                            self.adapter_biases[layer_index][client_id],
                            bias_gradients[layer_index],
                            adapter_bias_first[layer_index],
                            adapter_bias_second[layer_index],
                            adapter_step,
                            self.adapter_learning_rate,
                        )

                # Stage 2: recompute the gate with updated adapter and update
                # user/service embeddings with a separate Adam optimizer.
                if has_server:
                    omega, _ = self._adapter_forward(client_id, before_service, q_server)
                    effective_factor = self.adaptive_local_server_merge(
                        before_service, q_server, omega
                    )
                    effective_bias = np.float32(
                        (1.0 - np.float32(omega)) * before_service_bias
                        + np.float32(omega) * self._server_bias_for(client_id, service)
                    )
                else:
                    omega = 0.0
                    effective_factor = before_service
                    effective_bias = before_service_bias
                prediction = np.float32(sigmoid(np.asarray([
                    np.dot(before_user, effective_factor) + before_user_bias + effective_bias
                ]))[0])
                residual = np.float32(prediction - target)
                gradient = np.float32(residual * prediction * (1.0 - prediction))
                signature[: self.rank] += residual * before_service
                signature[-1] += residual

                user_gradient = (
                    gradient * effective_factor
                    + np.float32(self.regularization) * before_user
                ).astype(np.float32)
                user_bias_gradient = np.float32(
                    gradient + np.float32(self.regularization) * before_user_bias
                )
                service_gradient = (
                    gradient * np.float32(1.0 - omega) * before_user
                    + np.float32(self.regularization) * before_service
                ).astype(np.float32)
                service_bias_gradient = np.float32(
                    gradient * np.float32(1.0 - omega)
                    + np.float32(self.regularization) * before_service_bias
                )
                embedding_step += 1
                trace.append("embedding")
                self._adam_update(
                    self.private_user_factors[client_id], user_gradient,
                    user_first, user_second, embedding_step, self.learning_rate,
                )
                self._adam_update(
                    self.private_user_biases[client_id:client_id + 1],
                    np.asarray([user_bias_gradient], dtype=np.float32),
                    user_bias_first.reshape(1), user_bias_second.reshape(1),
                    embedding_step, self.learning_rate,
                )
                self._adam_update(
                    self.local_service_factors[client_id, service], service_gradient,
                    service_first, service_second, embedding_step, self.learning_rate,
                )
                self._adam_update(
                    self.local_service_biases[client_id, service:service + 1],
                    np.asarray([service_bias_gradient], dtype=np.float32),
                    service_bias_first.reshape(1), service_bias_second.reshape(1),
                    embedding_step, self.learning_rate,
                )

        self.last_update_trace = tuple(trace)
        if not all(np.all(np.isfinite(array)) for array in self.state_arrays()):
            raise FloatingPointError("FedEM local update produced non-finite state")
        support_entries = int(users.size * epochs)
        signature /= np.float32(max(1, support_entries))
        service_indices = np.arange(self.num_services, dtype=np.int64)
        support_counts = np.zeros(self.num_services, dtype=np.int32)
        support_counts[services] = 1
        adapter_weights = tuple(
            self.adapter_weights[layer_index][client_id].copy()
            for layer_index in range(len(self.adapter_weights))
        )
        adapter_biases = tuple(
            self.adapter_biases[layer_index][client_id].copy()
            for layer_index in range(len(self.adapter_biases))
        )
        adapter_bytes = int(sum(array.size for array in adapter_weights)
                            + sum(array.size for array in adapter_biases)) * FP32_BYTES
        dense_service_bytes = int(self.num_services * (self.rank + 1) * FP32_BYTES)
        # Dense service factors make indices implicit; service_indices_bytes is
        # deliberately zero under official dense accounting.
        return FedEMClientUpdate(
            client_id=client_id,
            service_indices=service_indices,
            service_factors=self.local_service_factors[client_id].copy(),
            service_biases=self.local_service_biases[client_id].copy(),
            support_counts=support_counts,
            support_entries=support_entries,
            support_signature=signature.copy(),
            adapter_weights=adapter_weights,
            adapter_biases=adapter_biases,
            service_indices_bytes=0,
            adapter_bytes=adapter_bytes,
            uploaded_bytes=int(dense_service_bytes + adapter_bytes),
            downloaded_bytes=dense_service_bytes,
        )

    def _coerce_adapter_payload(
        self,
        client_id: int,
        weights: Any,
        biases: Any,
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        try:
            weights = tuple(weights)
            biases = tuple(biases)
        except TypeError as exc:
            raise ValueError("adapter payload must contain layer arrays") from exc
        if len(weights) != len(self.adapter_weights) or len(biases) != len(self.adapter_biases):
            raise ValueError("adapter layer count mismatch")
        copied_weights: list[np.ndarray] = []
        copied_biases: list[np.ndarray] = []
        for layer_index in range(len(self.adapter_weights)):
            weight = _as_float32(weights[layer_index])
            bias = _as_float32(biases[layer_index])
            expected_weight_shape = self.adapter_weights[layer_index][client_id].shape
            expected_bias_shape = self.adapter_biases[layer_index][client_id].shape
            if weight.shape != expected_weight_shape or bias.shape != expected_bias_shape:
                raise ValueError("adapter layer shape mismatch")
            copied_weights.append(weight)
            copied_biases.append(bias)
        return tuple(copied_weights), tuple(copied_biases)

    def _coerce_update(self, update: Any) -> FedEMClientUpdate:
        if isinstance(update, FedEMClientUpdate):
            result = update
        elif isinstance(update, Mapping):
            required = (
                "client_id", "service_indices", "service_factors", "service_biases",
                "support_counts", "support_signature",
            )
            missing = [name for name in required if name not in update]
            if missing:
                raise ValueError("client update is missing: " + ", ".join(missing))
            client_id = self._validate_client(update["client_id"])
            service_indices = np.asarray(update["service_indices"], dtype=np.int64)
            service_factors = _as_float32(update["service_factors"])
            service_biases = _as_float32(update["service_biases"])
            support_counts = np.asarray(update["support_counts"], dtype=np.int32)
            support_signature = _as_float32(update["support_signature"])
            default_weights = tuple(array[client_id].copy() for array in self.adapter_weights)
            default_biases = tuple(array[client_id].copy() for array in self.adapter_biases)
            adapter_weights, adapter_biases = self._coerce_adapter_payload(
                client_id,
                update.get("adapter_weights", default_weights),
                update.get("adapter_biases", default_biases),
            )
            adapter_bytes = int(sum(array.size for array in adapter_weights)
                                + sum(array.size for array in adapter_biases)) * FP32_BYTES
            dense_service_bytes = int(self.num_services * (self.rank + 1) * FP32_BYTES)
            result = FedEMClientUpdate(
                client_id=client_id,
                service_indices=service_indices,
                service_factors=service_factors,
                service_biases=service_biases,
                support_counts=support_counts,
                support_entries=int(update.get("support_entries", int(np.sum(support_counts)))),
                support_signature=support_signature,
                adapter_weights=adapter_weights,
                adapter_biases=adapter_biases,
                service_indices_bytes=int(update.get("service_indices_bytes", 0)),
                adapter_bytes=int(update.get("adapter_bytes", adapter_bytes)),
                uploaded_bytes=int(update.get("uploaded_bytes", dense_service_bytes + adapter_bytes)),
                downloaded_bytes=int(update.get("downloaded_bytes", dense_service_bytes)),
            )
        else:
            raise TypeError("updates must be FedEMClientUpdate objects or mappings")

        client_id = self._validate_client(result.client_id)
        services = np.asarray(result.service_indices, dtype=np.int64)
        factors = np.asarray(result.service_factors, dtype=np.float32)
        biases = np.asarray(result.service_biases, dtype=np.float32)
        counts = np.asarray(result.support_counts, dtype=np.int32)
        signature = np.asarray(result.support_signature, dtype=np.float32)
        expected_indices = np.arange(self.num_services, dtype=np.int64)
        if not np.array_equal(services, expected_indices):
            raise ValueError("official dense service payload must contain all ordered service indices")
        if factors.shape != (self.num_services, self.rank):
            raise ValueError("dense service_factors shape mismatch")
        if biases.shape != (self.num_services,) or counts.shape != (self.num_services,):
            raise ValueError("dense service update shape mismatch")
        if signature.shape != (self.rank + 1,):
            raise ValueError("support_signature shape mismatch")
        if np.any(counts < 0) or int(result.support_entries) <= 0:
            raise ValueError("support counts must be non-negative and support entries positive")
        adapter_weights, adapter_biases = self._coerce_adapter_payload(
            client_id, result.adapter_weights, result.adapter_biases
        )
        expected_adapter_bytes = int(sum(array.size for array in adapter_weights)
                                     + sum(array.size for array in adapter_biases)) * FP32_BYTES
        dense_service_bytes = int(self.num_services * (self.rank + 1) * FP32_BYTES)
        if int(result.service_indices_bytes) != 0:
            raise ValueError("dense service payload must not carry sparse index metadata")
        if int(result.adapter_bytes) != expected_adapter_bytes:
            raise ValueError("adapter_bytes does not match the private MLP payload")
        if int(result.uploaded_bytes) != dense_service_bytes + expected_adapter_bytes:
            raise ValueError("uploaded_bytes does not match dense official accounting")
        if int(result.downloaded_bytes) != dense_service_bytes:
            raise ValueError("downloaded_bytes does not match dense official accounting")
        if not np.all(np.isfinite(factors)) or not np.all(np.isfinite(biases)):
            raise ValueError("service update values must be finite")
        if not np.all(np.isfinite(signature)):
            raise ValueError("support signature must be finite")
        return FedEMClientUpdate(
            client_id=client_id,
            service_indices=services.copy(),
            service_factors=factors.copy(),
            service_biases=biases.copy(),
            support_counts=counts.copy(),
            support_entries=int(result.support_entries),
            support_signature=signature.copy(),
            adapter_weights=tuple(array.copy() for array in adapter_weights),
            adapter_biases=tuple(array.copy() for array in adapter_biases),
            service_indices_bytes=0,
            adapter_bytes=expected_adapter_bytes,
            uploaded_bytes=dense_service_bytes + expected_adapter_bytes,
            downloaded_bytes=dense_service_bytes,
        )

    @staticmethod
    def _pca_first_component(service_factors: Any) -> np.ndarray:
        """Return the first centered PCA score for every service row."""

        service_factors = np.asarray(service_factors, dtype=np.float64)
        if service_factors.ndim != 2 or service_factors.shape[0] == 0:
            raise ValueError("service_factors must be a non-empty matrix")
        if not np.all(np.isfinite(service_factors)):
            raise ValueError("service_factors must be finite")
        centered = service_factors - np.mean(service_factors, axis=0, keepdims=True)
        _, _, right_singular = np.linalg.svd(centered, full_matrices=False)
        return (centered @ right_singular[:1].T).reshape(-1).astype(np.float32)

    @classmethod
    def similarity_from_factors(cls, factors: Any, sim_mode: str = "l2") -> np.ndarray:
        """Compute the official PCA-first-component similarity matrix."""

        factors = np.asarray(factors, dtype=np.float32)
        if factors.ndim != 3 or factors.shape[0] == 0:
            raise ValueError("factors must have shape [clients, services, rank]")
        projections = np.asarray([
            cls._pca_first_component(client_factors) for client_factors in factors
        ], dtype=np.float64)
        if sim_mode == "l2":
            norm_sq = np.sum(projections * projections, axis=1, keepdims=True)
            distances = np.sqrt(np.maximum(norm_sq + norm_sq.T - 2.0 * projections @ projections.T, 0.0))
            similarity = 1.0 / (distances + 1.0)
        elif sim_mode == "cos":
            norms = np.linalg.norm(projections, axis=1, keepdims=True)
            normalized = projections / np.maximum(norms, 1e-12)
            similarity = np.maximum(normalized @ normalized.T, 0.0)
        else:
            raise ValueError("sim_mode must be one of l2, cos")
        similarity[similarity > 0.9] = 1.0
        return similarity.astype(np.float32)

    def _similarity_matrix(self, updates: list[FedEMClientUpdate]) -> np.ndarray:
        factors = np.asarray([update.service_factors for update in updates], dtype=np.float32)
        return self.similarity_from_factors(factors, self.sim_mode)

    def _aggregation_weights(
        self,
        updates: list[FedEMClientUpdate],
        similarity: np.ndarray,
    ) -> np.ndarray:
        sample_counts = np.asarray(
            [float(update.support_entries) for update in updates], dtype=np.float64
        )
        total_samples = float(np.sum(sample_counts))
        if not np.isfinite(total_samples) or total_samples <= 0.0:
            raise FloatingPointError("invalid client sample count")
        proportions = sample_counts / total_samples
        n_clients = len(updates)
        if self.agg_mode == "avg":
            matrix = np.broadcast_to(proportions[None, :], (n_clients, n_clients)).copy()
        elif self.agg_mode == "sim":
            matrix = np.asarray(similarity, dtype=np.float64).copy()
            row_sums = np.sum(matrix, axis=1, keepdims=True)
            matrix = np.divide(
                matrix, row_sums, out=np.full_like(matrix, 1.0 / n_clients), where=row_sums > 0.0
            )
        else:
            # Official QP: min ||x||^2 + (2 alpha s - 2 p)^T x,
            # s = -similarity. Thus x = Proj_simplex(p + alpha*similarity).
            matrix = np.asarray([
                simplex_projection(proportions + self.alpha * similarity[row])
                for row in range(n_clients)
            ], dtype=np.float32)
        matrix[np.abs(matrix) < 1e-20] = 0.0
        row_sums = np.sum(matrix, axis=1, keepdims=True)
        matrix = np.divide(
            matrix, row_sums, out=np.full_like(matrix, 1.0 / n_clients), where=row_sums > 0.0
        )
        return matrix.astype(np.float32)

    def _aggregate_services(
        self,
        updates: list[FedEMClientUpdate],
        aggregation_positions: Any | None = None,
    ) -> None:
        """Aggregate dense service factors for a random official subset."""

        active = [self._coerce_update(update) for update in updates]
        if not active:
            raise ValueError("at least one update is required")
        if aggregation_positions is None:
            aggregation_positions = np.arange(len(active), dtype=np.int64)
        aggregation_positions = np.asarray(aggregation_positions, dtype=np.int64).reshape(-1)
        if aggregation_positions.size == 0 or np.any(aggregation_positions < 0) or np.any(aggregation_positions >= len(active)):
            raise ValueError("aggregation_positions must select at least one update")
        if len(np.unique(aggregation_positions)) != aggregation_positions.size:
            raise ValueError("aggregation_positions must be unique")
        selected = [active[int(position)] for position in aggregation_positions]
        selected_ids = tuple(int(update.client_id) for update in selected)

        self.last_support_counts.fill(0)
        self.last_service_support_counts.fill(0)
        for update in active:
            self.last_support_counts[update.client_id] = int(update.support_entries)
            self.last_service_support_counts[update.client_id] = update.support_counts
        similarity = self._similarity_matrix(selected)
        matrix = self._aggregation_weights(selected, similarity)
        self.last_similarity_matrix.fill(0.0)
        self.last_aggregation_matrix.fill(0.0)
        selected_ids_array = np.asarray(selected_ids, dtype=np.int64)
        self.last_similarity_matrix[np.ix_(selected_ids_array, selected_ids_array)] = similarity
        self.last_aggregation_matrix[np.ix_(selected_ids_array, selected_ids_array)] = matrix

        selected_factors = np.asarray([update.service_factors for update in selected], dtype=np.float32)
        selected_biases = np.asarray([update.service_biases for update in selected], dtype=np.float32)
        proportions = np.asarray(
            [float(update.support_entries) for update in selected], dtype=np.float64
        )
        proportions /= float(np.sum(proportions))
        self.server_service_factors = np.tensordot(
            proportions.astype(np.float32), selected_factors, axes=(0, 0)
        ).astype(np.float32)
        self.server_service_biases = np.tensordot(
            proportions.astype(np.float32), selected_biases, axes=(0, 0)
        ).astype(np.float32)

        # Only active clients not selected this round lose the current
        # personalized server model.  Inactive clients' state remains intact.
        active_ids = np.asarray([update.client_id for update in active], dtype=np.int64)
        self.server_available[active_ids] = False
        self.server_adapter_available[active_ids] = False
        for target_position, target_id in enumerate(selected_ids):
            self.server_service_factors_by_client[target_id] = np.tensordot(
                matrix[target_position], selected_factors, axes=(0, 0)
            ).astype(np.float32)
            self.server_service_biases_by_client[target_id] = np.tensordot(
                matrix[target_position], selected_biases, axes=(0, 0)
            ).astype(np.float32)
            self.server_available[target_id] = True
            source_update = selected[target_position]
            for layer_index in range(len(self.server_adapter_weights)):
                self.server_adapter_weights[layer_index][target_id] = source_update.adapter_weights[layer_index]
                self.server_adapter_biases[layer_index][target_id] = source_update.adapter_biases[layer_index]
            self.server_adapter_available[target_id] = True

        self.last_aggregation_clients = selected_ids
        self.round_index += 1
        if not all(np.all(np.isfinite(array)) for array in self.state_arrays()):
            raise FloatingPointError("FedEM aggregation produced non-finite state")

    def sync_round(
        self,
        batches: Mapping[Any, Any] | Iterable[Any],
        *,
        epochs: int = 1,
    ) -> dict[str, Any]:
        """Run one causal local round and a random official aggregation subset."""

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

        parsed: list[tuple[Any, Any]] = []
        clients: set[int] = set()
        for client, batch in records:
            client_id = self._validate_client(client)
            if client_id in clients:
                raise ValueError("client batches must have unique client ids")
            clients.add(client_id)
            if isinstance(batch, Mapping):
                aliases = {
                    "users": ("users", "user_indices"),
                    "services": ("services", "service_indices"),
                    "times": ("times", "time_indices", "time_slices"),
                    "targets": ("targets", "values", "support_values"),
                }
                resolved = {}
                for name, names in aliases.items():
                    for candidate in names:
                        if candidate in batch:
                            resolved[name] = batch[candidate]
                            break
                    if name not in resolved:
                        raise ValueError(f"client batch is missing {name}")
                parsed.append((client_id, (resolved["users"], resolved["services"], resolved["times"], resolved["targets"])))
            else:
                try:
                    users, services, times, targets = batch
                except (TypeError, ValueError) as exc:
                    raise ValueError("each client batch must be (users, services, times, targets)") from exc
                parsed.append((client_id, (users, services, times, targets)))

        participant_count = max(1, int(len(parsed) * self.clients_sample_ratio))
        if participant_count < len(parsed):
            chosen = np.sort(self.rng.choice(len(parsed), size=participant_count, replace=False))
            parsed = [parsed[int(index)] for index in chosen]
        self.last_participating_clients = tuple(int(client) for client, _ in parsed)

        updates = [
            self.update_client(client_id, batch[0], batch[1], batch[2], batch[3], epochs=epochs)
            for client_id, batch in parsed
        ]
        aggregation_count = max(1, int(len(updates) * self.agg_clients_ratio))
        if aggregation_count < len(updates):
            aggregation_positions = np.sort(
                self.rng.choice(len(updates), size=aggregation_count, replace=False)
            )
        else:
            aggregation_positions = np.arange(len(updates), dtype=np.int64)
        self._aggregate_services(updates, aggregation_positions=aggregation_positions)
        return {
            "updates": tuple(updates),
            "participating_clients": len(updates),
            "aggregation_clients": int(aggregation_positions.size),
            "aggregation_client_ids": tuple(
                int(updates[int(position)].client_id) for position in aggregation_positions
            ),
            "uplink_bytes": int(sum(item.uploaded_bytes for item in updates)),
            "downlink_bytes": int(sum(
                updates[int(position)].downloaded_bytes for position in aggregation_positions
            )),
            "similarity_matrix": self.last_similarity_matrix.copy(),
            "aggregation_matrix": self.last_aggregation_matrix.copy(),
            "round_index": int(self.round_index),
        }

    def predict(
        self,
        users: Any,
        services: Any,
        times: Any,
        client_ids: Any | None = None,
    ) -> np.ndarray:
        """Predict normalized QoS without changing any model state."""

        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        times = np.asarray(times, dtype=np.int64)
        if users.ndim != 1 or users.shape != services.shape or users.shape != times.shape:
            raise ValueError("prediction indices must be aligned one-dimensional arrays")
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise IndexError("user index out of range")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("service index out of range")
        if np.any(times < 0) or np.any(times >= self.num_times):
            raise IndexError("time index out of range")
        if client_ids is not None:
            client_ids = np.asarray(client_ids, dtype=np.int64)
            if client_ids.shape != users.shape or np.any(client_ids != users):
                raise ValueError("user-level client_ids must equal users")
        if users.size == 0:
            return np.empty(0, dtype=np.float32)

        local_factors = self.local_service_factors[users, services].copy()
        local_biases = self.local_service_biases[users, services].copy()
        effective_factors = local_factors.copy()
        effective_biases = local_biases.copy()
        ready = self.server_available[users]
        ready_positions = np.flatnonzero(ready)
        if ready_positions.size:
            ready_users = users[ready_positions]
            server_factors = self.server_service_factors_by_client[ready_users, services[ready_positions]]
            server_biases = self.server_service_biases_by_client[ready_users, services[ready_positions]]
            omegas = self._adapter_forward_batch(
                ready_users,
                local_factors[ready_positions],
                server_factors,
            )
            effective_factors[ready_positions] = (
                local_factors[ready_positions]
                + omegas[:, None] * (server_factors - local_factors[ready_positions])
            ).astype(np.float32)
            effective_biases[ready_positions] = (
                (1.0 - omegas) * local_biases[ready_positions]
                + omegas * server_biases
            ).astype(np.float32)
        logits = np.sum(self.private_user_factors[users] * effective_factors, axis=-1)
        logits += self.private_user_biases[users] + effective_biases
        return sigmoid(logits)

    def reconstruction_loss(
        self,
        users: Any,
        services: Any,
        times: Any,
        targets: Any,
    ) -> float:
        """Return support/query loss without mutating state."""

        targets = np.asarray(targets, dtype=np.float32)
        predictions = self.predict(users, services, times)
        if targets.shape != predictions.shape or targets.size == 0:
            raise ValueError("targets must be non-empty and aligned with predictions")
        if not np.all(np.isfinite(targets)):
            raise ValueError("targets must be finite")
        residual = predictions.astype(np.float64) - targets.astype(np.float64)
        return float(np.mean(residual * residual))


FedEMModel = FedEMQoSModel
FedEM = FedEMQoSModel
PersonalizedModelMergingQoS = FedEMQoSModel


__all__ = [
    "FP32_BYTES",
    "INDEX_BYTES",
    "FedEMClientUpdate",
    "FedEMQoSModel",
    "FedEMModel",
    "FedEM",
    "PersonalizedModelMergingQoS",
    "cosine_similarity",
    "simplex_projection",
    "sigmoid",
]
