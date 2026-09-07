"""Causal full-model FedProx for the online QoS baseline.

Each participating client starts from the same complete AMF model as the
server.  The local objective is the plain sigmoid squared-error AMF loss with
L2 regularisation and a FedProx term,

    f_k(theta) + (mu / 2) * ||theta - theta_global||^2.

All four parameter families (user vectors, service vectors, user biases and
service biases) receive the proximal gradient.  Sparse deltas are used only as
an implementation optimisation: untouched rows remain equal to the global
model and the aggregation denominator is still the sum of complete-client
support weights.

The frozen standalone baseline configuration uses ``mu=0.3`` and one local
epoch; ``mu=0`` remains the exact FedAvg-E1 control.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from methods.fedavg import (  # Reuse the frozen complete AMF model contract.
    FullAMFModel,
    PARAMETER_BYTES_FP32,
    PARAMETER_COUNT,
    client_sgd_delta,
    sigmoid,
)


@dataclass(frozen=True)
class FedProxClientDelta:
    """Sparse representation of one complete-model FedProx client model."""

    user: int
    service_indices: np.ndarray
    user_vector_delta: np.ndarray
    service_vector_deltas: np.ndarray
    user_bias_delta: np.float32
    service_bias_deltas: np.ndarray
    support_entries: int
    # Activity records the proximal-gradient term, not the ordinary AMF
    # gradient.  It is retained in memory for the anti-degeneration audit.
    prox_activity: dict[str, dict[str, float]]

    @property
    def prox_activity_total(self) -> float:
        return float(sum(item["norm_sum"] for item in self.prox_activity.values()))


def _activity() -> dict[str, dict[str, float]]:
    return {
        name: {"nonzero_steps": 0, "norm_sum": 0.0}
        for name in ("user_vector", "service_vector", "user_bias", "service_bias")
    }


def _record_activity(activity: dict[str, dict[str, float]], name: str, prox_gradient) -> None:
    norm = float(np.linalg.norm(np.asarray(prox_gradient, dtype=np.float32)))
    activity[name]["norm_sum"] += norm
    if norm > 0.0:
        activity[name]["nonzero_steps"] += 1


def _from_fedavg_delta(base, *, service_indices, prox_activity) -> FedProxClientDelta:
    """Convert a mu=0 FedAvg delta without changing any numerical values."""
    return FedProxClientDelta(
        user=int(base.user),
        service_indices=np.asarray(service_indices, dtype=np.int64).copy(),
        user_vector_delta=np.asarray(base.user_vector_delta, dtype=np.float32).copy(),
        service_vector_deltas=np.asarray(base.service_vector_deltas, dtype=np.float32).copy(),
        user_bias_delta=np.float32(base.user_bias_delta),
        service_bias_deltas=np.asarray(base.service_bias_deltas, dtype=np.float32).copy(),
        support_entries=int(base.support_entries),
        prox_activity=prox_activity,
    )


def client_fedprox_delta(
    global_model: FullAMFModel,
    user: int,
    services,
    targets,
    eta: float = 0.8,
    l2: float = 3e-4,
    mu: float = 0.3,
    local_epochs: int = 1,
) -> FedProxClientDelta:
    """Compute a client update for the complete-model FedProx objective.

    ``targets`` are normalized fixed-log1p QoS observations.  The local model
    is initialized from ``global_model`` for every client.  At every local
    SGD step the proximal gradient is ``mu * (local - global)`` for all four
    parameter families.  With ``mu=0`` this delegates to the frozen FedAvg
    implementation so that the zero-prox limit is numerically identical.
    """
    if not np.isfinite(float(mu)) or float(mu) < 0.0:
        raise ValueError("mu must be finite and non-negative")
    if float(mu) == 0.0:
        base = client_sgd_delta(
            global_model, user, services, targets, eta=eta, l2=l2,
            local_epochs=local_epochs,
        )
        return _from_fedavg_delta(
            base,
            service_indices=np.asarray(services, dtype=np.int64),
            prox_activity=_activity(),
        )

    if not (0 <= int(user) < global_model.user_vectors.shape[0]):
        raise IndexError("user index out of range")
    if eta <= 0 or l2 < 0 or int(local_epochs) <= 0:
        raise ValueError("invalid FedProx optimization parameters")
    service_indices = np.asarray(services, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.float32)
    if service_indices.ndim != 1 or targets.shape != service_indices.shape:
        raise ValueError("services and targets must be aligned vectors")
    if len(service_indices) == 0:
        raise ValueError("a participating client must have support entries")
    if np.any(service_indices[1:] < service_indices[:-1]):
        raise ValueError("services must be supplied in ascending order")
    if len(np.unique(service_indices)) != len(service_indices):
        raise ValueError("services must be unique within a time slice")
    if np.any(service_indices < 0) or np.any(service_indices >= global_model.service_vectors.shape[0]):
        raise IndexError("service index out of range")
    if not np.all(np.isfinite(targets)):
        raise ValueError("targets must be finite")

    # These snapshots are the complete client initialization values.  Only
    # touched rows need to be materialised because all other rows stay equal to
    # the server model and therefore have a zero delta.
    old_user = global_model.user_vectors[user].copy()
    old_user_bias = np.float32(global_model.user_biases[user])
    old_services = global_model.service_vectors[service_indices].copy()
    old_service_biases = global_model.service_biases[service_indices].copy()
    local_user = old_user.copy()
    local_user_bias = np.float32(old_user_bias)
    local_services = old_services.copy()
    local_service_biases = old_service_biases.copy()
    mu32 = np.float32(mu)
    eta32 = np.float32(eta)
    l2_32 = np.float32(l2)
    activity = _activity()

    for _ in range(int(local_epochs)):
        for index, target in enumerate(targets):
            before_user = local_user.copy()
            before_user_bias = np.float32(local_user_bias)
            before_service = local_services[index].copy()
            before_service_bias = np.float32(local_service_biases[index])
            prediction = sigmoid(
                np.dot(before_user, before_service)
                + float(before_user_bias)
                + float(before_service_bias)
            )
            gradient = np.float32(
                (prediction - float(target)) * prediction * (1.0 - prediction)
            )

            # FedProx gradient for each complete parameter family.  The
            # reference values are the corresponding global-model values.
            prox_user = mu32 * (before_user - old_user)
            prox_service = mu32 * (before_service - old_services[index])
            prox_user_bias = mu32 * (before_user_bias - old_user_bias)
            prox_service_bias = mu32 * (before_service_bias - old_service_biases[index])
            _record_activity(activity, "user_vector", prox_user)
            _record_activity(activity, "service_vector", prox_service)
            _record_activity(activity, "user_bias", prox_user_bias)
            _record_activity(activity, "service_bias", prox_service_bias)

            local_user -= eta32 * (gradient * before_service + l2_32 * before_user + prox_user)
            local_services[index] -= eta32 * (
                gradient * before_user + l2_32 * before_service + prox_service
            )
            local_user_bias -= eta32 * (
                gradient + l2_32 * before_user_bias + prox_user_bias
            )
            local_service_biases[index] -= eta32 * (
                gradient + l2_32 * before_service_bias + prox_service_bias
            )

    return FedProxClientDelta(
        user=int(user),
        service_indices=service_indices.copy(),
        user_vector_delta=(local_user - old_user).astype(np.float32),
        service_vector_deltas=(local_services - old_services).astype(np.float32),
        user_bias_delta=np.float32(local_user_bias - old_user_bias),
        service_bias_deltas=(local_service_biases - old_service_biases).astype(np.float32),
        support_entries=int(len(service_indices)),
        prox_activity=activity,
    )


def aggregate_sparse_deltas(
    global_model: FullAMFModel,
    client_deltas: list[FedProxClientDelta],
) -> FullAMFModel:
    """Apply exact sample-weighted full-model FedProx server aggregation.

    Every client model is ``global_model + delta``.  Untouched parameters are
    implicitly present in every complete model, so the denominator is the sum
    of all client support counts rather than a touched-row denominator.
    """
    if not client_deltas:
        raise ValueError("at least one client is required")
    total = sum(int(delta.support_entries) for delta in client_deltas)
    if total <= 0:
        raise ValueError("client support weights must be positive")
    result = global_model.copy()
    for delta in client_deltas:
        weight = np.float64(delta.support_entries) / np.float64(total)
        result.user_vectors[delta.user] += np.float32(weight) * delta.user_vector_delta
        result.user_biases[delta.user] += np.float32(weight) * delta.user_bias_delta
        result.service_vectors[delta.service_indices] += (
            np.float32(weight) * delta.service_vector_deltas
        )
        result.service_biases[delta.service_indices] += (
            np.float32(weight) * delta.service_bias_deltas
        )
    return result


def aggregate_full_models(
    global_model: FullAMFModel,
    client_models: list[FullAMFModel],
    support_counts,
) -> FullAMFModel:
    """Reference complete-tensor weighted average used by semantic tests."""
    if not client_models or len(client_models) != len(support_counts):
        raise ValueError("client models and support counts must be non-empty and aligned")
    counts = np.asarray(support_counts, dtype=np.float64)
    if np.any(counts <= 0) or not np.all(np.isfinite(counts)):
        raise ValueError("support counts must be finite and positive")
    weights = counts / counts.sum()
    if any(model.parameter_count != global_model.parameter_count for model in client_models):
        raise ValueError("client model shape mismatch")
    arrays = []
    for index in range(4):
        arrays.append(np.asarray(
            sum((weight * model.state_arrays()[index] for weight, model in zip(weights, client_models))),
            dtype=np.float32,
        ))
    return FullAMFModel(*arrays)
