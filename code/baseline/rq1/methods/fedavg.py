"""Causal full-model FedAvg for the online QoS baseline.

The baseline deliberately keeps the model and the aggregation rule simple:
every participating client starts a local step from the same complete model
and the server averages all four parameter tensors with the client's support
count as its weight.  The implementation uses sparse deltas internally, but
the untouched rows are implicitly present in every client's complete model;
there is therefore no service-wise re-normalisation or touched-row-only
denominator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PARAMETER_COUNT = 142 * 10 + 4500 * 10 + 142 + 4500
PARAMETER_BYTES_FP32 = PARAMETER_COUNT * 4


def sigmoid(value: float) -> float:
    """Numerically bounded logistic function used by the plain AMF SGD."""
    return float(1.0 / (1.0 + np.exp(-np.clip(float(value), -30.0, 30.0))))


@dataclass(frozen=True)
class FedAvgClientDelta:
    """Sparse representation of one complete-model client update.

    A client model is ``global + delta``.  The omitted rows have exactly zero
    delta because the client started from the global model and only its
    support tuples were consumed.
    """

    user: int
    service_indices: np.ndarray
    user_vector_delta: np.ndarray
    service_vector_deltas: np.ndarray
    user_bias_delta: np.float32
    service_bias_deltas: np.ndarray
    support_entries: int


class FullAMFModel:
    """Homogeneous complete AMF model held by every FedAvg client."""

    def __init__(self, user_vectors, service_vectors, user_biases, service_biases):
        self.user_vectors = np.asarray(user_vectors, dtype=np.float32)
        self.service_vectors = np.asarray(service_vectors, dtype=np.float32)
        self.user_biases = np.asarray(user_biases, dtype=np.float32)
        self.service_biases = np.asarray(service_biases, dtype=np.float32)
        if self.user_vectors.ndim != 2 or self.service_vectors.ndim != 2:
            raise ValueError("embedding tensors must be two-dimensional")
        if self.user_vectors.shape[1] != self.service_vectors.shape[1]:
            raise ValueError("user and service dimensions must match")
        if self.user_biases.shape != (self.user_vectors.shape[0],):
            raise ValueError("user bias shape mismatch")
        if self.service_biases.shape != (self.service_vectors.shape[0],):
            raise ValueError("service bias shape mismatch")

    @classmethod
    def random(cls, num_users: int, num_services: int, dimension: int, seed: int):
        if min(num_users, num_services, dimension) <= 0:
            raise ValueError("model dimensions must be positive")
        rng = np.random.default_rng(int(seed))
        return cls(
            rng.normal(0.0, 0.05, (num_users, dimension)).astype(np.float32),
            rng.normal(0.0, 0.05, (num_services, dimension)).astype(np.float32),
            np.zeros(num_users, dtype=np.float32),
            np.zeros(num_services, dtype=np.float32),
        )

    @property
    def dimension(self) -> int:
        return int(self.user_vectors.shape[1])

    @property
    def parameter_count(self) -> int:
        return int(self.user_vectors.size + self.service_vectors.size +
                   self.user_biases.size + self.service_biases.size)

    def copy(self) -> "FullAMFModel":
        return FullAMFModel(
            self.user_vectors.copy(), self.service_vectors.copy(),
            self.user_biases.copy(), self.service_biases.copy(),
        )

    def state_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.user_vectors, self.service_vectors,
                self.user_biases, self.service_biases)

    def predict_normalized(self, users, services) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        if users.shape != services.shape:
            raise ValueError("prediction indices must have the same shape")
        logits = (
            np.sum(self.user_vectors[users] * self.service_vectors[services], axis=-1)
            + self.user_biases[users] + self.service_biases[services]
        )
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def client_sgd_delta(
    global_model: FullAMFModel,
    user: int,
    services,
    targets,
    eta: float = 0.8,
    l2: float = 3e-4,
    local_epochs: int = 1,
) -> FedAvgClientDelta:
    """Compute one client's local SGD update from a shared global model.

    Targets are already normalized by the fixed log1p transform.  Support
    services are processed in ascending order.  The gradient is the ordinary
    squared-error gradient through the sigmoid, with L2 regularisation on the
    current parameter values, matching the plain AMF update contract.
    """
    if not (0 <= int(user) < global_model.user_vectors.shape[0]):
        raise IndexError("user index out of range")
    if eta <= 0 or l2 < 0 or int(local_epochs) <= 0:
        raise ValueError("invalid SGD hyperparameters")
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
    if not np.all(np.isfinite(targets)):
        raise ValueError("targets must be finite")

    dimension = global_model.dimension
    old_user = global_model.user_vectors[user].copy()
    old_user_bias = np.float32(global_model.user_biases[user])
    local_user = old_user.copy()
    local_user_bias = np.float32(old_user_bias)
    local_services = global_model.service_vectors[service_indices].copy()
    local_service_biases = global_model.service_biases[service_indices].copy()

    for _ in range(int(local_epochs)):
        for index, target in enumerate(targets):
            before_user = local_user.copy()
            before_user_bias = np.float32(local_user_bias)
            before_service = local_services[index].copy()
            before_service_bias = np.float32(local_service_biases[index])
            prediction = sigmoid(
                np.dot(before_user, before_service) +
                float(before_user_bias) + float(before_service_bias)
            )
            gradient = np.float32((prediction - float(target)) * prediction * (1.0 - prediction))
            local_user -= np.float32(eta) * (gradient * before_service + np.float32(l2) * before_user)
            local_services[index] -= np.float32(eta) * (gradient * before_user + np.float32(l2) * before_service)
            local_user_bias -= np.float32(eta) * (gradient + np.float32(l2) * before_user_bias)
            local_service_biases[index] -= np.float32(eta) * (
                gradient + np.float32(l2) * before_service_bias
            )

    return FedAvgClientDelta(
        user=int(user),
        service_indices=service_indices.copy(),
        user_vector_delta=(local_user - old_user).astype(np.float32),
        service_vector_deltas=(local_services - global_model.service_vectors[service_indices]).astype(np.float32),
        user_bias_delta=np.float32(local_user_bias - old_user_bias),
        service_bias_deltas=(local_service_biases - global_model.service_biases[service_indices]).astype(np.float32),
        # FedAvg weights a client's complete model by its support cardinality;
        # local epochs change the update, not the aggregation denominator.
        support_entries=int(len(service_indices)),
    )


def aggregate_sparse_deltas(
    global_model: FullAMFModel,
    client_deltas: list[FedAvgClientDelta],
) -> FullAMFModel:
    """Apply exact full-model FedAvg using sparse delta accumulation.

    The denominator is the sum of *all* client support counts.  Every client
    contributes its unchanged global parameters implicitly, so each tensor is
    still a full-model weighted average.
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
    """Reference implementation: weighted average of all four full tensors."""
    if not client_models or len(client_models) != len(support_counts):
        raise ValueError("client models and support counts must be non-empty and aligned")
    counts = np.asarray(support_counts, dtype=np.float64)
    if np.any(counts <= 0) or not np.all(np.isfinite(counts)):
        raise ValueError("support counts must be finite and positive")
    weights = counts / counts.sum()
    for model in client_models:
        if model.parameter_count != global_model.parameter_count:
            raise ValueError("client model shape mismatch")
    arrays = []
    for index in range(4):
        arrays.append(np.asarray(
            sum((weight * model.state_arrays()[index] for weight, model in zip(weights, client_models))),
            dtype=np.float32,
        ))
    return FullAMFModel(*arrays)
