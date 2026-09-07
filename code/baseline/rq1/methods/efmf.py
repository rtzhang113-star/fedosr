"""Causal reconstruction of FMF/EFMF for online QoS prediction.

The implementation follows Zhang, Zhang and Li (CollaborateCom 2020): user
latent factors stay private, a service latent matrix is shared, and each time
slice contributes only newly observed support tuples.  The optional EFMF
communication path adds random sparsification, stochastic b-bit quantization,
and delayed transmission.  Query labels are never accepted by this module.

The paper's cryptographic/differential-privacy sketch is intentionally not
implemented: it does not specify the sensitivity, privacy budget, keys, or the
fixed-point encoding required for a reproducible experiment.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


FP32_BYTES = 4
INDEX_BYTES = 4
_CENTER_EPS = 1e-6
_MIN_ABS_LOGIT = 1e-2


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def stochastic_uniform_quantize(vector, bits: int, rng: np.random.Generator):
    """Return an unbiased dequantized vector and its encoded payload bytes."""
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError("quantization input must be a finite vector")
    bits = int(bits)
    if bits <= 0 or bits > 32:
        raise ValueError("quantization bits must be in [1, 32]")
    if bits == 32 or vector.size == 0:
        return vector.copy(), int(vector.size * FP32_BYTES)
    lower = float(np.min(vector))
    upper = float(np.max(vector))
    if lower == upper:
        return vector.copy(), 2 * FP32_BYTES
    levels = (1 << bits) - 1
    scaled = (vector.astype(np.float64) - lower) * levels / (upper - lower)
    low = np.floor(scaled)
    probability_up = scaled - low
    codes = low + (rng.random(vector.size) < probability_up)
    decoded = lower + codes * (upper - lower) / levels
    payload = math.ceil(vector.size * bits / 8) + 2 * FP32_BYTES
    return decoded.astype(np.float32), int(payload)


@dataclass(frozen=True)
class EFMFRoundResult:
    support_entries: int
    participating_clients: int
    transmitted_service_vectors: int
    delayed_service_vectors: int
    uplink_bytes: int
    downlink_bytes: int

    def as_dict(self) -> dict:
        return {
            "support_entries": self.support_entries,
            "participating_clients": self.participating_clients,
            "transmitted_service_vectors": self.transmitted_service_vectors,
            "delayed_service_vectors": self.delayed_service_vectors,
            "uplink_bytes": self.uplink_bytes,
            "downlink_bytes": self.downlink_bytes,
        }


class EfficientFederatedMatrixFactorization:
    """User-private online FMF with optional EFMF communication controls."""

    def __init__(
        self,
        num_users: int,
        num_services: int,
        rank: int = 10,
        learning_rate: float = 0.01,
        regularization: float = 0.001,
        seed: int = 0,
        mode: str = "efmf",
        omit_fraction: float = 0.2,
        quantization_bits: int = 8,
        communication_threshold: float = 0.0,
        communication_enabled: bool = True,
    ):
        if min(num_users, num_services, rank) <= 0:
            raise ValueError("dimensions and rank must be positive")
        if learning_rate <= 0 or regularization < 0:
            raise ValueError("invalid optimization parameters")
        if mode not in {"fmf", "efmf"}:
            raise ValueError("mode must be 'fmf' or 'efmf'")
        if not 0.0 <= omit_fraction < 1.0:
            raise ValueError("omit_fraction must be in [0, 1)")
        if not 1 <= int(quantization_bits) <= 32:
            raise ValueError("quantization_bits must be in [1, 32]")
        if communication_threshold < 0 or not np.isfinite(communication_threshold):
            raise ValueError("communication_threshold must be finite and nonnegative")
        self.num_users = int(num_users)
        self.num_services = int(num_services)
        self.rank = int(rank)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.mode = mode
        self.omit_fraction = float(omit_fraction if mode == "efmf" else 0.0)
        self.quantization_bits = int(quantization_bits if mode == "efmf" else 32)
        self.communication_threshold = float(
            communication_threshold if mode == "efmf" else 0.0
        )
        self.communication_enabled = bool(communication_enabled)
        self.rng = np.random.default_rng(seed)
        self.private_user_factors = self.rng.normal(
            0.0, 0.05, (self.num_users, self.rank)
        ).astype(np.float32)
        self.shared_service_factors = self.rng.normal(
            0.0, 0.05, (self.num_services, self.rank)
        ).astype(np.float32)
        self.pending_updates: dict[tuple[int, int], np.ndarray] = {}
        self.round_index = 0
        # Scale adaptation is deliberately explicit and one-shot.  The runner
        # calls initialize_from_support_targets() with slice-0 support labels
        # before the first update; no query/future values can reach this path.
        self._support_scale_initialized = False
        self._support_scale_initialization_count = 0
        self._support_scale_info = {
            "initialized": False,
            "source": None,
            "support_entries": 0,
            "support_target_mean": None,
            "clipped_target_mean": None,
            "target_logit": None,
            "effective_logit": None,
            "factor_magnitude": None,
        }

    @property
    def shared_parameter_bytes(self) -> int:
        return int(self.shared_service_factors.size * FP32_BYTES)

    def state_copy(self) -> dict:
        return {
            "private_user_factors": self.private_user_factors.copy(),
            "shared_service_factors": self.shared_service_factors.copy(),
            "pending_updates": {
                key: value.copy() for key, value in self.pending_updates.items()
            },
            "round_index": self.round_index,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "support_scale_initialized": self._support_scale_initialized,
            "support_scale_initialization_count": self._support_scale_initialization_count,
            "support_scale_info": copy.deepcopy(self._support_scale_info),
        }

    def state_unchanged(self, snapshot: dict) -> bool:
        if self.round_index != snapshot["round_index"]:
            return False
        if self.rng.bit_generator.state != snapshot["rng_state"]:
            return False
        if (
            self._support_scale_initialized != snapshot["support_scale_initialized"]
            or self._support_scale_initialization_count
            != snapshot["support_scale_initialization_count"]
            or self._support_scale_info != snapshot["support_scale_info"]
        ):
            return False
        if not np.array_equal(
            self.private_user_factors, snapshot["private_user_factors"]
        ) or not np.array_equal(
            self.shared_service_factors, snapshot["shared_service_factors"]
        ):
            return False
        if self.pending_updates.keys() != snapshot["pending_updates"].keys():
            return False
        return all(
            np.array_equal(value, snapshot["pending_updates"][key])
            for key, value in self.pending_updates.items()
        )

    @property
    def support_scale_initialization_count(self) -> int:
        """Number of support-derived initializations performed (at most one)."""
        return int(self._support_scale_initialization_count)

    def support_scale_initialization_info(self) -> dict:
        """Return a JSON-safe copy of the disclosed initialization metadata."""
        return copy.deepcopy(self._support_scale_info)

    def initialize_from_support_targets(self, targets) -> dict:
        """Scale factors from slice-0 support targets exactly once.

        The model predicts ``sigmoid(U_i^T S_j)``.  A small zero-centred
        initializer therefore makes every initial prediction approximately
        0.5, which is badly mismatched for the low normalized TP targets.  We
        use only the normalized target mean from the first support slice to
        choose a common factor magnitude/sign whose typical dot product is the
        corresponding logit.  Small deterministic perturbations preserve
        learnable variation without changing that scale.

        This is an explicit reconstruction assumption, not a query-derived
        calibration step.  Calling the method after training or more than once
        is rejected so a later slice cannot silently refit the scale.
        """
        if self._support_scale_initialized:
            raise RuntimeError("support-scale initialization may run only once")
        if self.round_index != 0 or self.pending_updates:
            raise RuntimeError("support-scale initialization must precede training")

        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim != 1:
            raise ValueError("support targets must be a one-dimensional vector")
        if targets.size and (
            not np.all(np.isfinite(targets))
            or np.any(targets < 0.0)
            or np.any(targets > 1.0)
        ):
            raise ValueError("support targets must be finite normalized values in [0, 1]")

        if targets.size:
            support_mean = float(np.mean(targets, dtype=np.float64))
            source = "slice0_support_mean"
        else:
            # Empty slice-0 support contains no statistic.  Use the neutral
            # midpoint without consulting any other slice or query label.
            support_mean = 0.5
            source = "neutral_empty_slice0_support"
        clipped_mean = float(np.clip(support_mean, _CENTER_EPS, 1.0 - _CENTER_EPS))
        target_logit = float(np.log(clipped_mean) - np.log1p(-clipped_mean))
        effective_logit = target_logit
        if abs(effective_logit) < _MIN_ABS_LOGIT:
            # Keep a nonzero gradient at a 0.5 target while remaining within a
            # negligible distance of the requested support centre.
            effective_logit = _MIN_ABS_LOGIT if effective_logit >= 0.0 else -_MIN_ABS_LOGIT
        magnitude = float(np.sqrt(abs(effective_logit) / float(self.rank)))
        sign = 1.0 if effective_logit >= 0.0 else -1.0
        noise_scale = max(1e-4, min(1e-2, 0.05 * magnitude))

        self.private_user_factors = (
            magnitude
            + self.rng.normal(0.0, noise_scale, (self.num_users, self.rank))
        ).astype(np.float32)
        self.shared_service_factors = (
            sign * magnitude
            + self.rng.normal(0.0, noise_scale, (self.num_services, self.rank))
        ).astype(np.float32)
        if not np.all(np.isfinite(self.private_user_factors)) or not np.all(
            np.isfinite(self.shared_service_factors)
        ):
            raise FloatingPointError("scale-aware factor initialization was not finite")

        self._support_scale_initialized = True
        self._support_scale_initialization_count += 1
        self._support_scale_info = {
            "initialized": True,
            "source": source,
            "support_entries": int(targets.size),
            "support_target_mean": support_mean,
            "clipped_target_mean": clipped_mean,
            "target_logit": target_logit,
            "effective_logit": effective_logit,
            "factor_magnitude": magnitude,
            "noise_scale": noise_scale,
            "rank": int(self.rank),
        }
        return self.support_scale_initialization_info()

    def _validate_support(self, users, services, targets):
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float32)
        if users.ndim != 1 or users.shape != services.shape or users.shape != targets.shape:
            raise ValueError("users, services and targets must be aligned vectors")
        if users.size == 0:
            return users, services, targets
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise IndexError("user index out of range")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("service index out of range")
        if not np.all(np.isfinite(targets)) or np.any(targets < 0) or np.any(targets > 1):
            raise ValueError("targets must be finite normalized values in [0, 1]")
        return users, services, targets

    def _client_update(self, user: int, services, targets, epochs: int):
        if self.mode == "fmf":
            user_factor = self.rng.normal(0.0, 0.05, self.rank).astype(np.float32)
        else:
            user_factor = self.private_user_factors[user].copy()
        local_services = {
            int(service): self.shared_service_factors[int(service)].copy()
            for service in np.unique(services)
        }
        accumulated = {
            service: np.zeros(self.rank, dtype=np.float32) for service in local_services
        }
        lr = np.float32(self.learning_rate)
        reg = np.float32(self.regularization)
        for _ in range(int(epochs)):
            for service, target in zip(services, targets):
                service = int(service)
                u_before = user_factor.copy()
                s_before = local_services[service].copy()
                prediction = float(sigmoid(np.array([np.dot(u_before, s_before)]))[0])
                error = np.float32(prediction - float(target))
                derivative = np.float32(prediction * (1.0 - prediction))
                grad_user = error * derivative * s_before + reg * u_before
                grad_service = error * derivative * u_before + reg * s_before
                user_factor -= lr * grad_user
                local_services[service] -= lr * grad_service
                accumulated[service] += grad_service
        self.private_user_factors[user] = user_factor
        return accumulated, local_services

    def _compress(self, vector: np.ndarray):
        keep_probability = 1.0 - self.omit_fraction
        vector = vector.astype(np.float32, copy=True)
        if self.mode == "efmf" and self.omit_fraction > 0:
            keep = self.rng.random(self.rank) < keep_probability
            selected = (vector[keep] / keep_probability).astype(np.float32)
            decoded_selected, payload = stochastic_uniform_quantize(
                selected, self.quantization_bits, self.rng
            )
            decoded = np.zeros(self.rank, dtype=np.float32)
            decoded[keep] = decoded_selected
            # A rank-bit coordinate mask makes the sparse message decodable.
            payload += math.ceil(self.rank / 8)
            return decoded, int(payload)
        quantized, payload = stochastic_uniform_quantize(
            vector, self.quantization_bits, self.rng
        )
        return quantized, int(payload)

    def update_round(self, users, services, targets, epochs: int = 1) -> EFMFRoundResult:
        users, services, targets = self._validate_support(users, services, targets)
        if int(epochs) <= 0:
            raise ValueError("epochs must be positive")
        participants = np.unique(users)
        downlink = (
            int(len(participants) * self.shared_parameter_bytes)
            if self.communication_enabled else 0
        )
        transmitted: dict[int, list[np.ndarray]] = {}
        delayed = 0
        uplink = 0
        fmf_local_services: list[dict[int, np.ndarray]] = []
        for user in participants:
            selected = users == user
            client_gradients, local_services = self._client_update(
                int(user), services[selected], targets[selected], int(epochs)
            )
            if self.mode == "fmf":
                fmf_local_services.append(local_services)
                continue
            for service, gradient in client_gradients.items():
                key = (int(user), int(service))
                pending = self.pending_updates.get(
                    key, np.zeros(self.rank, dtype=np.float32)
                ) + gradient
                if (
                    not self.communication_enabled
                    or float(np.max(np.abs(pending))) <= self.communication_threshold
                ):
                    self.pending_updates[key] = pending.astype(np.float32)
                    delayed += 1
                    continue
                payload, payload_bytes = self._compress(pending)
                transmitted.setdefault(int(service), []).append(payload)
                self.pending_updates.pop(key, None)
                uplink += int(payload_bytes + 2 * INDEX_BYTES)
        if self.mode == "fmf":
            # Paper Eq. (12): average complete local service matrices over all
            # m users. Nonparticipating users and untouched rows contribute the
            # downloaded global rows, so the sparse delta form is equivalent.
            denominator = self.num_users
            if self.communication_enabled and denominator:
                global_before = self.shared_service_factors.copy()
                delta_sum: dict[int, np.ndarray] = {}
                for local_services in fmf_local_services:
                    for service, local_factor in local_services.items():
                        delta_sum.setdefault(
                            service, np.zeros(self.rank, dtype=np.float32)
                        )
                        delta_sum[service] += local_factor - global_before[service]
                for service, delta in delta_sum.items():
                    self.shared_service_factors[service] = (
                        global_before[service] + delta / denominator
                    )
                uplink = int(len(participants) * self.shared_parameter_bytes)
                transmitted_vectors = int(len(participants) * self.num_services)
            else:
                uplink = 0
                transmitted_vectors = 0
        else:
            lr = np.float32(self.learning_rate)
            for service, payloads in transmitted.items():
                mean_gradient = np.mean(np.stack(payloads), axis=0, dtype=np.float32)
                self.shared_service_factors[service] -= lr * mean_gradient
            transmitted_vectors = int(sum(map(len, transmitted.values())))
        self.round_index += 1
        return EFMFRoundResult(
            support_entries=int(users.size),
            participating_clients=int(len(participants)),
            transmitted_service_vectors=transmitted_vectors,
            delayed_service_vectors=int(delayed),
            uplink_bytes=int(uplink),
            downlink_bytes=int(downlink),
        )

    def predict(self, users: Iterable[int], services: Iterable[int]) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        if users.shape != services.shape:
            raise ValueError("users and services must have the same shape")
        if users.size and (
            np.any(users < 0) or np.any(users >= self.num_users)
            or np.any(services < 0) or np.any(services >= self.num_services)
        ):
            raise IndexError("prediction index out of range")
        logits = np.sum(
            self.private_user_factors[users] * self.shared_service_factors[services], axis=-1
        )
        return sigmoid(logits).astype(np.float32)

    def reconstruction_loss(self, users, services, targets) -> float:
        users, services, targets = self._validate_support(users, services, targets)
        predictions = self.predict(users, services)
        data = 0.5 * np.sum((predictions - targets) ** 2, dtype=np.float64)
        if users.size:
            unique_users = np.unique(users)
            unique_services = np.unique(services)
            penalty = 0.5 * self.regularization * (
                np.sum(self.private_user_factors[unique_users] ** 2, dtype=np.float64)
                + np.sum(self.shared_service_factors[unique_services] ** 2, dtype=np.float64)
            )
        else:
            penalty = 0.0
        return float(data + penalty)

    def support_reconstruction_metrics(self, users, services, targets) -> dict:
        """Return normalized, data-only support reconstruction audit metrics.

        The loss is mean squared error in the model's normalized target space;
        it intentionally excludes regularization so before/after values measure
        whether the observed support labels are fitted.  Empty support is
        represented by finite zeros for artifact/audit stability.
        """
        users, services, targets = self._validate_support(users, services, targets)
        predictions = self.predict(users, services)
        if not targets.size:
            return {
                "support_loss": 0.0,
                "support_prediction_mean": 0.0,
                "support_target_mean": 0.0,
            }
        errors = predictions.astype(np.float64) - targets.astype(np.float64)
        return {
            "support_loss": float(np.mean(errors * errors, dtype=np.float64)),
            "support_prediction_mean": float(np.mean(predictions, dtype=np.float64)),
            "support_target_mean": float(np.mean(targets, dtype=np.float64)),
        }
