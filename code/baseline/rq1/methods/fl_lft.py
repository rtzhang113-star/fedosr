"""Federated latent factorization of tensors (FL-LFT) reconstruction.

Implements the client/server alternating SGD in equations (10)--(15) of
Zhong, Tang and Wu (SMC 2024). User factors remain client-private; service
and time factors are shared. Only the supplied support tuples are consumed.
"""
from __future__ import annotations

import numpy as np


_SCALE_EPS = 1e-8


class FederatedLatentTensorFactorization:
    def __init__(
        self,
        num_users: int,
        num_services: int,
        num_times: int,
        rank: int = 10,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        seed: int = 0,
    ):
        if min(num_users, num_services, num_times, rank) <= 0:
            raise ValueError("tensor dimensions and rank must be positive")
        if learning_rate <= 0 or regularization < 0:
            raise ValueError("invalid FL-LFT optimization parameters")
        self.rank = int(rank)
        self._rng = np.random.default_rng(seed)
        self.private_user_factors = self._rng.normal(0.0, 0.05, (num_users, rank))
        self.shared_service_factors = self._rng.normal(0.0, 0.05, (num_services, rank))
        self.shared_time_factors = self._rng.normal(0.0, 0.05, (num_times, rank))
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        # FL-LFT is a multiplicative three-factor model.  A zero-centred
        # initializer makes its initial product almost exactly zero (and can
        # make all clipped predictions zero on the QoS scale), so the runner
        # performs one support-only scale initialization before the first
        # update.  Keep the metadata here so the audit can prove that it was
        # performed once and did not consume query/future labels.
        self._support_scale_initialized = False
        self._support_scale_initialization_count = 0
        self._support_scale_info = {
            "initialized": False,
            "source": None,
            "support_entries": 0,
            "support_target_mean": None,
            "effective_target_mean": None,
            "factor_magnitude": None,
            "noise_scale": None,
            "rank": int(rank),
            "mode": "positive_balanced_trilinear_scale",
        }

    @property
    def support_scale_initialization_count(self) -> int:
        """Number of support-derived initializations (at most one)."""
        return int(self._support_scale_initialization_count)

    def support_scale_initialization_info(self) -> dict:
        """Return JSON-safe metadata for the support-only initialization."""
        return dict(self._support_scale_info)

    def initialize_from_support_targets(self, targets) -> dict:
        """Initialize factor magnitudes from the first support slice only.

        For a target ``y`` and rank ``r``, three factors with a common
        magnitude ``m = (y/r)^(1/3)`` have expected reconstruction
        ``sum_k m^3 ~= y``.  Positive, balanced factors preserve the original
        FL-LFT trilinear reconstruction/update equations while avoiding the
        cancellation caused by three independent zero-mean random matrices.

        This method is intentionally one-shot: only the first support slice
        may provide the scale, and neither query nor future labels are ever
        accepted.  An empty first support slice uses a neutral normalized
        target of 0.5 rather than looking ahead.
        """
        if self._support_scale_initialized:
            raise RuntimeError("support-scale initialization may run only once")

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
            support_mean = 0.5
            source = "neutral_empty_slice0_support"

        # fixed_log1p values are strictly positive, but retain a floor so
        # direct unit tests and an empty/synthetic support set remain stable.
        effective_mean = float(max(support_mean, _SCALE_EPS))
        magnitude = float(np.cbrt(effective_mean / float(self.rank)))
        # Keep perturbations small relative to the common positive scale.  A
        # zero-mean perturbation restores learnable variation without allowing
        # the initialization itself to collapse back to sign cancellation.
        noise_scale = float(min(1e-2, max(1e-6, 0.05 * magnitude)))
        self.private_user_factors = (
            magnitude + self._rng.normal(0.0, noise_scale, self.private_user_factors.shape)
        ).astype(np.float64)
        self.shared_service_factors = (
            magnitude + self._rng.normal(0.0, noise_scale, self.shared_service_factors.shape)
        ).astype(np.float64)
        self.shared_time_factors = (
            magnitude + self._rng.normal(0.0, noise_scale, self.shared_time_factors.shape)
        ).astype(np.float64)
        if not all(np.all(np.isfinite(array)) for array in (
            self.private_user_factors, self.shared_service_factors, self.shared_time_factors
        )):
            raise FloatingPointError("support-only FL-LFT scale initialization was not finite")

        self._support_scale_initialized = True
        self._support_scale_initialization_count += 1
        self._support_scale_info = {
            "initialized": True,
            "source": source,
            "support_entries": int(targets.size),
            "support_target_mean": support_mean,
            "effective_target_mean": effective_mean,
            "factor_magnitude": magnitude,
            "noise_scale": noise_scale,
            "rank": int(self.rank),
            "mode": "positive_balanced_trilinear_scale",
        }
        return self.support_scale_initialization_info()

    def support_reconstruction_metrics(self, users, services, time_slices, values) -> dict:
        """Compute support-only reconstruction statistics without mutation."""
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        time_slices = np.asarray(time_slices, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if not (users.ndim == services.ndim == time_slices.ndim == values.ndim == 1):
            raise ValueError("support audit inputs must be one-dimensional vectors")
        if not (users.shape == services.shape == time_slices.shape == values.shape):
            raise ValueError("support audit inputs must have aligned shapes")
        if not np.all(np.isfinite(values)):
            raise ValueError("support audit targets must be finite")
        if not len(values):
            return {
                "support_target_mean": 0.0,
                "support_prediction_mean": 0.0,
                "support_loss": 0.0,
            }
        predictions = self.predict(users, services, time_slices)
        residual = predictions.astype(np.float64) - values
        return {
            "support_target_mean": float(np.mean(values, dtype=np.float64)),
            "support_prediction_mean": float(np.mean(predictions, dtype=np.float64)),
            "support_loss": float(np.mean(residual * residual, dtype=np.float64)),
        }

    def update_client(self, user: int, services, time_slices, values, epochs: int = 1) -> dict:
        services = np.asarray(services, dtype=np.int64)
        time_slices = np.asarray(time_slices, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if services.shape != time_slices.shape or services.shape != values.shape or services.ndim != 1:
            raise ValueError("services, time_slices and values must be aligned vectors")
        if not np.all(np.isfinite(values)) or epochs <= 0:
            raise ValueError("values must be finite and epochs positive")
        lr, reg = self.learning_rate, self.regularization
        uploaded_scalars = 0
        for _ in range(int(epochs)):
            for service, time_slice, target in zip(services, time_slices, values):
                d = self.private_user_factors[user].copy()
                e = self.shared_service_factors[service].copy()
                t = self.shared_time_factors[time_slice].copy()
                prediction = float(np.sum(d * e * t))
                error = prediction - target
                self.private_user_factors[user] -= lr * (error * e * t + reg * d)
                d_updated = self.private_user_factors[user].copy()
                grad_e = error * d_updated * t + reg * e
                grad_t = error * d_updated * e + reg * t
                self.shared_service_factors[service] -= lr * grad_e
                self.shared_time_factors[time_slice] -= lr * grad_t
                uploaded_scalars += grad_e.size + grad_t.size
        downloaded_scalars = self.shared_service_factors.size + self.shared_time_factors.size
        return {
            "uploaded_bytes": int(uploaded_scalars * 8),
            "downloaded_bytes": int(downloaded_scalars * 8),
            "support_entries": int(len(values) * epochs),
        }

    def predict(self, users, services, time_slices) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        time_slices = np.asarray(time_slices, dtype=np.int64)
        if users.shape != services.shape or users.shape != time_slices.shape:
            raise ValueError("prediction indices must have the same shape")
        return np.sum(
            self.private_user_factors[users]
            * self.shared_service_factors[services]
            * self.shared_time_factors[time_slices],
            axis=-1,
        )
