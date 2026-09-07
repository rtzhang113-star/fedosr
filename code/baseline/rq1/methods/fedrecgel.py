"""Independent NumPy QoS adaptation of FedRecGEL.

The public FedRecGEL implementation is a PyTorch implicit-feedback
recommender.  This file is an independent reimplementation of its mechanism;
it does not copy source code.  The item-centred split is retained:

* user/GMF and user/MLP embeddings are private client state;
* service/GMF and service/MLP embeddings, the MLP, and the regression head
  are shared server state;
* each client computes two SAM gradients (shared and non-shared groups);
* the server applies an equal-client-weighted shared-gradient update.

The objective here is a continuous QoS regression objective.  The original
binary negative-sampling/BCE objective is deliberately not used.  The runner
is responsible for the causal ``history + support[t] -> query[t]`` protocol.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np


FP32_BYTES = 4
_ADAM_EPS = 1e-8
_SAM_EPS = 1e-12
_CENTER_EPS = 1e-6
_OUTPUT_BOUNDARY_EPS = 1e-6


def _xavier_uniform(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = float(np.sqrt(6.0 / float(fan_in + fan_out)))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0).astype(np.float32, copy=False)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Evaluate sigmoid without overflowing for large-magnitude logits."""
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values, dtype=np.float32)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


class _PerturbationList(list):
    """SAM deltas carrying exact pre-perturbation FP32 snapshots."""

    def __init__(self, values: Iterable[np.ndarray], originals: Iterable[np.ndarray]):
        super().__init__(values)
        self.originals = [np.array(array, copy=True) for array in originals]


@dataclass(frozen=True)
class FedRecGELRoundResult:
    """Communication and participation accounting for one support round."""

    support_entries: int
    participating_clients: int
    active_services: int
    shared_gradient_clients: int
    uplink_bytes: int
    downlink_bytes: int

    def as_dict(self) -> dict:
        return {
            "support_entries": int(self.support_entries),
            "participating_clients": int(self.participating_clients),
            "active_services": int(self.active_services),
            "shared_gradient_clients": int(self.shared_gradient_clients),
            "uplink_bytes": int(self.uplink_bytes),
            "downlink_bytes": int(self.downlink_bytes),
        }


class _Adam:
    """Small NumPy Adam state used for temporary client/server parameters."""

    def __init__(self, arrays: Iterable[np.ndarray]):
        self.m = [np.zeros_like(array, dtype=np.float32) for array in arrays]
        self.v = [np.zeros_like(array, dtype=np.float32) for array in arrays]
        self.step = 0

    def update(self, arrays: list[np.ndarray], gradients: list[np.ndarray], learning_rate: float) -> None:
        if len(arrays) != len(gradients) or len(arrays) != len(self.m):
            raise ValueError("Adam arrays and gradients must be aligned")
        for array, gradient in zip(arrays, gradients):
            if not np.all(np.isfinite(array)) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError("FedRecGEL Adam received a non-finite array or gradient")
        self.step += 1
        beta1, beta2 = 0.9, 0.999
        correction_1 = 1.0 - beta1 ** self.step
        correction_2 = 1.0 - beta2 ** self.step
        for array, gradient, first, second in zip(arrays, gradients, self.m, self.v):
            gradient = np.asarray(gradient, dtype=np.float32)
            first *= beta1
            first += (1.0 - beta1) * gradient
            second *= beta2
            second += (1.0 - beta2) * (gradient * gradient)
            first_hat = first / correction_1
            second_hat = second / correction_2
            array -= np.float32(learning_rate) * first_hat / (np.sqrt(second_hat) + _ADAM_EPS)
            if not np.all(np.isfinite(array)):
                raise FloatingPointError("FedRecGEL Adam produced a non-finite parameter")


class FedRecGELQoS:
    """FedRecGEL's shared/private/SAM mechanism for continuous QoS values.

    ``factor_num`` is the GMF width.  The MLP branch follows the public
    implementation's dimensions: the user and service MLP embeddings each
    have ``factor_num * 2**(num_layers-1)`` coordinates, and each linear-ReLU
    block halves the width until ``factor_num`` remains.  The QoS regression
    head emits a sigmoid-bounded normalized prediction in ``[0, 1]``.  Its
    bias is initialized once from the first non-empty support mean by the
    causal runner.
    """

    def __init__(
        self,
        num_users: int,
        num_services: int,
        factor_num: int = 32,
        num_layers: int = 3,
        local_lr: float = 0.01,
        global_lr: float = 0.001,
        rho_sh: float = 0.02,
        rho_ns: float = 0.02,
        local_epochs: int = 1,
        client_fraction: float = 0.1,
        seed: int = 42,
        communication_enabled: bool = True,
    ):
        if min(int(num_users), int(num_services), int(factor_num), int(num_layers)) <= 0:
            raise ValueError("dimensions and num_layers must be positive")
        if not np.isfinite(local_lr) or local_lr <= 0.0:
            raise ValueError("local_lr must be finite and positive")
        if not np.isfinite(global_lr) or global_lr <= 0.0:
            raise ValueError("global_lr must be finite and positive")
        if not np.isfinite(rho_sh) or rho_sh < 0.0:
            raise ValueError("rho_sh must be finite and non-negative")
        if not np.isfinite(rho_ns) or rho_ns < 0.0:
            raise ValueError("rho_ns must be finite and non-negative")
        if int(local_epochs) <= 0:
            raise ValueError("local_epochs must be positive")
        if not np.isfinite(client_fraction) or not 0.0 < float(client_fraction) <= 1.0:
            raise ValueError("client_fraction must be in (0, 1]")

        self.num_users = int(num_users)
        self.num_services = int(num_services)
        self.factor_num = int(factor_num)
        self.num_layers = int(num_layers)
        self.mlp_embedding_dim = self.factor_num * (2 ** (self.num_layers - 1))
        self.local_lr = float(local_lr)
        self.global_lr = float(global_lr)
        self.rho_sh = float(rho_sh)
        self.rho_ns = float(rho_ns)
        self.local_epochs = int(local_epochs)
        self.client_fraction = float(client_fraction)
        self.communication_enabled = bool(communication_enabled)
        self.rng = np.random.default_rng(int(seed))

        # Private non-shared client state.
        self.private_user_gmf = self.rng.normal(
            0.0, 0.01, size=(self.num_users, self.factor_num)
        ).astype(np.float32)
        self.private_user_mlp = self.rng.normal(
            0.0, 0.01, size=(self.num_users, self.mlp_embedding_dim)
        ).astype(np.float32)

        # Shared server state.  It is intentionally separated from private
        # user state so that the communication audit can inspect the boundary.
        self.shared_service_gmf = self.rng.normal(
            0.0, 0.01, size=(self.num_services, self.factor_num)
        ).astype(np.float32)
        self.shared_service_mlp = self.rng.normal(
            0.0, 0.01, size=(self.num_services, self.mlp_embedding_dim)
        ).astype(np.float32)
        self.mlp_weights: list[np.ndarray] = []
        self.mlp_biases: list[np.ndarray] = []
        input_dim = 2 * self.mlp_embedding_dim
        for _ in range(self.num_layers):
            output_dim = input_dim // 2
            if output_dim <= 0:
                raise ValueError("num_layers produces an invalid MLP width")
            self.mlp_weights.append(_xavier_uniform(self.rng, input_dim, output_dim))
            self.mlp_biases.append(np.zeros(output_dim, dtype=np.float32))
            input_dim = output_dim
        if input_dim != self.factor_num:
            raise ValueError("MLP width does not reduce to factor_num")
        self.head_weight = _xavier_uniform(self.rng, 2 * self.factor_num, 1).reshape(-1)
        self.head_bias = np.zeros(1, dtype=np.float32)

        # Persistent optimizer state for private users and server shared state.
        self._user_gmf_m = np.zeros_like(self.private_user_gmf)
        self._user_gmf_v = np.zeros_like(self.private_user_gmf)
        self._user_mlp_m = np.zeros_like(self.private_user_mlp)
        self._user_mlp_v = np.zeros_like(self.private_user_mlp)
        self._user_steps = np.zeros(self.num_users, dtype=np.int64)
        self._server_adam = _Adam(self._shared_parameter_arrays())
        self.round_index = 0
        # The original method's binary task starts with a zero-centred
        # implicit-feedback head.  For continuous QoS, a one-shot bias
        # calibration from the first non-empty support mean prevents the
        # initial sigmoid output from being fixed near 0.5.  The runner calls
        # this before the first update; update_round also guards direct use.
        self._support_scale_initialized = False
        self._support_scale_initialization_count = 0
        self._support_scale_info = {
            "initialized": False,
            "source": None,
            "support_entries": 0,
            "support_target_mean": None,
            "clipped_target_mean": None,
            "target_logit": None,
            "head_bias": None,
        }

    # ------------------------------------------------------------------
    # Public state and communication boundary
    # ------------------------------------------------------------------
    def _shared_parameter_arrays(self) -> list[np.ndarray]:
        return [
            self.shared_service_gmf,
            self.shared_service_mlp,
            *self.mlp_weights,
            *self.mlp_biases,
            self.head_weight,
            self.head_bias,
        ]

    @property
    def shared_parameter_count(self) -> int:
        return int(sum(array.size for array in self._shared_parameter_arrays()))

    @property
    def shared_parameter_bytes(self) -> int:
        return int(self.shared_parameter_count * FP32_BYTES)

    def private_state_arrays(self) -> tuple[np.ndarray, ...]:
        """Return only model arrays belonging to clients."""
        return self.private_user_gmf, self.private_user_mlp

    def shared_state_arrays(self) -> tuple[np.ndarray, ...]:
        """Return only model arrays transmitted by the server."""
        return tuple(self._shared_parameter_arrays())

    def state_arrays(self) -> tuple[np.ndarray, ...]:
        """Return all mutable arrays for byte-level query read-only audits."""
        return (
            *self.private_state_arrays(),
            *self.shared_state_arrays(),
            self._user_gmf_m,
            self._user_gmf_v,
            self._user_mlp_m,
            self._user_mlp_v,
            self._user_steps,
            *self._server_adam.m,
            *self._server_adam.v,
        )

    def state_copy(self) -> dict:
        """Capture every mutable model/optimizer scalar and RNG state.

        Query evaluation is required to be read-only.  Array-only snapshots
        are insufficient because the round counter, server Adam step and RNG
        state are mutable state as well, so they are included explicitly.
        """
        return {
            "arrays": tuple(np.array(array, copy=True) for array in self.state_arrays()),
            "round_index": int(self.round_index),
            "server_adam_step": int(self._server_adam.step),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "support_scale_initialized": bool(self._support_scale_initialized),
            "support_scale_initialization_count": int(self._support_scale_initialization_count),
            "support_scale_info": copy.deepcopy(self._support_scale_info),
        }

    @staticmethod
    def _nested_state_equal(left, right) -> bool:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            try:
                return np.array_equal(np.asarray(left), np.asarray(right))
            except (TypeError, ValueError):
                return False
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict):
                return False
            return left.keys() == right.keys() and all(
                FedRecGELQoS._nested_state_equal(left[key], right[key]) for key in left
            )
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
                return False
            return len(left) == len(right) and all(
                FedRecGELQoS._nested_state_equal(item_left, item_right)
                for item_left, item_right in zip(left, right)
            )
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False

    def state_unchanged(self, snapshot: dict) -> bool:
        """Return whether the complete state still matches ``state_copy``."""
        if not isinstance(snapshot, dict):
            return False
        current = self.state_copy()
        return (
            self._nested_state_equal(snapshot.get("arrays"), current["arrays"])
            and snapshot.get("round_index") == current["round_index"]
            and snapshot.get("server_adam_step") == current["server_adam_step"]
            and self._nested_state_equal(snapshot.get("rng_state"), current["rng_state"])
            and snapshot.get("support_scale_initialized") == current["support_scale_initialized"]
            and snapshot.get("support_scale_initialization_count") == current["support_scale_initialization_count"]
            and self._nested_state_equal(snapshot.get("support_scale_info"), current["support_scale_info"])
        )

    @property
    def support_scale_initialization_count(self) -> int:
        return int(self._support_scale_initialization_count)

    @property
    def support_scale_info(self) -> dict:
        return copy.deepcopy(self._support_scale_info)

    def initialize_output_scale(
        self, targets: Iterable[float], source: str = "first_nonempty_support"
    ) -> bool:
        """Initialize the bounded QoS head once using support-only evidence."""
        targets = np.asarray(targets, dtype=np.float32)
        if targets.ndim != 1:
            raise ValueError("scale-initialization targets must be a vector")
        if self._support_scale_initialized or targets.size == 0:
            return False
        if not np.all(np.isfinite(targets)) or np.any(targets < 0.0) or np.any(targets > 1.0):
            raise ValueError("scale-initialization targets must be finite normalized values")
        target_mean = float(np.mean(targets, dtype=np.float64))
        clipped_mean = float(np.clip(target_mean, _CENTER_EPS, 1.0 - _CENTER_EPS))
        target_logit = float(np.log(clipped_mean / (1.0 - clipped_mean)))
        self.head_bias.fill(np.float32(target_logit))
        self._support_scale_initialized = True
        self._support_scale_initialization_count += 1
        self._support_scale_info = {
            "initialized": True,
            "source": str(source),
            "support_entries": int(targets.size),
            "support_target_mean": target_mean,
            "clipped_target_mean": clipped_mean,
            "target_logit": target_logit,
            "head_bias": float(self.head_bias[0]),
        }
        return True

    def _server_shared_bundle(self) -> dict:
        """Return the server shared parameters as one explicit forward bundle."""
        return {
            "service_gmf": self.shared_service_gmf,
            "service_mlp": self.shared_service_mlp,
            "mlp_weights": self.mlp_weights,
            "mlp_biases": self.mlp_biases,
            "head_weight": self.head_weight,
            "head_bias": self.head_bias,
        }

    # ------------------------------------------------------------------
    # Forward and manual backpropagation
    # ------------------------------------------------------------------
    def _forward(
        self,
        user_gmf: np.ndarray,
        user_mlp: np.ndarray,
        shared: dict,
        services_local: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        service_gmf = shared["service_gmf"]
        service_mlp = shared["service_mlp"]
        mlp_weights = shared["mlp_weights"]
        mlp_biases = shared["mlp_biases"]
        head_weight = shared["head_weight"]
        head_bias = shared["head_bias"]
        services_local = np.asarray(services_local, dtype=np.int64)
        count = int(services_local.size)
        user_gmf_batch = np.broadcast_to(user_gmf, (count, self.factor_num))
        user_mlp_batch = np.broadcast_to(user_mlp, (count, self.mlp_embedding_dim))
        service_gmf_batch = service_gmf[services_local]
        service_mlp_batch = service_mlp[services_local]
        gmf = user_gmf_batch * service_gmf_batch
        activation = np.concatenate((user_mlp_batch, service_mlp_batch), axis=1).astype(np.float32)
        activations = [activation]
        preactivations = []
        for weight, bias in zip(mlp_weights, mlp_biases):
            preactivation = activation @ weight + bias
            preactivations.append(preactivation.astype(np.float32, copy=False))
            activation = _relu(preactivation)
            activations.append(activation)
        combined = np.concatenate((gmf, activation), axis=1).astype(np.float32)
        logits = (combined @ head_weight + head_bias[0]).astype(np.float32)
        prediction = _sigmoid(logits)
        return prediction, {
            "user_gmf_batch": user_gmf_batch,
            "user_mlp_batch": user_mlp_batch,
            "service_gmf_batch": service_gmf_batch,
            "service_mlp_batch": service_mlp_batch,
            "gmf": gmf,
            "activations": activations,
            "preactivations": preactivations,
            "combined": combined,
            "logits": logits,
            "prediction": prediction,
        }

    def _forward_backward(
        self,
        user_gmf: np.ndarray,
        user_mlp: np.ndarray,
        shared: dict,
        services_local: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float, dict[str, np.ndarray], dict[str, list[np.ndarray] | np.ndarray]]:
        service_gmf = shared["service_gmf"]
        service_mlp = shared["service_mlp"]
        mlp_weights = shared["mlp_weights"]
        mlp_biases = shared["mlp_biases"]
        head_weight = shared["head_weight"]
        targets = np.asarray(targets, dtype=np.float32)
        services_local = np.asarray(services_local, dtype=np.int64)
        if targets.ndim != 1 or targets.size != len(services_local):
            raise ValueError("targets and service indices must be aligned vectors")
        if targets.size == 0:
            empty_private = {
                "user_gmf": np.zeros_like(user_gmf),
                "user_mlp": np.zeros_like(user_mlp),
            }
            empty_shared = self._empty_shared_gradients(service_gmf.shape[0], service_mlp.shape[0])
            return 0.0, empty_private, empty_shared

        prediction, cache = self._forward(
            user_gmf, user_mlp, shared, services_local
        )
        residual = prediction - targets
        loss = float(0.5 * np.mean(residual.astype(np.float64) ** 2))
        # The QoS target is normalized to [0, 1].  Backpropagate through the
        # bounded sigmoid head so training and inverse normalization use the
        # same output domain.
        sigmoid_slope = cache["prediction"] * (1.0 - cache["prediction"])
        grad_prediction = (
            (residual / np.float32(targets.size)) * sigmoid_slope
        ).astype(np.float32)

        head_gradient = cache["combined"].T @ grad_prediction
        head_bias_gradient = np.array([np.sum(grad_prediction, dtype=np.float64)], dtype=np.float32)
        grad_combined = grad_prediction[:, None] * head_weight[None, :]
        grad_gmf = grad_combined[:, : self.factor_num]
        grad_mlp = grad_combined[:, self.factor_num :]

        mlp_weight_gradients = [np.zeros_like(weight) for weight in mlp_weights]
        mlp_bias_gradients = [np.zeros_like(bias) for bias in mlp_biases]
        for index in range(self.num_layers - 1, -1, -1):
            grad_pre = grad_mlp * (cache["preactivations"][index] > 0.0)
            mlp_weight_gradients[index] = cache["activations"][index].T @ grad_pre
            mlp_bias_gradients[index] = np.sum(grad_pre, axis=0, dtype=np.float64).astype(np.float32)
            grad_mlp = grad_pre @ mlp_weights[index].T

        grad_user_gmf = np.sum(grad_gmf * cache["service_gmf_batch"], axis=0, dtype=np.float64).astype(np.float32)
        grad_service_gmf = grad_gmf * cache["user_gmf_batch"]
        # The concatenated MLP input keeps private and shared derivatives in
        # separate halves; no user gradient is transmitted to the server.
        grad_input_mlp = grad_mlp
        grad_user_mlp = np.sum(grad_input_mlp[:, : self.mlp_embedding_dim], axis=0, dtype=np.float64).astype(np.float32)
        grad_service_mlp = grad_input_mlp[:, self.mlp_embedding_dim :]

        service_gmf_gradient = np.zeros_like(service_gmf)
        service_mlp_gradient = np.zeros_like(service_mlp)
        np.add.at(service_gmf_gradient, services_local, grad_service_gmf)
        np.add.at(service_mlp_gradient, services_local, grad_service_mlp)

        private_gradients = {"user_gmf": grad_user_gmf, "user_mlp": grad_user_mlp}
        shared_gradients = {
            "service_gmf": service_gmf_gradient,
            "service_mlp": service_mlp_gradient,
            "mlp_weights": mlp_weight_gradients,
            "mlp_biases": mlp_bias_gradients,
            "head_weight": head_gradient.astype(np.float32),
            "head_bias": head_bias_gradient,
        }
        return loss, private_gradients, shared_gradients

    def _empty_shared_gradients(self, service_gmf_rows: int, service_mlp_rows: int) -> dict:
        return {
            "service_gmf": np.zeros((service_gmf_rows, self.factor_num), dtype=np.float32),
            "service_mlp": np.zeros((service_mlp_rows, self.mlp_embedding_dim), dtype=np.float32),
            "mlp_weights": [np.zeros_like(weight) for weight in self.mlp_weights],
            "mlp_biases": [np.zeros_like(bias) for bias in self.mlp_biases],
            "head_weight": np.zeros_like(self.head_weight),
            "head_bias": np.zeros_like(self.head_bias),
        }

    @staticmethod
    def _gradient_norm(arrays: Iterable[np.ndarray]) -> float:
        total = 0.0
        for array in arrays:
            total += float(np.sum(np.asarray(array, dtype=np.float64) ** 2))
        return float(np.sqrt(total))

    @staticmethod
    def _perturb(arrays: list[np.ndarray], gradients: list[np.ndarray], radius: float) -> list[np.ndarray]:
        originals = [np.array(array, copy=True) for array in arrays]
        norm = FedRecGELQoS._gradient_norm(gradients)
        if radius == 0.0 or norm <= _SAM_EPS:
            return _PerturbationList([np.zeros_like(array) for array in arrays], originals)
        scale = np.float32(radius / (norm + _SAM_EPS))
        perturbations = [gradient.astype(np.float32) * scale for gradient in gradients]
        for array, perturbation in zip(arrays, perturbations):
            array += perturbation
        return _PerturbationList(perturbations, originals)

    @staticmethod
    def _restore(arrays: list[np.ndarray], perturbations: list[np.ndarray]) -> None:
        originals = getattr(perturbations, "originals", None)
        if originals is not None:
            for array, original in zip(arrays, originals):
                np.copyto(array, original)
            return
        for array, perturbation in zip(arrays, perturbations):
            array -= perturbation

    def _shared_gradient_list(self, gradients: dict) -> list[np.ndarray]:
        return [
            gradients["service_gmf"],
            gradients["service_mlp"],
            *gradients["mlp_weights"],
            *gradients["mlp_biases"],
            gradients["head_weight"],
            gradients["head_bias"],
        ]

    def _client_update(
        self,
        user: int,
        services: np.ndarray,
        targets: np.ndarray,
        epochs: int,
    ) -> tuple[np.ndarray, dict, float, float]:
        unique_services = np.unique(services).astype(np.int64)
        local_indices = np.searchsorted(unique_services, services)
        local_user_gmf = self.private_user_gmf[user].copy()
        local_user_mlp = self.private_user_mlp[user].copy()
        local_shared = {
            "service_gmf": self.shared_service_gmf[unique_services].copy(),
            "service_mlp": self.shared_service_mlp[unique_services].copy(),
            "mlp_weights": [weight.copy() for weight in self.mlp_weights],
            "mlp_biases": [bias.copy() for bias in self.mlp_biases],
            "head_weight": self.head_weight.copy(),
            "head_bias": self.head_bias.copy(),
        }
        local_shared_arrays = self._shared_dict_arrays(local_shared)
        local_adam = _Adam(local_shared_arrays)
        before_loss = 0.0
        after_loss = 0.0
        final_epoch_shared_gradients = self._empty_shared_gradients(
            len(unique_services), len(unique_services)
        )

        for epoch in range(int(epochs)):
            before_loss, base_private, base_shared = self._forward_backward(
                local_user_gmf,
                local_user_mlp,
                local_shared,
                local_indices,
                targets,
            )

            # Shared SAM: perturb only shared parameters, recompute the loss,
            # capture shared gradients, then restore the unperturbed point.
            shared_arrays = self._shared_dict_arrays(local_shared)
            shared_base_gradients = self._shared_gradient_list(base_shared)
            shared_perturbations = self._perturb(shared_arrays, shared_base_gradients, self.rho_sh)
            _, _, shared_sam = self._forward_backward(
                local_user_gmf,
                local_user_mlp,
                local_shared,
                local_indices,
                targets,
            )
            shared_sam_gradients = self._shared_gradient_list(shared_sam)
            self._restore(shared_arrays, shared_perturbations)

            # Non-shared SAM: perturb only this client's private embeddings,
            # recompute, capture private gradients, then restore.
            private_arrays = [local_user_gmf, local_user_mlp]
            private_base_gradients = [base_private["user_gmf"], base_private["user_mlp"]]
            private_perturbations = self._perturb(private_arrays, private_base_gradients, self.rho_ns)
            _, private_sam, _ = self._forward_backward(
                local_user_gmf,
                local_user_mlp,
                local_shared,
                local_indices,
                targets,
            )
            private_sam_gradients = [private_sam["user_gmf"], private_sam["user_mlp"]]
            self._restore(private_arrays, private_perturbations)

            # The public implementation uses Adam for both groups.  The
            # temporary shared copy is discarded after this client call, but
            # retaining its within-call updates preserves local_epochs>1.
            local_adam.update(shared_arrays, shared_sam_gradients, self.local_lr)
            self._update_private_user(user, private_sam_gradients)
            local_user_gmf = self.private_user_gmf[user].copy()
            local_user_mlp = self.private_user_mlp[user].copy()
            after_loss, _, _ = self._forward_backward(
                local_user_gmf,
                local_user_mlp,
                local_shared,
                local_indices,
                targets,
            )
            # FedRecGEL uploads one shared SAM gradient per client.  With
            # multiple local epochs this is the final epoch's gradient (the
            # source optimizer's last ``p.grad``), not a sum across epochs.
            # It is evaluated from the updated local shared bundle above.
            final_epoch_shared_gradients = {
                "service_gmf": shared_sam["service_gmf"].copy(),
                "service_mlp": shared_sam["service_mlp"].copy(),
                "mlp_weights": [gradient.copy() for gradient in shared_sam["mlp_weights"]],
                "mlp_biases": [gradient.copy() for gradient in shared_sam["mlp_biases"]],
                "head_weight": shared_sam["head_weight"].copy(),
                "head_bias": shared_sam["head_bias"].copy(),
            }

        return unique_services, final_epoch_shared_gradients, float(before_loss), float(after_loss)

    @staticmethod
    def _shared_dict_arrays(shared: dict) -> list[np.ndarray]:
        return [
            shared["service_gmf"],
            shared["service_mlp"],
            *shared["mlp_weights"],
            *shared["mlp_biases"],
            shared["head_weight"],
            shared["head_bias"],
        ]

    def _update_private_user(self, user: int, gradients: list[np.ndarray]) -> None:
        user = int(user)
        if any(not np.all(np.isfinite(gradient)) for gradient in gradients):
            raise FloatingPointError("FedRecGEL private update received a non-finite gradient")
        self._user_steps[user] += 1
        step = int(self._user_steps[user])
        beta1, beta2 = 0.9, 0.999
        correction_1 = 1.0 - beta1 ** step
        correction_2 = 1.0 - beta2 ** step
        for array, first, second, gradient in (
            (self.private_user_gmf[user], self._user_gmf_m[user], self._user_gmf_v[user], gradients[0]),
            (self.private_user_mlp[user], self._user_mlp_m[user], self._user_mlp_v[user], gradients[1]),
        ):
            first *= beta1
            first += (1.0 - beta1) * gradient
            second *= beta2
            second += (1.0 - beta2) * (gradient * gradient)
            array -= np.float32(self.local_lr) * (first / correction_1) / (
                np.sqrt(second / correction_2) + _ADAM_EPS
            )
            if not np.all(np.isfinite(array)):
                raise FloatingPointError("FedRecGEL private update produced a non-finite parameter")

    def _assert_finite_state(self) -> None:
        if not all(np.all(np.isfinite(array)) for array in self.state_arrays()):
            raise FloatingPointError("FedRecGEL state is not finite after the support update")

    def update_round(
        self,
        users: Iterable[int],
        services: Iterable[int],
        targets: Iterable[float],
        epochs: int | None = None,
    ) -> FedRecGELRoundResult:
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float32)
        if users.ndim != 1 or users.shape != services.shape or users.shape != targets.shape:
            raise ValueError("users, services and targets must be aligned vectors")
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise IndexError("user index out of range")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("service index out of range")
        if not np.all(np.isfinite(targets)):
            raise ValueError("targets must be finite normalized values")
        if np.any(targets < 0.0) or np.any(targets > 1.0):
            raise ValueError("targets must lie in the normalized [0, 1] domain")
        self.initialize_output_scale(targets)
        local_epochs = self.local_epochs if epochs is None else int(epochs)
        if local_epochs <= 0:
            raise ValueError("epochs must be positive")
        eligible_clients = np.unique(users)
        if eligible_clients.size == 0:
            self.round_index += 1
            self._assert_finite_state()
            return FedRecGELRoundResult(0, 0, 0, 0, 0, 0)

        # The public server samples a fixed fraction of clients once per
        # global round.  Here the eligible population is restricted to users
        # having current-slice support, preserving the causal QoS protocol.
        selected_count = max(1, int(self.client_fraction * eligible_clients.size))
        if selected_count >= eligible_clients.size:
            participants = eligible_clients
        else:
            participants = np.sort(
                self.rng.choice(eligible_clients, size=selected_count, replace=False)
            )
        selected_rows = np.isin(users, participants)
        users = users[selected_rows]
        services = services[selected_rows]
        targets = targets[selected_rows]
        active_services = np.unique(services)

        aggregate = self._empty_shared_gradients(0, 0)
        # Full service-row gradients are accumulated in server coordinates.
        aggregate["service_gmf"] = np.zeros_like(self.shared_service_gmf)
        aggregate["service_mlp"] = np.zeros_like(self.shared_service_mlp)
        for user in participants:
            selected = users == int(user)
            local_services, gradients, _, _ = self._client_update(
                int(user), services[selected], targets[selected], local_epochs
            )
            client_count = float(participants.size)
            aggregate["service_gmf"][local_services] += gradients["service_gmf"] / np.float32(client_count)
            aggregate["service_mlp"][local_services] += gradients["service_mlp"] / np.float32(client_count)
            scale = np.float32(1.0 / client_count)
            for target_array, source_array in zip(aggregate["mlp_weights"], gradients["mlp_weights"]):
                target_array += scale * source_array
            for target_array, source_array in zip(aggregate["mlp_biases"], gradients["mlp_biases"]):
                target_array += scale * source_array
            aggregate["head_weight"] += scale * gradients["head_weight"]
            aggregate["head_bias"] += scale * gradients["head_bias"]

        if self.communication_enabled:
            self._server_adam.update(
                self._shared_parameter_arrays(),
                self._shared_gradient_list(aggregate),
                self.global_lr,
            )
            bytes_each = self.shared_parameter_bytes
            uplink_bytes = int(participants.size * bytes_each)
            downlink_bytes = int(participants.size * bytes_each)
        else:
            uplink_bytes = 0
            downlink_bytes = 0
        self.round_index += 1
        self._assert_finite_state()
        return FedRecGELRoundResult(
            support_entries=int(users.size),
            participating_clients=int(participants.size),
            active_services=int(active_services.size),
            shared_gradient_clients=int(participants.size),
            uplink_bytes=uplink_bytes,
            downlink_bytes=downlink_bytes,
        )

    def predict(self, users: Iterable[int], services: Iterable[int]) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        if users.ndim != 1 or users.shape != services.shape:
            raise ValueError("users and services must be aligned vectors")
        if np.any(users < 0) or np.any(users >= self.num_users):
            raise IndexError("user index out of range")
        if np.any(services < 0) or np.any(services >= self.num_services):
            raise IndexError("service index out of range")
        if users.size == 0:
            return np.empty(0, dtype=np.float32)
        predictions = np.empty(users.size, dtype=np.float32)
        shared = self._server_shared_bundle()
        for user in np.unique(users):
            selected = users == int(user)
            prediction, _ = self._forward(
                self.private_user_gmf[int(user)],
                self.private_user_mlp[int(user)],
                shared,
                services[selected],
            )
            predictions[selected] = prediction
        if not np.all(np.isfinite(predictions)):
            raise FloatingPointError("FedRecGEL predictions are not finite")
        return predictions.astype(np.float32)

    def reconstruction_loss(
        self, users: Iterable[int], services: Iterable[int], targets: Iterable[float]
    ) -> float:
        targets = np.asarray(targets, dtype=np.float32)
        predictions = self.predict(users, services)
        if predictions.size != targets.size:
            raise ValueError("targets and predictions must be aligned")
        if predictions.size == 0:
            return 0.0
        return float(0.5 * np.mean((predictions.astype(np.float64) - targets.astype(np.float64)) ** 2))

    def support_reconstruction_metrics(
        self, users: Iterable[int], services: Iterable[int], targets: Iterable[float]
    ) -> dict:
        targets = np.asarray(targets, dtype=np.float32)
        predictions = self.predict(users, services)
        if predictions.size == 0:
            return {
                "support_loss": 0.0,
                "support_prediction_mean": 0.0,
                "support_target_mean": 0.0,
                "support_prediction_min": 0.0,
                "support_prediction_max": 0.0,
                "support_prediction_boundary_fraction": 0.0,
            }
        errors = predictions.astype(np.float64) - targets.astype(np.float64)
        boundary = (predictions <= _OUTPUT_BOUNDARY_EPS) | (
            predictions >= 1.0 - _OUTPUT_BOUNDARY_EPS
        )
        return {
            "support_loss": float(0.5 * np.mean(errors * errors)),
            "support_prediction_mean": float(np.mean(predictions, dtype=np.float64)),
            "support_target_mean": float(np.mean(targets, dtype=np.float64)),
            "support_prediction_min": float(np.min(predictions)),
            "support_prediction_max": float(np.max(predictions)),
            "support_prediction_boundary_fraction": float(np.mean(boundary)),
        }


# A short alias is convenient in runners and keeps the method name explicit.
FedRecGEL = FedRecGELQoS
