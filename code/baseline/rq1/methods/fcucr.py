"""Independent causal NumPy adaptation of FCUCR for QoS regression.

The public FCUCR repository is used only as an algorithm reference.  This
module is a clean implementation for the WSDream user--service--time stream;
it does not copy the unlicensed repository code.  The adaptation keeps the
two FCUCR mechanisms that are meaningful for QoS:

* a shared service/time representation and a private user prediction state;
* previous-slice representation self-distillation; and
* a server prototype knowledge base that is queried only from ``KB(<t)``.

The local QoS objective uses the public FCUCR ratio/cap distillation scaling,
and local clients use Adam with beta=(0.9, 0.98).  Private Adam moments remain
with their user across slices; only the configured shared representation is
aggregated.

The model consumes normalized fixed-log1p QoS targets in ``[0, 1]``.  A
client's query labels are never passed to this module.  ``update_slice``
trains on the supplied support tuples, aggregates only the shared arrays,
then writes the post-update support prototype to the knowledge base.  The
``predict`` path is read-only and therefore can be audited byte-for-byte.

Because a zero-logit sigmoid maps to a badly mis-scaled raw QoS value under
the frozen inverse-log transform, each client's private output bias is
calibrated exactly once from that client's first non-empty support batch.
This scale initialization never reads query or future labels.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np


FP32_BYTES = 4
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.98
_ADAM_EPS = 1e-8
_SCALE_TARGET_EPS = 1e-6

# The QoS migration needs an explicit scale calibration because the sigmoid
# head is evaluated in fixed-log1p space.  This calibration is deliberately
# based only on the first support batch seen by each client.
SCALE_INITIALIZATION_STRATEGY = (
    "support_only_first_participation_target_mean_logit_centering"
)
TASD_STRATEGY = "previous_slice_shared_representation_mse"
MIN_TASD_EPOCHS = 2


def sigmoid(values):
    """Numerically stable sigmoid for scalars or arrays."""
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, -40.0, 40.0)
    result = np.empty_like(clipped)
    positive = clipped >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-clipped[positive]))
    exp_values = np.exp(clipped[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


@dataclass(frozen=True)
class PrototypeEntry:
    """One immutable-in-meaning server-side prototype record."""

    user_id: int
    time_slice: int
    prototype: np.ndarray


@dataclass(frozen=True)
class FCUCRRoundResult:
    """Auditable counters and support-only diagnostics for one time slice."""

    support_entries: int
    participating_clients: int
    kb_entries_before: int
    kb_entries_after: int
    retrieved_prototypes: int
    support_loss_before: float
    support_loss_after: float
    distillation_loss: float
    uplink_bytes: int
    downlink_bytes: int

    def as_dict(self) -> dict:
        return {
            "support_entries": int(self.support_entries),
            "participating_clients": int(self.participating_clients),
            "kb_entries_before": int(self.kb_entries_before),
            "kb_entries_after": int(self.kb_entries_after),
            "retrieved_prototypes": int(self.retrieved_prototypes),
            "support_loss_before": float(self.support_loss_before),
            "support_loss_after": float(self.support_loss_after),
            "distillation_loss": float(self.distillation_loss),
            "uplink_bytes": int(self.uplink_bytes),
            "downlink_bytes": int(self.downlink_bytes),
        }


class FCUCRQoS:
    """FCUCR-style continual federated QoS regressor.

    ``shared_service_factors`` and ``shared_time_factors`` are the analogue
    of FCUCR's shared representation module.  ``private_user_factors``,
    ``private_user_bias`` and ``private_prototype_gates`` (including their
    Adam moments) are kept per client and are never averaged by the server.
    """

    scale_initialization_strategy = SCALE_INITIALIZATION_STRATEGY
    tasd_strategy = TASD_STRATEGY

    def __init__(
        self,
        num_users: int,
        num_services: int,
        num_times: int,
        rank: int = 10,
        learning_rate: float = 0.01,
        regularization: float = 0.001,
        distillation_weight: float = 1.0,
        distil_cap: float = 10.0,
        top_k: int = 20,
        kb_window: int = 0,
        aggregation_mode: str = "equal",
        seed: int = 0,
    ):
        if min(int(num_users), int(num_services), int(num_times), int(rank)) <= 0:
            raise ValueError("dimensions and rank must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")
        if not np.isfinite(distillation_weight) or distillation_weight < 0:
            raise ValueError("distillation_weight must be finite and nonnegative")
        if not np.isfinite(distil_cap) or distil_cap < 0:
            raise ValueError("distil_cap must be finite and nonnegative")
        if int(top_k) < 0:
            raise ValueError("top_k must be nonnegative")
        if int(kb_window) < 0:
            raise ValueError("kb_window must be nonnegative")
        if aggregation_mode not in {"equal", "sample_weighted"}:
            raise ValueError("aggregation_mode must be 'equal' or 'sample_weighted'")

        self.num_users = int(num_users)
        self.num_services = int(num_services)
        self.num_times = int(num_times)
        self.rank = int(rank)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.distillation_weight = float(distillation_weight)
        self.distil_cap = float(distil_cap)
        self.top_k = int(top_k)
        self.kb_window = int(kb_window)
        self.aggregation_mode = str(aggregation_mode)
        self.rng = np.random.default_rng(int(seed))

        self.private_user_factors = self.rng.normal(
            0.0, 0.05, (self.num_users, self.rank)
        ).astype(np.float32)
        self.private_user_bias = np.zeros(self.num_users, dtype=np.float32)
        # Causal scale-calibration state.  Each client is initialized exactly
        # once, at its first support-only participation.  The diagnostic
        # arrays retain the calibration inputs for auditability without ever
        # reading query or future labels.
        self.scale_initialized = np.zeros(self.num_users, dtype=bool)
        self.scale_initialization_target_mean = np.zeros(
            self.num_users, dtype=np.float32
        )
        self.scale_initialization_nonbias_logit_mean = np.zeros(
            self.num_users, dtype=np.float32
        )
        self.scale_initialization_bias = np.zeros(
            self.num_users, dtype=np.float32
        )
        self.private_prototype_gates = self.rng.normal(
            0.0, 0.05, (self.num_users, self.rank)
        ).astype(np.float32)
        # Adam moments for private client state persist across time slices.
        # They are not transmitted or averaged, but are part of the causal
        # model state and therefore must be included in read-only audits.
        self._private_user_factors_m = np.zeros_like(self.private_user_factors)
        self._private_user_factors_v = np.zeros_like(self.private_user_factors)
        self._private_user_bias_m = np.zeros_like(self.private_user_bias)
        self._private_user_bias_v = np.zeros_like(self.private_user_bias)
        self._private_prototype_gates_m = np.zeros_like(self.private_prototype_gates)
        self._private_prototype_gates_v = np.zeros_like(self.private_prototype_gates)
        self._private_optimizer_steps = np.zeros(self.num_users, dtype=np.int64)
        self.shared_service_factors = self.rng.normal(
            0.0, 0.05, (self.num_services, self.rank)
        ).astype(np.float32)
        self.shared_time_factors = self.rng.normal(
            0.0, 0.05, (self.num_times, self.rank)
        ).astype(np.float32)

        # These are always independent copies.  The first slice has no
        # distillation term, but retaining a teacher array makes the state
        # explicit and prevents accidental aliasing in later slices.
        self.teacher_service_factors = self.shared_service_factors.copy()
        self.teacher_time_factors = self.shared_time_factors.copy()
        self.prototype_kb: list[PrototypeEntry] = []
        self.current_prototypes: dict[int, np.ndarray] = {}
        self.current_fused_prototypes: dict[int, np.ndarray] = {}
        self.round_index = 0

    @property
    def shared_parameter_bytes(self) -> int:
        return int(
            (self.shared_service_factors.size + self.shared_time_factors.size)
            * FP32_BYTES
        )

    @property
    def prototype_bytes(self) -> int:
        return int(self.rank * FP32_BYTES)

    @property
    def scale_initialization_count(self) -> int:
        """Return the number of clients calibrated from support data."""
        return int(np.count_nonzero(self.scale_initialized))

    def state_arrays(self) -> tuple[np.ndarray, ...]:
        """Return every mutable array relevant to causal/read-only audits.

        Private Adam moments, teacher arrays, and all KB/current prototype
        arrays are deliberately included.  Callers must treat the returned
        arrays as read-only.
        """
        arrays = [
            self.private_user_factors,
            self.private_user_bias,
            self.scale_initialized,
            self.scale_initialization_target_mean,
            self.scale_initialization_nonbias_logit_mean,
            self.scale_initialization_bias,
            self.private_prototype_gates,
            self._private_user_factors_m,
            self._private_user_factors_v,
            self._private_user_bias_m,
            self._private_user_bias_v,
            self._private_prototype_gates_m,
            self._private_prototype_gates_v,
            self._private_optimizer_steps,
            self.shared_service_factors,
            self.shared_time_factors,
            self.teacher_service_factors,
            self.teacher_time_factors,
        ]
        arrays.extend(entry.prototype for entry in self.prototype_kb)
        arrays.extend(self.current_prototypes.values())
        arrays.extend(self.current_fused_prototypes.values())
        return tuple(arrays)

    def assert_finite_state(self) -> None:
        """Raise if any mutable model, teacher, or KB array is non-finite.

        The check is intentionally over the complete state exposed by
        :meth:`state_arrays`, including retained prototypes.  It is called
        after each support update so a numerical failure cannot be mistaken
        for a valid causal round.
        """
        for index, array in enumerate(self.state_arrays()):
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(
                    f"non-finite FCUCR state array after update (index {index})"
                )

    def state_copy(self) -> dict:
        """Deep-copy model, teacher, KB and RNG state."""
        return {
            "private_user_factors": self.private_user_factors.copy(),
            "private_user_bias": self.private_user_bias.copy(),
            "scale_initialized": self.scale_initialized.copy(),
            "scale_initialization_target_mean": self.scale_initialization_target_mean.copy(),
            "scale_initialization_nonbias_logit_mean": self.scale_initialization_nonbias_logit_mean.copy(),
            "scale_initialization_bias": self.scale_initialization_bias.copy(),
            "private_prototype_gates": self.private_prototype_gates.copy(),
            "_private_user_factors_m": self._private_user_factors_m.copy(),
            "_private_user_factors_v": self._private_user_factors_v.copy(),
            "_private_user_bias_m": self._private_user_bias_m.copy(),
            "_private_user_bias_v": self._private_user_bias_v.copy(),
            "_private_prototype_gates_m": self._private_prototype_gates_m.copy(),
            "_private_prototype_gates_v": self._private_prototype_gates_v.copy(),
            "_private_optimizer_steps": self._private_optimizer_steps.copy(),
            "shared_service_factors": self.shared_service_factors.copy(),
            "shared_time_factors": self.shared_time_factors.copy(),
            "teacher_service_factors": self.teacher_service_factors.copy(),
            "teacher_time_factors": self.teacher_time_factors.copy(),
            "prototype_kb": [
                (entry.user_id, entry.time_slice, entry.prototype.copy())
                for entry in self.prototype_kb
            ],
            "current_prototypes": {
                int(user): value.copy() for user, value in self.current_prototypes.items()
            },
            "current_fused_prototypes": {
                int(user): value.copy()
                for user, value in self.current_fused_prototypes.items()
            },
            "round_index": int(self.round_index),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def state_unchanged(self, snapshot: dict) -> bool:
        """Return whether a read-only prediction changed any model state."""
        for name in (
            "private_user_factors",
            "private_user_bias",
            "scale_initialized",
            "scale_initialization_target_mean",
            "scale_initialization_nonbias_logit_mean",
            "scale_initialization_bias",
            "private_prototype_gates",
            "_private_user_factors_m",
            "_private_user_factors_v",
            "_private_user_bias_m",
            "_private_user_bias_v",
            "_private_prototype_gates_m",
            "_private_prototype_gates_v",
            "_private_optimizer_steps",
            "shared_service_factors",
            "shared_time_factors",
            "teacher_service_factors",
            "teacher_time_factors",
        ):
            if not np.array_equal(getattr(self, name), snapshot[name]):
                return False
        if self.round_index != snapshot["round_index"]:
            return False
        if self.rng.bit_generator.state != snapshot["rng_state"]:
            return False

        current_kb = [
            (entry.user_id, entry.time_slice, entry.prototype)
            for entry in self.prototype_kb
        ]
        old_kb = snapshot["prototype_kb"]
        if len(current_kb) != len(old_kb):
            return False
        for (user_a, time_a, value_a), (user_b, time_b, value_b) in zip(
            current_kb, old_kb
        ):
            if (user_a, time_a) != (user_b, time_b) or not np.array_equal(
                value_a, value_b
            ):
                return False
        for name in ("current_prototypes", "current_fused_prototypes"):
            current = getattr(self, name)
            old = snapshot[name]
            if current.keys() != old.keys():
                return False
            if not all(np.array_equal(current[key], old[key]) for key in current):
                return False
        return True

    def _validate_indices(self, users, services, time_slices, targets=None):
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        times = np.asarray(time_slices, dtype=np.int64)
        if users.ndim != 1 or services.shape != users.shape or times.shape != users.shape:
            raise ValueError("users, services and time_slices must be aligned vectors")
        if targets is not None:
            targets = np.asarray(targets, dtype=np.float32)
            if targets.shape != users.shape:
                raise ValueError("targets must align with users")
            if not np.all(np.isfinite(targets)) or np.any(targets < 0.0) or np.any(targets > 1.0):
                raise ValueError("targets must be finite normalized values in [0, 1]")
        if users.size:
            if np.any(users < 0) or np.any(users >= self.num_users):
                raise IndexError("user index out of range")
            if np.any(services < 0) or np.any(services >= self.num_services):
                raise IndexError("service index out of range")
            if np.any(times < 0) or np.any(times >= self.num_times):
                raise IndexError("time index out of range")
        return users, services, times, targets

    def _support_prototype(self, services, time_slices, service_factors=None, time_factors=None):
        services = np.asarray(services, dtype=np.int64)
        times = np.asarray(time_slices, dtype=np.int64)
        if service_factors is None:
            service_factors = self.shared_service_factors
        if time_factors is None:
            time_factors = self.shared_time_factors
        if services.size == 0:
            return np.zeros(self.rank, dtype=np.float32)
        representation = service_factors[services] + time_factors[times]
        return np.mean(representation, axis=0, dtype=np.float64).astype(np.float32)

    def retrieve_prototype(self, user_id: int, time_slice: int, query_prototype):
        """Retrieve a softmax-fused prototype from strict ``KB(<time_slice)``.

        The metadata is returned for tests and communication/audit reporting.
        Candidate ordering is deterministic under equal cosine scores.
        """
        user_id = int(user_id)
        time_slice = int(time_slice)
        query_prototype = np.asarray(query_prototype, dtype=np.float32)
        if query_prototype.shape != (self.rank,) or not np.all(np.isfinite(query_prototype)):
            raise ValueError("query_prototype must be a finite rank vector")
        if not 0 <= user_id < self.num_users:
            raise IndexError("user index out of range")
        if time_slice < 0:
            raise ValueError("time_slice must be nonnegative")

        query_norm = float(np.linalg.norm(query_prototype))
        candidates = []
        for position, entry in enumerate(self.prototype_kb):
            # This is the causal boundary: no current/future entry and no own
            # client prototype may influence the current slice.
            if entry.time_slice >= time_slice or entry.user_id == user_id:
                continue
            entry_norm = float(np.linalg.norm(entry.prototype))
            if query_norm == 0.0 or entry_norm == 0.0:
                similarity = 0.0
            else:
                similarity = float(
                    np.dot(query_prototype, entry.prototype) / (query_norm * entry_norm)
                )
            candidates.append((similarity, entry.time_slice, entry.user_id, position, entry))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        selected = candidates[: self.top_k]
        fused = np.zeros(self.rank, dtype=np.float32)
        weights = np.zeros(len(selected), dtype=np.float64)
        if selected:
            scores = np.asarray([item[0] for item in selected], dtype=np.float64)
            scores -= np.max(scores)
            weights = np.exp(scores)
            weights /= np.sum(weights)
            for weight, item in zip(weights, selected):
                fused += np.float32(weight) * item[4].prototype

        metadata = {
            "candidate_count": int(len(candidates)),
            "selected_count": int(len(selected)),
            "selected_users": tuple(int(item[2]) for item in selected),
            "selected_times": tuple(int(item[1]) for item in selected),
            "weights": tuple(float(value) for value in weights),
        }
        return fused, metadata

    def _predict_one(
        self,
        user: int,
        service: int,
        time_slice: int,
        local_prototype: np.ndarray,
        fused_prototype: np.ndarray,
        service_factors: np.ndarray,
        time_factors: np.ndarray,
        user_factors: np.ndarray,
        user_bias: np.ndarray,
        prototype_gates: np.ndarray,
    ) -> float:
        representation = service_factors[service] + time_factors[time_slice]
        local_feature = representation + local_prototype
        logit = (
            float(np.dot(user_factors[user], local_feature))
            + float(user_bias[user])
            + float(np.dot(prototype_gates[user], fused_prototype))
        )
        return float(sigmoid(np.asarray([logit]))[0])

    def _loss_for_user(
        self,
        user: int,
        services,
        time_slices,
        targets,
        local_prototype,
        fused_prototype,
        service_factors,
        time_factors,
        user_factors,
        user_bias,
        prototype_gates,
    ) -> float:
        if len(targets) == 0:
            return 0.0
        predictions = [
            self._predict_one(
                user,
                int(service),
                int(time_slice),
                local_prototype,
                fused_prototype,
                service_factors,
                time_factors,
                user_factors,
                user_bias,
                prototype_gates,
            )
            for service, time_slice in zip(services, time_slices)
        ]
        return float(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))

    def _initialize_client_scale(
        self,
        user: int,
        services,
        time_slices,
        targets,
        local_prototype: np.ndarray,
        fused_prototype: np.ndarray,
        service_factors: np.ndarray,
        time_factors: np.ndarray,
        user_factor: np.ndarray,
        prototype_gate: np.ndarray,
        user_bias: np.ndarray,
    ) -> None:
        """Calibrate a first-time client's sigmoid scale from support only.

        ``targets`` are already fixed-log1p normalized support targets.  The
        target mean determines the desired logit; the mean current logit with
        the bias removed is subtracted so the private bias performs a pure
        scale/offset calibration.  No query, future, or KB label is read.
        """
        user = int(user)
        if self.scale_initialized[user]:
            return
        targets = np.asarray(targets, dtype=np.float64)
        services = np.asarray(services, dtype=np.int64)
        time_slices = np.asarray(time_slices, dtype=np.int64)
        if targets.size == 0:
            raise ValueError("cannot scale-initialize a client without support")

        target_mean = float(np.mean(targets, dtype=np.float64))
        target_mean = float(
            np.clip(target_mean, _SCALE_TARGET_EPS, 1.0 - _SCALE_TARGET_EPS)
        )
        target_logit = math.log(target_mean) - math.log1p(-target_mean)

        representation = service_factors[services] + time_factors[time_slices]
        local_features = representation + np.asarray(local_prototype, dtype=np.float32)
        nonbias_logits = np.asarray(
            np.einsum("ij,j->i", local_features, user_factor)
            + float(np.dot(prototype_gate, fused_prototype)),
            dtype=np.float64,
        )
        nonbias_logit_mean = float(np.mean(nonbias_logits, dtype=np.float64))
        calibrated_bias = target_logit - nonbias_logit_mean
        if not np.isfinite(calibrated_bias):
            raise FloatingPointError("non-finite support-only FCUCR scale calibration")

        user_bias[0] = np.float32(calibrated_bias)
        # Persist the calibration before local optimization so the complete
        # first-participation state is visible to causal/read-only audits.
        self.private_user_bias[user] = np.float32(calibrated_bias)
        self.scale_initialized[user] = True
        self.scale_initialization_target_mean[user] = np.float32(target_mean)
        self.scale_initialization_nonbias_logit_mean[user] = np.float32(
            nonbias_logit_mean
        )
        self.scale_initialization_bias[user] = np.float32(calibrated_bias)

    @staticmethod
    def _adam_update(
        array: np.ndarray,
        gradient: np.ndarray,
        first_moment: np.ndarray,
        second_moment: np.ndarray,
        step: int,
        learning_rate: float,
    ) -> None:
        """Apply one official-parameterized Adam update in place.

        FCUCR's public implementation constructs Adam with beta=(0.9, 0.98).
        The shared local moments are temporary per-client state, while the
        private moments passed by the caller persist across slices.
        """
        if int(step) <= 0:
            raise ValueError("Adam step must be positive")
        gradient = np.asarray(gradient, dtype=np.float32)
        if gradient.shape != array.shape:
            raise ValueError("Adam gradient and parameter shapes must match")
        first_moment *= np.float32(_ADAM_BETA1)
        first_moment += np.float32(1.0 - _ADAM_BETA1) * gradient
        second_moment *= np.float32(_ADAM_BETA2)
        second_moment += np.float32(1.0 - _ADAM_BETA2) * (gradient * gradient)
        correction_1 = 1.0 - _ADAM_BETA1 ** int(step)
        correction_2 = 1.0 - _ADAM_BETA2 ** int(step)
        first_hat = first_moment / np.float32(correction_1)
        second_hat = second_moment / np.float32(correction_2)
        array -= np.float32(learning_rate) * first_hat / (
            np.sqrt(second_hat) + np.float32(_ADAM_EPS)
        )

    def _distillation_coefficient(self, base_loss: float, distillation_loss: float) -> float:
        """Return ``lambda * min(base/(dist+eps), distil_cap)``."""
        base_loss = float(base_loss)
        distillation_loss = float(distillation_loss)
        if not np.isfinite(base_loss) or base_loss < 0.0:
            raise ValueError("base_loss must be finite and nonnegative")
        if not np.isfinite(distillation_loss) or distillation_loss < 0.0:
            raise ValueError("distillation_loss must be finite and nonnegative")
        ratio = base_loss / (distillation_loss + 1e-8)
        return float(self.distillation_weight * min(ratio, self.distil_cap))

    def _aggregation_weights(self, participant_weights) -> np.ndarray:
        """Return equal-client weights by default, with an explicit sample mode."""
        counts = np.asarray(participant_weights, dtype=np.float64)
        if counts.ndim != 1:
            raise ValueError("participant weights must be a vector")
        if counts.size == 0:
            return counts
        if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
            raise ValueError("participant weights must be finite and nonnegative")
        if self.aggregation_mode == "equal":
            return np.full(counts.size, 1.0 / float(counts.size), dtype=np.float64)
        total = float(np.sum(counts))
        if total <= 0.0:
            return np.full(counts.size, 1.0 / float(counts.size), dtype=np.float64)
        return counts / total

    def update_slice(self, users, services, time_slice: int, targets, epochs: int = 4) -> FCUCRRoundResult:
        """Train one causal support slice and append post-update prototypes."""
        time_slice = int(time_slice)
        if time_slice != self.round_index:
            raise ValueError(
                f"time slices must be processed in order: expected {self.round_index}, got {time_slice}"
            )
        epochs = int(epochs)
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        users = np.asarray(users, dtype=np.int64)
        services = np.asarray(services, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float32)
        times = np.full(users.shape, time_slice, dtype=np.int64)
        users, services, times, targets = self._validate_indices(
            users, services, times, targets
        )

        participants = np.unique(users)
        if (
            time_slice > 0
            and participants.size
            and self.distillation_weight > 0.0
            and self.distil_cap > 0.0
            and epochs < MIN_TASD_EPOCHS
        ):
            raise ValueError(
                "TASD requires at least 2 local epochs when a later slice has "
                "participating clients and active distillation"
            )

        kb_before = len(self.prototype_kb)
        # Deep copy the previous global representation before any client update.
        # It is the teacher for this slice and cannot be modified through local
        # or aggregation arrays.
        self.teacher_service_factors = self.shared_service_factors.copy()
        self.teacher_time_factors = self.shared_time_factors.copy()
        distillation_enabled = (
            time_slice > 0
            and self.distillation_weight > 0.0
            and self.distil_cap > 0.0
        )

        local_shared_services = []
        local_shared_times = []
        participant_weights = []
        retrievals = {}
        local_prototypes = {}
        local_fused = {}
        support_loss_before = []
        distillation_total = 0.0
        distillation_count = 0

        for user_value in participants:
            user = int(user_value)
            selected = users == user
            user_services = services[selected]
            user_times = times[selected]
            user_targets = targets[selected]
            local_services_factors = self.shared_service_factors.copy()
            local_time_factors = self.shared_time_factors.copy()
            user_factor = self.private_user_factors[user].copy()
            user_bias = np.asarray([self.private_user_bias[user]], dtype=np.float32)
            prototype_gate = self.private_prototype_gates[user].copy()
            local_service_m = np.zeros_like(local_services_factors)
            local_service_v = np.zeros_like(local_services_factors)
            local_time_m = np.zeros_like(local_time_factors)
            local_time_v = np.zeros_like(local_time_factors)
            local_optimizer_step = 0

            local_prototype = self._support_prototype(
                user_services,
                user_times,
                local_services_factors,
                local_time_factors,
            )
            # Retrieval and top-k softmax fusion are server-side.  The client
            # receives only this one fused rank vector; individual KB entries
            # are never part of the client payload.
            fused_prototype, retrieval = self.retrieve_prototype(
                user, time_slice, local_prototype
            )
            retrievals[user] = retrieval
            local_prototypes[user] = local_prototype
            local_fused[user] = fused_prototype
            self._initialize_client_scale(
                user,
                user_services,
                user_times,
                user_targets,
                local_prototype,
                fused_prototype,
                local_services_factors,
                local_time_factors,
                user_factor,
                prototype_gate,
                user_bias,
            )
            support_loss_before.append(
                self._loss_for_user(
                    user,
                    user_services,
                    user_times,
                    user_targets,
                    local_prototype,
                    fused_prototype,
                    local_services_factors,
                    local_time_factors,
                    self.private_user_factors,
                    self.private_user_bias,
                    self.private_prototype_gates,
                )
            )

            for _ in range(epochs):
                # Match the public client's one optimizer step per local epoch:
                # form a support-batch QoS loss and its representation teacher
                # loss before applying Adam to the local copy.
                support_count = int(len(user_services))
                service_grad = np.zeros_like(local_services_factors)
                time_grad = np.zeros(self.rank, dtype=np.float32)
                user_factor_grad = np.zeros_like(user_factor)
                user_bias_grad = np.zeros(1, dtype=np.float32)
                prototype_gate_grad = np.zeros_like(prototype_gate)
                residuals = []
                differences = []
                local_features = []
                dlogits = []
                for service_value, target_value in zip(user_services, user_targets):
                    service = int(service_value)
                    target = float(target_value)
                    representation = (
                        local_services_factors[service]
                        + local_time_factors[time_slice]
                    )
                    local_feature = representation + local_prototype
                    logit = (
                        float(np.dot(user_factor, local_feature))
                        + float(user_bias[0])
                        + float(np.dot(prototype_gate, fused_prototype))
                    )
                    prediction = float(sigmoid(np.asarray([logit]))[0])
                    error = prediction - target
                    # The QoS base loss is mean squared error, so its
                    # derivative contributes the factor two here.
                    dlogit = 2.0 * error * prediction * (1.0 - prediction)
                    residuals.append(error)
                    local_features.append(local_feature)
                    dlogits.append(dlogit)
                    if distillation_enabled:
                        teacher_representation = (
                            self.teacher_service_factors[service]
                            + self.teacher_time_factors[time_slice]
                        )
                        differences.append(representation - teacher_representation)

                base_loss = float(np.mean(np.asarray(residuals, dtype=np.float64) ** 2))
                if distillation_enabled:
                    distillation_values = np.asarray(differences, dtype=np.float64)
                    distillation_loss = float(np.mean(distillation_values * distillation_values))
                    coefficient = self._distillation_coefficient(
                        base_loss, distillation_loss
                    )
                    distillation_total += float(np.sum(distillation_values * distillation_values))
                    distillation_count += support_count
                else:
                    distillation_loss = 0.0
                    coefficient = 0.0

                for index, (service_value, dlogit, local_feature) in enumerate(
                    zip(user_services, dlogits, local_features)
                ):
                    service = int(service_value)
                    grad_shared = np.float32(dlogit) * user_factor
                    if distillation_enabled:
                        grad_shared = grad_shared + (
                            np.float32(coefficient * 2.0 / float(self.rank))
                            * differences[index].astype(np.float32)
                        )
                    service_grad[service] += grad_shared
                    time_grad += grad_shared
                    user_factor_grad += np.float32(dlogit) * local_feature
                    user_bias_grad[0] += np.float32(dlogit)
                    prototype_gate_grad += np.float32(dlogit) * fused_prototype

                service_grad /= np.float32(support_count)
                time_grad /= np.float32(support_count)
                user_factor_grad /= np.float32(support_count)
                user_bias_grad /= np.float32(support_count)
                prototype_gate_grad /= np.float32(support_count)
                for service in np.unique(user_services):
                    service = int(service)
                    service_grad[service] += (
                        self.regularization * local_services_factors[service]
                    ).astype(np.float32)
                time_grad += (
                    self.regularization * local_time_factors[time_slice]
                ).astype(np.float32)
                user_factor_grad += (
                    self.regularization * user_factor
                ).astype(np.float32)
                user_bias_grad[0] += np.float32(
                    self.regularization * user_bias[0]
                )
                prototype_gate_grad += (
                    self.regularization * prototype_gate
                ).astype(np.float32)

                local_optimizer_step += 1
                for service in np.unique(user_services):
                    service = int(service)
                    self._adam_update(
                        local_services_factors[service],
                        service_grad[service],
                        local_service_m[service],
                        local_service_v[service],
                        local_optimizer_step,
                        self.learning_rate,
                    )
                self._adam_update(
                    local_time_factors[time_slice],
                    time_grad,
                    local_time_m[time_slice],
                    local_time_v[time_slice],
                    local_optimizer_step,
                    self.learning_rate,
                )

                private_step = int(self._private_optimizer_steps[user]) + 1
                self._private_optimizer_steps[user] = private_step
                self._adam_update(
                    user_factor,
                    user_factor_grad,
                    self._private_user_factors_m[user],
                    self._private_user_factors_v[user],
                    private_step,
                    self.learning_rate,
                )
                self._adam_update(
                    user_bias,
                    user_bias_grad,
                    self._private_user_bias_m[user : user + 1],
                    self._private_user_bias_v[user : user + 1],
                    private_step,
                    self.learning_rate,
                )
                self._adam_update(
                    prototype_gate,
                    prototype_gate_grad,
                    self._private_prototype_gates_m[user],
                    self._private_prototype_gates_v[user],
                    private_step,
                    self.learning_rate,
                )

            # Private parameters are retained by this client and never sent to
            # the server.  Shared arrays are collected for configured FedAvg.
            self.private_user_factors[user] = user_factor
            self.private_user_bias[user] = np.float32(user_bias[0])
            self.private_prototype_gates[user] = prototype_gate
            local_shared_services.append(local_services_factors)
            local_shared_times.append(local_time_factors)
            participant_weights.append(float(len(user_targets)))

        if participants.size:
            weights = self._aggregation_weights(participant_weights)
            self.shared_service_factors = np.tensordot(
                weights, np.stack(local_shared_services), axes=(0, 0)
            ).astype(np.float32)
            self.shared_time_factors = np.tensordot(
                weights, np.stack(local_shared_times), axes=(0, 0)
            ).astype(np.float32)

        # Use the post-aggregation shared representation for the current
        # support prototype, then append it.  Therefore current prototypes can
        # never be selected by this slice's retrieval.
        self.current_prototypes = {}
        self.current_fused_prototypes = {
            user: fused.copy() for user, fused in local_fused.items()
        }
        support_loss_after = []
        for user_value in participants:
            user = int(user_value)
            selected = users == user
            user_services = services[selected]
            user_times = times[selected]
            user_targets = targets[selected]
            post_prototype = self._support_prototype(
                user_services,
                user_times,
                self.shared_service_factors,
                self.shared_time_factors,
            )
            self.current_prototypes[user] = post_prototype.copy()
            support_loss_after.append(
                self._loss_for_user(
                    user,
                    user_services,
                    user_times,
                    user_targets,
                    post_prototype,
                    local_fused[user],
                    self.shared_service_factors,
                    self.shared_time_factors,
                    self.private_user_factors,
                    self.private_user_bias,
                    self.private_prototype_gates,
                )
            )
            self.prototype_kb.append(
                PrototypeEntry(user, time_slice, post_prototype.copy())
            )

        if self.kb_window > 0:
            cutoff = time_slice - self.kb_window + 1
            self.prototype_kb = [
                entry for entry in self.prototype_kb if entry.time_slice >= cutoff
            ]

        # Validate the complete post-update state before advancing the causal
        # clock or returning counters.  This includes teacher and KB arrays.
        self.assert_finite_state()
        self.round_index += 1
        participant_count = int(len(participants))
        shared_payload = self.shared_parameter_bytes
        prototype_payload = self.prototype_bytes
        payload_per_client = shared_payload + prototype_payload
        result = FCUCRRoundResult(
            support_entries=int(len(users)),
            participating_clients=participant_count,
            kb_entries_before=int(kb_before),
            kb_entries_after=int(len(self.prototype_kb)),
            retrieved_prototypes=int(
                sum(item["selected_count"] for item in retrievals.values())
            ),
            support_loss_before=float(np.mean(support_loss_before))
            if support_loss_before
            else 0.0,
            support_loss_after=float(np.mean(support_loss_after))
            if support_loss_after
            else 0.0,
            distillation_loss=float(distillation_total / distillation_count)
            if distillation_count
            else 0.0,
            uplink_bytes=int(participant_count * payload_per_client),
            downlink_bytes=int(participant_count * payload_per_client),
        )
        return result

    def predict(self, users, services, time_slices, prototypes=None, fused_prototypes=None):
        """Predict normalized QoS without changing any model/KB state."""
        users, services, times, _ = self._validate_indices(
            users, services, time_slices, targets=None
        )
        if prototypes is None:
            prototypes = self.current_prototypes
        if fused_prototypes is None:
            fused_prototypes = self.current_fused_prototypes

        zero = np.zeros(self.rank, dtype=np.float32)
        predicted = np.empty(len(users), dtype=np.float32)
        for index, (user_value, service_value, time_value) in enumerate(
            zip(users, services, times)
        ):
            user = int(user_value)
            if isinstance(prototypes, dict):
                local_prototype = np.asarray(prototypes.get(user, zero), dtype=np.float32)
            else:
                local_prototype = np.asarray(prototypes[index], dtype=np.float32)
            if isinstance(fused_prototypes, dict):
                fused_prototype = np.asarray(
                    fused_prototypes.get(user, zero), dtype=np.float32
                )
            else:
                fused_prototype = np.asarray(fused_prototypes[index], dtype=np.float32)
            if local_prototype.shape != (self.rank,) or fused_prototype.shape != (self.rank,):
                raise ValueError("prediction prototypes must be rank vectors")
            predicted[index] = np.float32(
                self._predict_one(
                    user,
                    int(service_value),
                    int(time_value),
                    local_prototype,
                    fused_prototype,
                    self.shared_service_factors,
                    self.shared_time_factors,
                    self.private_user_factors,
                    self.private_user_bias,
                    self.private_prototype_gates,
                )
            )
        return predicted

    def prototype_kb_snapshot(self) -> list[tuple[int, int, np.ndarray]]:
        """Return a defensive copy useful for tests and artifact audits."""
        return [
            (entry.user_id, entry.time_slice, entry.prototype.copy())
            for entry in self.prototype_kb
        ]


# Descriptive aliases make the method easy to locate from runners and tables.
FederatedContinualUserCentricQoS = FCUCRQoS
