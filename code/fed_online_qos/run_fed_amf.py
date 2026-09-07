"""Causal Fed-AMF runner with shared and Local C-OMF service-state modes."""

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from csotu_update import (output_trust_budget, support_output_trust_update,
                          support_prediction_rms_change)
from service_raw_residual_calibration import (apply_dual_constrained_log_service_residual,
                                              apply_log_domain_service_residual,
                                              apply_prediction_conditioned_service_residual,
                                              apply_service_raw_residual,
                                              log_domain_service_residual,
                                              log_domain_service_residual_extra_bytes,
                                              mean_service_residual,
                                              prediction_conditioned_service_residual,
                                              prediction_conditioned_service_residual_extra_bytes,
                                              reliable_service_residual,
                                              service_raw_residual_extra_bytes,
                                              variance_aware_service_residual)
from event_sparse_service_aggregation import plan_service_downlink
from output_trust_bounded_compression import bounded_output_trust_compress_client_block
from prepare_rt_protocol import (NUM_SERVICES, NUM_TIMES, NUM_USERS, RT_UPPER_EXCLUSIVE,
                                 CausalDriftScorer as RT_CausalDriftScorer,
                                 drift_groups as RT_drift_groups,
                                 fixed_log_transform as RT_fixed_log_transform,
                                 inverse_fixed_log_transform as RT_inverse_fixed_log_transform)
from prepare_tp_protocol import (NUM_SERVICES as TP_NUM_SERVICES, NUM_TIMES as TP_NUM_TIMES,
                                 NUM_USERS as TP_NUM_USERS, TP_LOG_UPPER, TP_UPPER_EXCLUSIVE,
                                 CausalDriftScorer as TP_CausalDriftScorer,
                                 drift_groups as TP_drift_groups,
                                 fixed_log_transform as TP_fixed_log_transform,
                                 inverse_fixed_log_transform as TP_inverse_fixed_log_transform)

RT_VALUE_BINS = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, RT_UPPER_EXCLUSIVE)
TP_VALUE_BINS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, TP_UPPER_EXCLUSIVE)
CLIENT_CALIBRATION_RIDGE = 1.0
OTBC_CLIENT_BLOCK_SIZE = 8

# Public names remain RT-compatible for callers that import these helpers.
CausalDriftScorer = RT_CausalDriftScorer
drift_groups = RT_drift_groups
_ACTIVE_QOS = "rt"


def _qos_spec(qos_type="rt"):
    """Return the frozen data/transform contract for one QoS attribute."""
    if qos_type == "tp":
        return {
            "qos_type": "tp", "qos_name": "TP", "qos_unit": "kbps",
            "num_users": TP_NUM_USERS, "num_services": TP_NUM_SERVICES, "num_times": TP_NUM_TIMES,
            "valid_upper": TP_UPPER_EXCLUSIVE, "log_upper": TP_LOG_UPPER,
            "value_bins": TP_VALUE_BINS, "causal_drift": TP_CausalDriftScorer,
            "drift_groups": TP_drift_groups, "fixed": TP_fixed_log_transform,
            "inverse": TP_inverse_fixed_log_transform, "last_record_wins": True,
        }
    if qos_type != "rt":
        raise ValueError(f"unsupported qos_type: {qos_type}")
    # Read RT dimensions from the module globals so the existing unit tests'
    # small-shape patches remain valid and the default RT path is unchanged.
    return {
        "qos_type": "rt", "qos_name": "RT", "qos_unit": "seconds",
        "num_users": NUM_USERS, "num_services": NUM_SERVICES, "num_times": NUM_TIMES,
        "valid_upper": RT_UPPER_EXCLUSIVE, "log_upper": 20.0,
        "value_bins": RT_VALUE_BINS, "causal_drift": RT_CausalDriftScorer,
        "drift_groups": RT_drift_groups, "fixed": RT_fixed_log_transform,
        "inverse": RT_inverse_fixed_log_transform, "last_record_wins": False,
    }


def _activate_qos(qos_type):
    """Temporarily bind legacy module globals to the selected QoS contract."""
    global _ACTIVE_QOS, NUM_USERS, NUM_SERVICES, NUM_TIMES, RT_UPPER_EXCLUSIVE
    global RT_VALUE_BINS, CausalDriftScorer, drift_groups
    saved = (_ACTIVE_QOS, NUM_USERS, NUM_SERVICES, NUM_TIMES, RT_UPPER_EXCLUSIVE,
             RT_VALUE_BINS, CausalDriftScorer, drift_groups)
    spec = _qos_spec(qos_type)
    _ACTIVE_QOS = qos_type
    NUM_USERS, NUM_SERVICES, NUM_TIMES = spec["num_users"], spec["num_services"], spec["num_times"]
    RT_UPPER_EXCLUSIVE = spec["valid_upper"]
    RT_VALUE_BINS = spec["value_bins"]
    CausalDriftScorer = spec["causal_drift"]
    drift_groups = spec["drift_groups"]
    return saved


def _restore_qos(saved):
    global _ACTIVE_QOS, NUM_USERS, NUM_SERVICES, NUM_TIMES, RT_UPPER_EXCLUSIVE
    global RT_VALUE_BINS, CausalDriftScorer, drift_groups
    (_ACTIVE_QOS, NUM_USERS, NUM_SERVICES, NUM_TIMES, RT_UPPER_EXCLUSIVE,
     RT_VALUE_BINS, CausalDriftScorer, drift_groups) = saved


def fixed_log_transform(values, upper_bound=None):
    spec = _qos_spec(_ACTIVE_QOS)
    return spec["fixed"](values, spec["log_upper"] if upper_bound is None else upper_bound)


def inverse_fixed_log_transform(values, upper_bound=None):
    spec = _qos_spec(_ACTIVE_QOS)
    return spec["inverse"](values, spec["log_upper"] if upper_bound is None else upper_bound)


def sigmoid(value):
    return 1 / (1 + math.exp(-max(min(value, 30.0), -30.0)))


def load_masks(path, qos_type="rt"):
    spec = _qos_spec(qos_type)
    with np.load(path) as data:
        shape = tuple(data["shape"])
        size = int(np.prod(shape))
        available = np.unpackbits(data["available_bits"])[:size].reshape(shape).astype(bool)
        support = np.unpackbits(data["support_bits"])[:size].reshape(shape).astype(bool)
    assert shape == (spec["num_times"], spec["num_users"], spec["num_services"])
    assert not np.any(support & ~available)
    return available, support


def load_values(path, qos_type="rt"):
    spec = _qos_spec(qos_type)
    values = np.full((spec["num_times"], spec["num_users"], spec["num_services"]), np.nan, dtype=np.float32)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for line in file:
            digest.update(line)
            user, service, time_slice, value = line.split()
            user, service, time_slice = int(user), int(service), int(time_slice)
            value = float(value)
            valid = np.isfinite(value) and 0 < value < spec["valid_upper"]
            if spec["last_record_wins"]:
                # Unlike the historical RT loader, TP explicitly clears an
                # earlier valid duplicate when its last row is invalid.
                values[time_slice, user, service] = value if valid else np.nan
            elif valid:
                values[time_slice, user, service] = value
    return values, digest.hexdigest()


def evaluate(real, predicted):
    error = np.abs(predicted - real)
    relative = np.sort(error / real)
    return {
        "mae": float(error.mean()), "rmse": float(np.sqrt((error**2).mean())),
        "mre": float(np.median(relative)), "npre": float(relative[int(.9 * len(relative))]),
    }


def causal_read_only_audit(args, values, available, support, state):
    """Validate query/future prediction calls do not mutate model state.

    ``state`` is the tuple of mutable prediction state arrays created by the
    active runner.  This helper is deliberately independent of target values:
    it snapshots the arrays, evaluates bounded current/future query rows with
    the supplied read-only callback, and compares the snapshots byte-for-byte.
    """
    if not getattr(args, "audit_read_only", False):
        return {"query_read_only": None, "future_read_only": None, "checked": False}
    callback = getattr(args, "read_only_predict", None)
    if callback is None:
        return {"query_read_only": True, "future_read_only": True, "checked": False, "declaration": "runner query path has no update calls"}
    snapshots = [np.array(item, copy=True) for item in state if isinstance(item, np.ndarray)]
    query_ok = bool(callback(0, values, available, support, future=False))
    future_ok = bool(callback(min(1, available.shape[0] - 1), values, available, support, future=True))
    unchanged = all(np.array_equal(before, after) for before, after in zip(snapshots, state) if isinstance(after, np.ndarray))
    return {"query_read_only": query_ok and unchanged, "future_read_only": future_ok and unchanged, "checked": True}


def _array_state_snapshot(items):
    """Return immutable copies of mutable ndarray model state."""
    return tuple(np.array(item, copy=True) for item in items if isinstance(item, np.ndarray))


def _snapshot_unchanged(before, items) -> bool:
    after = tuple(item for item in items if isinstance(item, np.ndarray))
    return len(before) == len(after) and all(np.array_equal(left, right) for left, right in zip(before, after))


def _read_only_future_probe(time_slice, available, support, users, services, user_biases,
                            service_biases, client_services=None, client_service_biases=None):
    """Read a bounded future query probe without consuming labels or updating state.

    ``client_services``/``client_service_biases`` are supplied by the
    Local-C-OMF path so the audit exercises each client's private service
    state.  The shared arrays remain the default for the existing FedAvg/STCA
    and OTBC paths.
    """
    if time_slice >= available.shape[0]:
        return 0
    users_idx, services_idx = np.nonzero(available[time_slice] & ~support[time_slice])
    if len(users_idx) > 16:
        users_idx, services_idx = users_idx[:16], services_idx[:16]
    if not len(users_idx):
        return 0
    if client_services is None:
        logits = ((users[users_idx] * services[services_idx]).sum(axis=1) +
                  user_biases[users_idx] + service_biases[services_idx])
    else:
        logits = np.asarray([
            users[user] @ client_services[user, service] + user_biases[user] +
            client_service_biases[user, service]
            for user, service in zip(users_idx, services_idx)
        ])
    normalized = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    if not np.all(np.isfinite(normalized)):
        raise FloatingPointError("non-finite future read-only probe")
    return int(len(normalized))


def common_initialization_digest(users, services, user_biases, service_biases):
    """Canonical float32 digest of the AMF common initialization tables."""
    digest = hashlib.sha256()
    for name, value in (
        ("user_embedding", users),
        ("service_embedding", services),
        ("user_bias", user_biases),
        ("service_bias", service_biases),
    ):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(value, dtype=np.float32).tobytes())
    return digest.hexdigest()


def normalized_weight_entropy(weights):
    return 0.0 if len(weights) == 1 else float(-np.sum(weights * np.log(weights + 1e-15)) / math.log(len(weights)))


def normalized_information(weights):
    information = float(np.clip(1.0 - normalized_weight_entropy(weights), 0.0, 1.0))
    return 0.0 if information < 1e-12 else information


def bidirectional_evidence_controls(history_gate, drift_intensity, proposal_uncertainty):
    """Split service change into coherent reaction and uncertainty-risk suppression."""
    values = np.asarray((history_gate, drift_intensity, proposal_uncertainty), dtype=float)
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("bidirectional evidence controls must lie in [0, 1]")
    reactive_transfer = float(drift_intensity * (1.0 - proposal_uncertainty))
    risk_suppression = float(drift_intensity * proposal_uncertainty)
    effective_gate = float((1.0 - drift_intensity) * history_gate + reactive_transfer)
    return effective_gate, reactive_transfer, risk_suppression


def history_evidence_expert(deltas, current_reliability, history_delta, age):
    """Build one HEGSA expert from server-side history and current proposals."""
    deltas = np.asarray(deltas, dtype=float)
    current_reliability = np.asarray(current_reliability, dtype=float)
    if deltas.ndim != 2 or current_reliability.shape != (len(deltas),):
        raise ValueError("expert deltas and current reliability must have matching rows")
    if not np.all(np.isfinite(deltas)) or not np.all(np.isfinite(current_reliability)):
        raise ValueError("expert inputs must be finite")
    temporal_reliability = np.ones(len(deltas))
    history_cosine = np.zeros(len(deltas))
    valid_history = np.zeros(len(deltas), dtype=bool)
    history_used = False
    if history_delta is not None and age is not None and age >= 1:
        history_delta = np.asarray(history_delta, dtype=float)
        if history_delta.shape != (deltas.shape[1],) or not np.all(np.isfinite(history_delta)):
            raise ValueError("expert history must be a finite matching update vector")
        delta_norm = np.linalg.norm(deltas, axis=1)
        history_norm = np.linalg.norm(history_delta)
        valid_history = (delta_norm > 1e-12) & (history_norm > 1e-12)
        if np.any(valid_history):
            history_cosine[valid_history] = np.clip(
                (deltas[valid_history] @ history_delta) /
                (delta_norm[valid_history] * history_norm), -1.0, 1.0)
            temporal_reliability[valid_history] = np.exp(
                (history_cosine[valid_history] - 1.0) / float(age))
            history_used = True
    temporal_weights = temporal_reliability / temporal_reliability.sum()
    consensus_gate = normalized_information(temporal_weights)
    scores = temporal_reliability * current_reliability ** consensus_gate
    weights = scores / scores.sum() if scores.sum() > 1e-15 else np.full(len(deltas), 1.0 / len(deltas))
    return weights, {
        "temporal_reliability": temporal_reliability,
        "history_cosine": history_cosine,
        "valid_history": valid_history,
        "history_used": history_used,
        "consensus_gate": consensus_gate,
    }


def expert_support_risk_and_dispersion(weights, deltas, candidate_errors):
    """Return support-derived error-state risk and weighted proposal dispersion."""
    weights = np.asarray(weights, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    candidate_errors = np.asarray(candidate_errors, dtype=float)
    if deltas.ndim != 2 or weights.shape != (len(deltas),) or candidate_errors.shape != (len(deltas),):
        raise ValueError("expert weights, deltas, and errors must have matching rows")
    if (not np.all(np.isfinite(weights)) or not np.all(np.isfinite(deltas)) or
            not np.all(np.isfinite(candidate_errors)) or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("expert risk inputs must be finite with nonnegative positive-mass weights")
    weights = weights / weights.sum()
    mean_delta = weights @ deltas
    dispersion = float(weights @ np.sum((deltas - mean_delta) ** 2, axis=1))
    return float(weights @ candidate_errors), dispersion


def pareto_prefers_reactive(stable_risk, stable_dispersion, reactive_risk, reactive_dispersion):
    """Select the reactive expert only under threshold-free Pareto dominance."""
    values = np.asarray((stable_risk, stable_dispersion, reactive_risk, reactive_dispersion), dtype=float)
    if not np.all(np.isfinite(values)) or stable_dispersion < 0 or reactive_dispersion < 0:
        raise ValueError("expert risks and dispersions must be finite with nonnegative dispersion")
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(values))))
    no_worse = (reactive_risk <= stable_risk + tolerance and
                reactive_dispersion <= stable_dispersion + tolerance)
    strictly_better = (reactive_risk < stable_risk - tolerance or
                       reactive_dispersion < stable_dispersion - tolerance)
    return bool(no_worse and strictly_better)


def causal_midranks(values):
    """Deterministic mid-ranks in (0, 1), computed from current support summaries only."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    _, starts, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    sorted_ranks = np.empty(len(values), dtype=float)
    for start, count in zip(starts, counts):
        sorted_ranks[start:start + count] = (start + 0.5 * count) / len(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def persistent_drift_proxy(drift_scores, previous_ranks):
    """Return support-only persistent drift and update the previous-rank state."""
    drift_scores = np.asarray(drift_scores, dtype=float)
    if previous_ranks.shape != drift_scores.shape:
        raise ValueError("previous drift ranks must match client scores")
    current = np.full(len(drift_scores), np.nan)
    eligible = np.isfinite(drift_scores)
    if eligible.sum() >= 2 and np.ptp(drift_scores[eligible]) > 0:
        current[eligible] = causal_midranks(drift_scores[eligible])
    eligible = np.isfinite(current)
    persistent = np.zeros(len(drift_scores))
    has_history = eligible & np.isfinite(previous_ranks)
    persistent[has_history] = np.minimum(current[has_history], previous_ranks[has_history])
    previous_ranks[eligible] = current[eligible]
    return persistent, current


def fit_client_fast_calibrator(support_targets, support_predictions, previous_parameters,
                               ridge=CLIENT_CALIBRATION_RIDGE):
    """Fit one private cross-service affine calibrator from current support only."""
    targets = np.asarray(support_targets, dtype=float)
    predictions = np.asarray(support_predictions, dtype=float)
    previous = np.asarray(previous_parameters, dtype=float)
    if targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("support targets and predictions must be matching one-dimensional arrays")
    if previous.shape != (2,) or not np.all(np.isfinite(previous)):
        raise ValueError("previous calibration parameters must contain finite gain and offset")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("calibration ridge must be positive and finite")
    if not len(targets):
        return previous.copy()
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(predictions)):
        raise ValueError("calibration support must be finite")
    design = np.column_stack((predictions, np.ones(len(predictions))))
    system = design.T @ design + ridge * np.eye(2)
    right_hand_side = design.T @ targets + ridge * previous
    parameters = np.linalg.solve(system, right_hand_side)
    if not np.all(np.isfinite(parameters)):
        raise FloatingPointError("client calibration produced non-finite parameters")
    return parameters


def apply_client_fast_calibrator(predictions, parameters):
    """Apply a private client-wide affine state without reading labels."""
    predictions = np.asarray(predictions, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (2,) or not np.all(np.isfinite(parameters)):
        raise ValueError("calibration parameters must contain finite gain and offset")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("base predictions must be finite")
    gain, offset = parameters
    return np.clip(gain * predictions + offset, 0.0, np.nextafter(1.0, 0.0))


def aggregate_service_proposals(reference_vector, reference_bias, reference_error, candidate_vectors,
                                candidate_biases, candidate_errors, mode="fedavg", previous_delta=None, age=None,
                                stca_component="full", detailed_diagnostics=True, candidate_persistent_drift=None,
                                reactive_delta=None, reactive_age=None):
    """Aggregate one service without raw QoS or query labels entering the server API."""
    candidate_vectors = np.asarray(candidate_vectors, dtype=float)
    candidate_biases = np.asarray(candidate_biases, dtype=float)
    candidate_errors = np.asarray(candidate_errors, dtype=float)
    if candidate_vectors.ndim != 2 or not len(candidate_vectors):
        raise ValueError("candidate_vectors must contain at least one two-dimensional proposal")
    if candidate_biases.shape != (len(candidate_vectors),) or candidate_errors.shape != (len(candidate_vectors),):
        raise ValueError("candidate biases/errors must match candidate vectors")
    if mode not in ("fedavg", "stca"):
        raise ValueError(f"unknown aggregation mode: {mode}")
    if stca_component not in ("full", "consensus", "history", "evidence", "adaptive_evidence",
                              "bidirectional_evidence", "selective_dual_memory", "persistent"):
        raise ValueError(f"unknown STCA component: {stca_component}")

    deltas = np.column_stack((candidate_vectors - np.asarray(reference_vector),
                              candidate_biases - float(reference_bias),
                              candidate_errors - float(reference_error)))
    weights = np.full(len(deltas), 1.0 / len(deltas))
    center = np.median(deltas, axis=0)
    squared_distance = np.sum((deltas - center) ** 2, axis=1)
    positive = squared_distance[squared_distance > 1e-15]
    current_reliability = (np.ones(len(deltas)) if not len(positive) else
                           np.exp(-squared_distance / (np.median(positive) + 1e-12)))
    current_weights = current_reliability / current_reliability.sum() if detailed_diagnostics else None
    delta_norm = np.linalg.norm(deltas, axis=1)
    positive_norm = delta_norm[delta_norm > 1e-12]
    delta_scale = float(np.median(positive_norm)) if len(positive_norm) else 0.0
    relative_dispersion = float(np.sqrt(np.mean(squared_distance)) / (delta_scale + 1e-12))
    proposal_uncertainty = float(relative_dispersion / (1.0 + relative_dispersion))
    temporal_reliability = np.ones(len(deltas))
    history_cosine = np.zeros(len(deltas))
    valid_history = np.zeros(len(deltas), dtype=bool)
    center_history_cosine = 0.0
    service_drift_intensity = 0.0
    history_information_gate = 0.0
    effective_consensus_gate = 0.0
    reactive_transfer = 0.0
    risk_suppression = 0.0
    history_used = False
    service_risk = 0.0
    reactive_selected = False
    reactive_history_used = False
    reactive_consensus_gate = 0.0
    stable_support_risk = reactive_support_risk = 0.0
    stable_proposal_dispersion = reactive_proposal_dispersion = 0.0
    if mode == "stca" and len(deltas) > 1:
        if stca_component != "consensus" and previous_delta is not None and age is not None and age >= 1:
            previous_delta = np.asarray(previous_delta, dtype=float)
            previous_norm = np.linalg.norm(previous_delta)
            valid_history = (delta_norm > 1e-12) & (previous_norm > 1e-12)
            if np.any(valid_history):
                history_cosine[valid_history] = np.clip(
                    (deltas[valid_history] @ previous_delta) / (delta_norm[valid_history] * previous_norm), -1.0, 1.0)
                temporal_reliability[valid_history] = np.exp((history_cosine[valid_history] - 1.0) / float(age))
                center_norm = np.linalg.norm(center)
                if center_norm > 1e-12:
                    center_history_cosine = float(np.clip(
                        center @ previous_delta / (center_norm * previous_norm), -1.0, 1.0))
                change_scale = center_norm + previous_norm
                if change_scale > 1e-12:
                    service_drift_intensity = float(np.clip(
                        np.linalg.norm(center - previous_delta) / change_scale, 0.0, 1.0))
                history_used = True
        if stca_component == "consensus":
            scores = current_reliability
        elif stca_component == "history":
            scores = temporal_reliability
        elif stca_component in ("evidence", "persistent"):
            temporal_weights = temporal_reliability / temporal_reliability.sum()
            consensus_gate = normalized_information(temporal_weights)
            scores = temporal_reliability * current_reliability ** consensus_gate
            history_information_gate = consensus_gate
            effective_consensus_gate = consensus_gate
        elif stca_component == "adaptive_evidence":
            temporal_weights = temporal_reliability / temporal_reliability.sum()
            history_information_gate = normalized_information(temporal_weights)
            effective_consensus_gate = (history_information_gate + service_drift_intensity *
                                        (1.0 - history_information_gate))
            retained_history = temporal_reliability ** (1.0 - service_drift_intensity)
            scores = retained_history * current_reliability ** effective_consensus_gate
        elif stca_component == "bidirectional_evidence":
            temporal_weights = temporal_reliability / temporal_reliability.sum()
            history_information_gate = normalized_information(temporal_weights)
            effective_consensus_gate, reactive_transfer, risk_suppression = bidirectional_evidence_controls(
                history_information_gate, service_drift_intensity, proposal_uncertainty)
            retained_history = temporal_reliability ** (1.0 - reactive_transfer)
            scores = retained_history * current_reliability ** effective_consensus_gate
        elif stca_component == "selective_dual_memory":
            temporal_weights = temporal_reliability / temporal_reliability.sum()
            history_information_gate = normalized_information(temporal_weights)
            stable_scores = temporal_reliability * current_reliability ** history_information_gate
            stable_weights = (stable_scores / stable_scores.sum() if stable_scores.sum() > 1e-15 else
                              np.full(len(deltas), 1.0 / len(deltas)))
            reactive_weights, reactive_diagnostics = history_evidence_expert(
                deltas, current_reliability, reactive_delta, reactive_age)
            reactive_history_used = reactive_diagnostics["history_used"]
            reactive_consensus_gate = reactive_diagnostics["consensus_gate"]
            stable_support_risk, stable_proposal_dispersion = expert_support_risk_and_dispersion(
                stable_weights, deltas, candidate_errors)
            reactive_support_risk, reactive_proposal_dispersion = expert_support_risk_and_dispersion(
                reactive_weights, deltas, candidate_errors)
            reactive_selected = bool(
                reactive_history_used and pareto_prefers_reactive(
                    stable_support_risk, stable_proposal_dispersion,
                    reactive_support_risk, reactive_proposal_dispersion))
            scores = reactive_weights if reactive_selected else stable_weights
        else:
            scores = current_reliability * temporal_reliability
        weights = scores / scores.sum() if scores.sum() > 1e-15 else weights
        if stca_component == "persistent":
            persistent = np.asarray(candidate_persistent_drift, dtype=float)
            if persistent.shape != (len(weights),) or not np.all(np.isfinite(persistent)):
                raise ValueError("candidate persistent drift must match proposals and be finite")
            if np.any((persistent < 0) | (persistent > 1)):
                raise ValueError("candidate persistent drift must lie in [0, 1]")
            service_risk = float(np.mean(persistent))
            weights = (1.0 - service_risk) * weights + service_risk / len(weights)

    vector = weights @ candidate_vectors
    bias = float(weights @ candidate_biases)
    error = float(weights @ candidate_errors)
    aggregate_delta = np.concatenate((vector - np.asarray(reference_vector),
                                      [bias - float(reference_bias), error - float(reference_error)]))
    entropy = normalized_weight_entropy(weights)
    diagnostics = {
        "weights": weights, "normalized_entropy": entropy,
        "effective_clients": float(1.0 / np.sum(weights ** 2)), "history_used": history_used,
    }
    if stca_component == "persistent":
        diagnostics["persistent_service_risk"] = service_risk
    if stca_component == "selective_dual_memory":
        diagnostics.update({
            "reactive_selected": reactive_selected,
            "reactive_history_used": reactive_history_used,
            "stable_support_risk": stable_support_risk,
            "reactive_support_risk": reactive_support_risk,
            "stable_proposal_dispersion": stable_proposal_dispersion,
            "reactive_proposal_dispersion": reactive_proposal_dispersion,
            "stable_consensus_gate": history_information_gate,
            "reactive_consensus_gate": reactive_consensus_gate,
        })
    if detailed_diagnostics:
        temporal_weights = temporal_reliability / temporal_reliability.sum()
        diagnostics.update({
            "relative_dispersion": relative_dispersion,
            "proposal_uncertainty": proposal_uncertainty,
            "current_entropy": normalized_weight_entropy(current_weights),
            "current_effective_clients": float(1.0 / np.sum(current_weights ** 2)),
            "temporal_entropy": normalized_weight_entropy(temporal_weights),
            "temporal_effective_clients": float(1.0 / np.sum(temporal_weights ** 2)),
            "mean_history_cosine": float(np.mean(history_cosine[valid_history])) if np.any(valid_history) else 0.0,
            "center_history_cosine": center_history_cosine,
            "valid_history_fraction": float(np.mean(valid_history)),
            "aggregate_delta_norm": float(np.linalg.norm(aggregate_delta)),
            "consensus_gate": (effective_consensus_gate if stca_component in ("adaptive_evidence", "bidirectional_evidence") else
                               normalized_information(temporal_weights)),
            "history_information_gate": history_information_gate,
            "service_drift_intensity": service_drift_intensity,
            "reactive_transfer": reactive_transfer,
            "risk_suppression": risk_suppression,
        })
    return vector, bias, error, aggregate_delta, diagnostics


def _coqmf_local_update(services, targets, user_vector, user_bias, user_error,
                        vectors, biases, errors, args, rng,
                        reference_vectors, reference_biases, robust_scale=None):
    """Run the unchanged per-support-sample COQMF proposal update."""
    fedprox_mu = getattr(args, "fedprox_mu", 0.0)
    for _ in range(args.local_epochs):
        raw_balance = None
        if getattr(args, "dual_scale_gradient", False):
            initial_logits = vectors @ user_vector + user_bias + biases
            initial_predictions = 1.0 / (1.0 + np.exp(-np.clip(initial_logits, -30.0, 30.0)))
            raw_balance = raw_gradient_balance(
                initial_predictions, targets, _qos_spec(_ACTIVE_QOS)["log_upper"])
        for index in rng.permutation(len(services)):
            target, before = float(targets[index]), user_vector.copy()
            prediction = sigmoid(float(before @ vectors[index] + user_bias + biases[index]))
            relative_error = abs(prediction - target) / target
            user_weight = user_error / (user_error + errors[index])
            service_weight = 1 - user_weight
            user_error = args.beta * user_weight * relative_error + (1 - args.beta * user_weight) * user_error
            errors[index] = args.beta * service_weight * relative_error + (1 - args.beta * service_weight) * errors[index]
            if getattr(args, "target_gated_tail_gradient", False):
                gradient = target_gated_tail_logit_gradient(
                    prediction, target, _qos_spec(_ACTIVE_QOS)["log_upper"])
            elif getattr(args, "prediction_gated_tail_gradient", False):
                gradient = prediction_gated_tail_logit_gradient(
                    prediction, target, _qos_spec(_ACTIVE_QOS)["log_upper"])
            elif getattr(args, "bounded_tail_gradient", False):
                gradient = bounded_tail_logit_gradient(
                    prediction, target, _qos_spec(_ACTIVE_QOS)["log_upper"])
            elif getattr(args, "dual_scale_gradient", False):
                gradient = balanced_logit_gradient(
                    prediction, target, _qos_spec(_ACTIVE_QOS)["log_upper"], raw_balance)
            elif getattr(args, "robust_amf", False):
                if robust_scale is None:
                    raise ValueError("Robust-AMF requires a positive client-private support scale")
                probability_gradient = pseudo_huber_gradient(
                    np.array([prediction - target]),
                    delta=args.robust_delta,
                    scale=robust_scale,
                )[0]
                gradient = probability_gradient * prediction * (1 - prediction)
            else:
                gradient = (prediction - target) * prediction * (1 - prediction)
            user_vector -= args.eta * (user_weight * gradient * vectors[index] + args.l2 * before)
            vectors[index] -= args.eta * (service_weight * gradient * before + args.l2 * vectors[index] +
                                           fedprox_mu * (vectors[index] - reference_vectors[index]))
            user_bias -= args.eta * (user_weight * gradient + args.l2 * user_bias)
            biases[index] -= args.eta * (user_weight * gradient + args.l2 * biases[index] +
                                          fedprox_mu * (biases[index] - reference_biases[index]))
    return user_vector, user_bias, user_error, vectors, biases, errors


def local_update(services, targets, user_vector, user_bias, user_error,
                 service_vectors, service_biases, service_errors, args, rng,
                 reference_vectors=None, reference_biases=None, robust_scale=None,
                 novelty_history=None):
    vectors = service_vectors[services].copy()
    biases = service_biases[services].copy()
    errors = service_errors[services].copy()
    reference_vectors = vectors.copy() if reference_vectors is None else reference_vectors
    reference_biases = biases.copy() if reference_biases is None else reference_biases
    csotu_enabled = getattr(args, "causal_support_output_trust_update", False)
    scspu_enabled = getattr(args, "support_conditioned_stability_plasticity_update", False)
    sns_enabled = getattr(args, "support_novelty_subspace_update", False)
    if sum(bool(enabled) for enabled in (csotu_enabled, scspu_enabled, sns_enabled)) > 1:
        raise ValueError("CSOTU, SCSPU, and SNSU are mutually exclusive")
    fedprox_mu = getattr(args, "fedprox_mu", 0.0)
    if csotu_enabled:
        return support_output_trust_update(
            targets, user_vector, user_bias, user_error, vectors, biases, errors,
            eta=args.eta, l2=args.l2, beta=args.beta,
            local_epochs=args.local_epochs, fedprox_mu=fedprox_mu,
            reference_vectors=reference_vectors, reference_biases=reference_biases,
        )
    if scspu_enabled:
        # The anchor and drift signal are frozen before any COQMF update.  The
        # proposal path below is exactly the original local update, including
        # its support-only seeded permutation and AMF error-state update.
        anchor_user_vector = np.asarray(user_vector, dtype=float).copy()
        anchor_user_bias = float(user_bias)
        anchor_vectors = vectors.copy()
        anchor_biases = biases.copy()
        before_predictions = 1.0 / (1.0 + np.exp(-np.clip(
            anchor_vectors @ anchor_user_vector + anchor_user_bias + anchor_biases,
            -30.0, 30.0)))
        proposal = _coqmf_local_update(
            services, targets, anchor_user_vector.copy(), anchor_user_bias,
            float(user_error), anchor_vectors.copy(), anchor_biases.copy(),
            errors.copy(), args, rng, reference_vectors, reference_biases,
            robust_scale=robust_scale,
        )
        projected = support_conditioned_stability_plasticity_update(
            before_predictions, targets, user_error, errors,
            anchor_user_vector, anchor_user_bias, anchor_vectors, anchor_biases,
            proposal[0], proposal[1], proposal[2], proposal[3], proposal[4],
            proposal[5],
        )
        return projected[:6]
    if sns_enabled:
        # SNSU consumes the unchanged support-only COQMF proposal and rotates
        # only its joint user-vector/user-bias displacement.  Shared service
        # proposals and both AMF error states pass through byte-for-byte.
        anchor_user_vector = np.asarray(user_vector, dtype=float).copy()
        anchor_user_bias = float(user_bias)
        if novelty_history is None:
            novelty_history = np.zeros(anchor_user_vector.size + 1, dtype=float)
        novelty_history = np.asarray(novelty_history, dtype=float)
        proposal = _coqmf_local_update(
            services, targets, anchor_user_vector.copy(), anchor_user_bias,
            float(user_error), vectors.copy(), biases.copy(), errors.copy(),
            args, rng, reference_vectors, reference_biases,
            robust_scale=robust_scale,
        )
        accepted_user, accepted_bias, history_after, _ = support_novelty_subspace_update(
            anchor_user_vector, anchor_user_bias,
            proposal[0], proposal[1], novelty_history,
        )
        if novelty_history.shape != history_after.shape:
            raise ValueError("novelty history shape changed during support update")
        novelty_history[...] = history_after
        return (accepted_user, accepted_bias, proposal[2], proposal[3],
                proposal[4], proposal[5])
    return _coqmf_local_update(
        services, targets, user_vector, user_bias, user_error,
        vectors, biases, errors, args, rng, reference_vectors, reference_biases,
        robust_scale=robust_scale,
    )


def run(args, values, available, support):
    """Run one causal stream under the selected RT/TP contract."""
    saved = _activate_qos(getattr(args, "qos_type", "rt"))
    try:
        return _run_active(args, values, available, support)
    finally:
        _restore_qos(saved)


def _run_active(args, values, available, support):
    rng = np.random.default_rng(args.seed)
    local_mode = args.aggregation == "local"
    users = rng.normal(0, .05, (NUM_USERS, args.dimension))
    services = rng.normal(0, .05, (NUM_SERVICES, args.dimension))
    user_biases, service_biases = np.zeros(NUM_USERS), np.zeros(NUM_SERVICES)
    # Keep the actual matched common-state digest on the run namespace for
    # resolved_config emission; prediction/update behavior is unchanged.
    args.common_state_digest = common_initialization_digest(
        users, services, user_biases, service_biases
    )
    user_errors, service_errors = np.ones(NUM_USERS), np.ones(NUM_SERVICES)
    otbc_enabled = bool(getattr(args, "output_trust_bounded_compression", False))
    # Local mode keeps an independent service-state table for every client.
    # Unlike OTBC, local tables are updated directly by the owning client and
    # never reconstructed from a server downlink.
    client_state_enabled = local_mode or otbc_enabled
    communication_parameter_bytes = int(getattr(args, "communication_parameter_bytes", 8))
    # OTBC keeps the server state exact and gives every client a real decoded
    # service-state cache.  Cache-relative deltas automatically carry the
    # previous reconstruction residual into the next transmission.
    client_services = (np.repeat(services[None, :, :], NUM_USERS, axis=0)
                       if client_state_enabled else None)
    client_service_biases = (np.repeat(service_biases[None, :], NUM_USERS, axis=0)
                             if client_state_enabled else None)
    client_service_errors = (np.repeat(service_errors[None, :], NUM_USERS, axis=0)
                             if client_state_enabled else None)
    support_novelty_enabled = getattr(args, "support_novelty_subspace_update", False)
    novelty_histories = (np.zeros((NUM_USERS, args.dimension + 1), dtype=float)
                         if support_novelty_enabled else None)
    residuals = np.zeros((NUM_USERS, NUM_SERVICES)) if args.private_service_residual else None
    residual_counts = np.zeros((NUM_USERS, NUM_SERVICES)) if args.private_service_residual else None
    client_fast_calibration = getattr(args, "client_fast_calibration", False)
    client_monotone_calibration = getattr(args, "client_monotone_calibration", False)
    client_logit_calibration = getattr(args, "client_logit_calibration", False)
    calibration_parameters = (np.tile(CPLC_IDENTITY, (NUM_USERS, 1))
                              if client_logit_calibration else
                              (np.tile(CPMC_IDENTITY, (NUM_USERS, 1))
                              if client_monotone_calibration else
                              (np.tile(np.array([1.0, 0.0]), (NUM_USERS, 1))
                               if client_fast_calibration else None)))
    dtnf_bridge = getattr(args, "dtnf_bridge", False)
    private_fast = np.zeros((NUM_USERS, args.dimension)) if dtnf_bridge else None
    shared_slow = np.zeros((NUM_SERVICES, 2)) if dtnf_bridge else None
    temporal_gates = np.zeros(2) if dtnf_bridge else None
    causal_time_factor = getattr(args, "causal_time_factor", False)
    cttf_delta = np.zeros(args.dimension) if causal_time_factor else None
    state_space_mf = getattr(args, "state_space_mf", False)
    state_private_level = np.zeros((NUM_USERS, args.dimension)) if state_space_mf else None
    state_private_velocity = np.zeros((NUM_USERS, args.dimension)) if state_space_mf else None
    state_shared_level = np.zeros((NUM_SERVICES, args.dimension)) if state_space_mf else None
    state_shared_velocity = np.zeros((NUM_SERVICES, args.dimension)) if state_space_mf else None
    state_space_gates = np.zeros(STATE_SPACE_GATE_DIMENSION) if state_space_mf else None
    graph_residual = getattr(args, "graph_residual_backbone", False)
    graph_client_residual = np.zeros(NUM_USERS) if graph_residual else None
    graph_client_confidence = np.zeros(NUM_USERS) if graph_residual else None
    graph_service_residual = np.zeros(NUM_SERVICES) if graph_residual else None
    graph_service_confidence = np.zeros(NUM_SERVICES) if graph_residual else None
    graph_gate = 0.0
    robust_amf = getattr(args, "robust_amf", False)
    tail_safe_aggregation = getattr(args, "tail_safe_aggregation", False)
    service_raw_residual_calibration = getattr(args, "service_raw_residual_calibration", False)
    service_residual_reliability_shrinkage = getattr(
        args, "service_residual_reliability_shrinkage", False)
    service_residual_variance_aware_shrinkage = getattr(
        args, "service_residual_variance_aware_shrinkage", False)
    prediction_conditioned_service_calibration = getattr(
        args, "prediction_conditioned_service_calibration", False)
    log_domain_service_calibration = getattr(
        args, "log_domain_service_calibration", False)
    dual_constrained_log_calibration = getattr(
        args, "dual_constrained_log_calibration", False)
    robust_scales = np.full(NUM_USERS, np.nan) if robust_amf else None
    dynamic_online_routing = getattr(args, "dynamic_online_routing", False)
    dor_states = [initial_dor_state() for _ in range(NUM_USERS)] if dynamic_online_routing else None
    conformal_mode = getattr(args, "conformal_mode", "off")
    global_conformal_errors = []
    private_conformal_errors = [[] for _ in range(NUM_USERS)]
    conformal_radii = np.zeros(NUM_USERS)
    previous_service_delta = np.zeros((NUM_SERVICES, args.dimension + 2))
    previous_service_update = np.full(NUM_SERVICES, -1, dtype=int)
    selective_dual_memory = args.stca_component == "selective_dual_memory"
    reactive_service_delta = (np.zeros((NUM_SERVICES, args.dimension + 2))
                              if selective_dual_memory else None)
    reactive_service_update = (np.full(NUM_SERVICES, -1, dtype=int)
                               if selective_dual_memory else None)
    previous_drift_rank = np.full(NUM_USERS, np.nan)
    audit_read = bool(getattr(args, "audit_read_only", False))
    audit_future = bool(getattr(args, "audit_future", False))
    read_only_audit = {"query_read_only": audit_read, "future_read_only": audit_future,
                       "checked": bool(audit_read or audit_future),
                       "protocol": "snapshot -> read-only query/future probe -> exact state comparison"}
    rows, client_rows, group_rows, value_bin_rows = [], [], [], []
    service_time_rows, service_aggregation_rows = [], []
    drift_scorer = CausalDriftScorer(NUM_USERS)
    for time_slice in range(args.max_time_slices or NUM_TIMES):
        # Query/future probes are taken only from the read path.  Snapshot the
        # mutable model arrays before probing and require exact equality after.
        if audit_read or audit_future:
            audit_state = (users, services, user_biases, service_biases, user_errors, service_errors,
                           residuals, residual_counts, private_fast, shared_slow, temporal_gates,
                           cttf_delta, state_private_level, state_private_velocity, state_shared_level,
                           state_shared_velocity, state_space_gates, graph_client_residual,
                           graph_client_confidence, graph_service_residual, graph_service_confidence,
                           robust_scales, conformal_radii, calibration_parameters,
                           novelty_histories, client_services, client_service_biases,
                           client_service_errors)
            if selective_dual_memory:
                audit_state = audit_state + (reactive_service_delta, reactive_service_update)
            audit_before = _array_state_snapshot(audit_state)
            if audit_read:
                _read_only_future_probe(
                    time_slice, available, support, users, services, user_biases, service_biases,
                    client_services if client_state_enabled else None,
                    client_service_biases if client_state_enabled else None,
                )
            if not _snapshot_unchanged(audit_before, audit_state):
                raise AssertionError(f"RT/TP query probe mutated state at slice={time_slice}")
            if audit_future and time_slice + 1 < available.shape[0]:
                _read_only_future_probe(
                    time_slice + 1, available, support, users, services, user_biases, service_biases,
                    client_services if client_state_enabled else None,
                    client_service_biases if client_state_enabled else None,
                )
            if not _snapshot_unchanged(audit_before, audit_state):
                raise AssertionError(f"RT/TP future probe mutated state at slice={time_slice}")
        # Validation-only snapshots and read probes are deliberately outside
        # the algorithm timer.  They can be much larger for methods that keep
        # decoded client caches and are not part of online model updating.
        started = time.perf_counter()
        otbc_slack = np.zeros(NUM_USERS, dtype=float)
        otbc_user_norm = np.zeros(NUM_USERS, dtype=float)
        if residual_counts is not None:
            residual_counts *= args.residual_decay
        vector_sum = np.zeros_like(services)
        bias_sum, error_sum = np.zeros_like(service_biases), np.zeros_like(service_errors)
        proposals = [[] for _ in range(NUM_SERVICES)] if args.aggregation == "stca" else None
        proposal_users = ([[] for _ in range(NUM_SERVICES)]
                          if proposals is not None and tail_safe_aggregation else None)
        counts = np.zeros(NUM_SERVICES, dtype=int)
        slow_target_sums = np.zeros(NUM_SERVICES) if dtnf_bridge else None
        state_service_counts = np.zeros(NUM_SERVICES) if state_space_mf else None
        state_innovation_sums = (np.zeros((NUM_SERVICES, args.dimension))
                                 if state_space_mf else None)
        state_touched_client_service_rows = 0
        graph_service_counts = np.zeros(NUM_SERVICES) if graph_residual else None
        graph_service_residual_sums = np.zeros(NUM_SERVICES) if graph_residual else None
        graph_touched_client_service_rows = 0
        participants, uplink = 0, 0
        participant_users = [] if otbc_enabled else None
        scores = np.full(NUM_USERS, np.nan)
        for user in range(NUM_USERS):
            indices = np.flatnonzero(support[time_slice, user])
            if not len(indices):
                continue
            participants += 1
            if participant_users is not None:
                participant_users.append(user)
            targets = fixed_log_transform(values[time_slice, user, indices])
            scores[user] = drift_scorer.update_and_score(user, targets)
            local_services = client_services[user] if client_state_enabled else services
            local_service_biases = (client_service_biases[user]
                                    if client_state_enabled else service_biases)
            local_service_errors = (client_service_errors[user]
                                    if client_state_enabled else service_errors)
            pre_update_user = users[user].copy()
            pre_update_user_bias = float(user_biases[user])
            pre_update_service_vectors = local_services[indices].copy()
            pre_update_service_biases = local_service_biases[indices].copy()
            pre_update_logits = (users[user] @ local_services[indices].T + user_biases[user]
                                 + local_service_biases[indices])
            pre_update_predictions = np.array([sigmoid(value) for value in pre_update_logits])
            if robust_amf:
                previous_scale = (float(robust_scales[user])
                                  if np.isfinite(robust_scales[user]) else None)
                robust_scales[user] = adaptive_support_scale(
                    pre_update_predictions - targets,
                    previous_scale,
                    decay=args.robust_scale_decay,
                    floor=args.robust_scale_floor,
                )
            if dtnf_bridge:
                fast_proposal = update_private_fast_state(
                    private_fast[user], targets, services[indices], pre_update_predictions)
                slow_target_sums[indices] += targets
            if state_space_mf:
                support_residuals = targets - pre_update_predictions
                state_private_level[user], state_private_velocity[user] = update_private_state(
                    state_private_level[user], state_private_velocity[user],
                    services[indices], support_residuals)
                client_counts, client_innovation = service_state_statistics(
                    indices,
                    np.repeat(users[user][None, :], len(indices), axis=0),
                    support_residuals,
                    NUM_SERVICES,
                )
                state_service_counts += client_counts
                state_innovation_sums += client_innovation
                state_touched_client_service_rows += len(indices)
            if graph_residual:
                graph_client_residual, graph_client_confidence = update_client_private_graph_state(
                    graph_client_residual, graph_client_confidence,
                    np.full(len(indices), user, dtype=int), targets,
                    pre_update_predictions, NUM_USERS)
                client_graph_counts, client_graph_sums = service_graph_support_statistics(
                    indices, targets, pre_update_predictions, NUM_SERVICES)
                graph_service_counts += client_graph_counts
                graph_service_residual_sums += client_graph_sums
                graph_touched_client_service_rows += len(indices)
            if residuals is not None:
                base_logits = users[user] @ services[indices].T + user_biases[user] + service_biases[indices]
                residuals[user, indices] = ((1 - args.residual_eta * args.residual_l2) * residuals[user, indices] +
                                            args.residual_eta * (targets - np.array([sigmoid(value) for value in base_logits])))
                residual_counts[user, indices] += 1
            result = local_update(
                indices, targets, users[user].copy(), user_biases[user], user_errors[user],
                local_services, local_service_biases, local_service_errors, args,
                np.random.default_rng(np.random.SeedSequence([args.seed, time_slice, user])),
                robust_scale=(float(robust_scales[user]) if robust_amf else None),
                novelty_history=(novelty_histories[user] if novelty_histories is not None else None),
            )
            users[user], user_biases[user], user_errors[user], vectors, biases, errors = result
            if local_mode:
                # Local C-OMF commits the support update to this client's
                # private service table.  No proposal is exposed to a server,
                # and no other client's table can observe this update.
                client_services[user, indices] = vectors
                client_service_biases[user, indices] = biases
                client_service_errors[user, indices] = errors
            elif otbc_enabled:
                trust_budget = output_trust_budget(pre_update_predictions, targets)
                used_budget = support_prediction_rms_change(
                    pre_update_user, pre_update_user_bias,
                    pre_update_service_vectors, pre_update_service_biases,
                    users[user], user_biases[user], vectors, biases)
                otbc_slack[user] = max(float(trust_budget - used_budget), 0.0)
                otbc_user_norm[user] = float(np.linalg.norm(users[user]))
            if dtnf_bridge:
                private_fast[user] = fast_proposal
            vector_sum[indices] += vectors
            bias_sum[indices] += biases
            error_sum[indices] += errors
            counts[indices] += 1
            if proposals is not None:
                for position, service in enumerate(indices):
                    proposal = (vectors[position], biases[position], errors[position])
                    proposals[service].append(proposal + ((user,) if args.stca_component == "persistent" else ()))
                    if proposal_users is not None:
                        proposal_users[service].append(user)
            if not local_mode:
                uplink += len(indices) * (args.dimension + 2) * communication_parameter_bytes
            if otbc_enabled and not local_mode:
                # One FP32 slack and one FP32 private-user norm certificate.
                uplink += 2 * 4
            if args.stca_component == "persistent" and not local_mode:
                uplink += 8
        changed = counts > 0
        persistent_drift = (persistent_drift_proxy(scores, previous_drift_rank)[0]
                            if args.stca_component == "persistent" else None)
        aggregation_entropy, effective_clients, persistent_service_risk, history_services = [], [], [], 0
        adaptive_drift_intensity, adaptive_consensus_gate = [], []
        proposal_uncertainty_rows, reactive_transfer_rows, risk_suppression_rows = [], [], []
        sdm_reactive_selected = sdm_reactive_eligible = 0
        sdm_stable_risk, sdm_reactive_risk = [], []
        sdm_stable_dispersion, sdm_reactive_dispersion = [], []
        tail_safe_fallback_services = 0
        tail_safe_validated_services = 0
        tail_safe_fedavg_support_sse = 0.0
        tail_safe_stca_support_sse = 0.0
        if local_mode:
            # The global arrays are intentionally left at their initialization
            # values.  All support updates were committed to client_services
            # above, so this branch is the complete local-only aggregator.
            pass
        elif args.aggregation == "fedavg":
            services[changed] = vector_sum[changed] / counts[changed, None]
            service_biases[changed] = bias_sum[changed] / counts[changed]
            service_errors[changed] = error_sum[changed] / counts[changed]
        else:
            for service in np.flatnonzero(changed):
                reference_vector = services[service].copy()
                reference_bias = float(service_biases[service])
                reference_error = float(service_errors[service])
                candidate_vectors = np.stack([proposal[0] for proposal in proposals[service]])
                candidate_biases = np.array([proposal[1] for proposal in proposals[service]])
                candidate_errors = np.array([proposal[2] for proposal in proposals[service]])
                age = None if previous_service_update[service] < 0 else time_slice - previous_service_update[service]
                reactive_age = (None if not selective_dual_memory or reactive_service_update[service] < 0 else
                                time_slice - reactive_service_update[service])
                result = aggregate_service_proposals(
                    reference_vector, reference_bias, reference_error,
                    candidate_vectors, candidate_biases, candidate_errors,
                    mode="stca", previous_delta=previous_service_delta[service], age=age,
                    stca_component=args.stca_component,
                    detailed_diagnostics=args.diagnostic_service_aggregation,
                    candidate_persistent_drift=(np.array([persistent_drift[proposal[3]] for proposal in proposals[service]])
                                                if args.stca_component == "persistent" else None),
                    reactive_delta=(reactive_service_delta[service] if selective_dual_memory else None),
                    reactive_age=reactive_age,
                )
                stca_vector, stca_bias, stca_error, delta, diagnostics = result
                if tail_safe_aggregation:
                    fedavg_vector = candidate_vectors.mean(axis=0)
                    fedavg_bias = float(candidate_biases.mean())
                    fedavg_error = float(candidate_errors.mean())
                    support_users = np.asarray(proposal_users[service], dtype=int)
                    selected = select_support_raw_scale_state(
                        users[support_users], user_biases[support_users],
                        values[time_slice, support_users, service],
                        fedavg_vector, fedavg_bias, fedavg_error,
                        stca_vector, stca_bias, stca_error, _qos_spec(_ACTIVE_QOS)["log_upper"],
                    )
                    selected_vector, selected_bias, selected_error, tail_diagnostics = selected
                    services[service] = selected_vector
                    service_biases[service] = selected_bias
                    service_errors[service] = selected_error
                    delta = np.concatenate((selected_vector - reference_vector,
                                            [selected_bias - reference_bias,
                                             selected_error - reference_error]))
                    tail_safe_validated_services += 1
                    tail_safe_fallback_services += int(tail_diagnostics["selected_fedavg"])
                    tail_safe_fedavg_support_sse += tail_diagnostics["fedavg_support_raw_sse"]
                    tail_safe_stca_support_sse += tail_diagnostics["stca_support_raw_sse"]
                else:
                    services[service], service_biases[service], service_errors[service] = (
                        stca_vector, stca_bias, stca_error)
                previous_service_delta[service] = delta
                previous_service_update[service] = time_slice
                if selective_dual_memory:
                    reactive_service_delta[service] = np.median(np.column_stack((
                        candidate_vectors - reference_vector,
                        candidate_biases - reference_bias,
                        candidate_errors - reference_error,
                    )), axis=0)
                    reactive_service_update[service] = time_slice
                aggregation_entropy.append(diagnostics["normalized_entropy"])
                effective_clients.append(diagnostics["effective_clients"])
                if "persistent_service_risk" in diagnostics:
                    persistent_service_risk.append(diagnostics["persistent_service_risk"])
                history_services += int(diagnostics["history_used"])
                if args.stca_component in ("adaptive_evidence", "bidirectional_evidence") and args.diagnostic_service_aggregation:
                    adaptive_drift_intensity.append(diagnostics["service_drift_intensity"])
                    adaptive_consensus_gate.append(diagnostics["consensus_gate"])
                    proposal_uncertainty_rows.append(diagnostics["proposal_uncertainty"])
                    reactive_transfer_rows.append(diagnostics["reactive_transfer"])
                    risk_suppression_rows.append(diagnostics["risk_suppression"])
                if selective_dual_memory:
                    sdm_reactive_selected += int(diagnostics["reactive_selected"])
                    sdm_reactive_eligible += int(diagnostics["reactive_history_used"])
                    sdm_stable_risk.append(diagnostics["stable_support_risk"])
                    sdm_reactive_risk.append(diagnostics["reactive_support_risk"])
                    sdm_stable_dispersion.append(diagnostics["stable_proposal_dispersion"])
                    sdm_reactive_dispersion.append(diagnostics["reactive_proposal_dispersion"])
                if args.diagnostic_service_aggregation:
                    aggregation_row = {
                        "time_slice": time_slice, "service": int(service), "support_clients": int(counts[service]),
                        "history_age": 0 if age is None else int(age), **{key: diagnostics[key] for key in (
                            "relative_dispersion", "proposal_uncertainty", "current_entropy", "current_effective_clients",
                            "temporal_entropy", "temporal_effective_clients", "mean_history_cosine",
                            "center_history_cosine", "valid_history_fraction", "normalized_entropy",
                            "effective_clients", "aggregate_delta_norm", "history_information_gate",
                            "service_drift_intensity", "reactive_transfer", "risk_suppression", "consensus_gate",
                        )},
                    }
                    if selective_dual_memory:
                        aggregation_row.update({
                            "reactive_history_age": 0 if reactive_age is None else int(reactive_age),
                            **{key: diagnostics[key] for key in (
                                "reactive_selected", "reactive_history_used",
                                "stable_support_risk", "reactive_support_risk",
                                "stable_proposal_dispersion", "reactive_proposal_dispersion",
                                "stable_consensus_gate", "reactive_consensus_gate",
                            )},
                        })
                    service_aggregation_rows.append(aggregation_row)
        # Freeze the ESSD selection before any post-aggregation support read.
        # Both control and OTBC use the same FP32 communication contract.
        calibration_extra_row_bytes = 0
        calibration_extra_per_participant_bytes = 0
        if service_raw_residual_calibration:
            if log_domain_service_calibration or prediction_conditioned_service_calibration:
                calibration_extra_row_bytes = 4
                calibration_extra_per_participant_bytes = (4 if prediction_conditioned_service_calibration else 0)
            else:
                calibration_extra_row_bytes = 8
        if local_mode:
            # Keep the per-time schema stable while making the absence of any
            # server-to-client service payload explicit in every field.
            event_sparse_plan = {
                "mode": "none", "bitmap_bytes": 0, "transmitted_rows": 0,
                "full_bytes": 0, "sparse_bytes": 0, "chosen_bytes": 0,
                "saved_bytes": 0, "chosen_extra_bytes": 0,
                "extra_row_bytes": 0, "extra_per_participant_bytes": 0,
            }
        else:
            event_sparse_plan = plan_service_downlink(
                participants, NUM_SERVICES, args.dimension, int(changed.sum()),
                value_bytes=communication_parameter_bytes,
                extra_row_bytes=calibration_extra_row_bytes,
                extra_per_participant_bytes=calibration_extra_per_participant_bytes,
                enabled=bool(getattr(args, "event_sparse_service_downlink", False)),
            )
        otbc_int16_rows = 0
        otbc_fp32_rows = 0
        otbc_model_downlink = 0
        otbc_quantization_bounds = np.empty((0, 0), dtype=float)
        otbc_selected_bounds = np.empty((0, 0), dtype=float)
        otbc_trust_violations = 0
        otbc_cache_max_abs_error = 0.0
        if otbc_enabled:
            transmitted_mask = (changed if event_sparse_plan["mode"] == "sparse"
                                else np.ones(NUM_SERVICES, dtype=bool))
            transmitted_indices = np.flatnonzero(transmitted_mask)
            bitmap_per_client = int(event_sparse_plan["bitmap_bytes"])
            participant_users = np.asarray(participant_users, dtype=int)
            target_vectors = services[transmitted_indices]
            target_biases = service_biases[transmitted_indices]
            target_errors = np.asarray(
                service_errors[transmitted_indices], dtype=np.float32).astype(float)
            # Preserve the established client-major, service-minor diagnostic
            # order without materializing hundreds of thousands of Python
            # floats via ``tolist``.  ``np.mean`` below sees the same C-order
            # float64 values as the previous flattened list.
            otbc_quantization_bounds = np.empty(
                (len(participant_users), len(transmitted_indices)), dtype=float)
            otbc_selected_bounds = np.empty_like(otbc_quantization_bounds)
            for block_start in range(0, len(participant_users), OTBC_CLIENT_BLOCK_SIZE):
                block_users = participant_users[
                    block_start:block_start + OTBC_CLIENT_BLOCK_SIZE]
                block_stop = block_start + len(block_users)
                selector = (block_users[:, None], transmitted_indices[None, :])
                cached_vectors = client_services[selector]
                cached_biases = client_service_biases[selector]
                # Target-minus-cache deltas contain missed updates and the
                # prior reconstruction residual, including after inactivity.
                compressed = bounded_output_trust_compress_client_block(
                    target_vectors[None, :, :] - cached_vectors,
                    target_biases[None, :] - cached_biases,
                    users[block_users], otbc_slack[block_users],
                    parameter_width=communication_parameter_bytes,
                )
                decoded_vectors = cached_vectors + compressed.reconstructed_vectors
                decoded_biases = cached_biases + compressed.reconstructed_biases
                client_services[selector] = decoded_vectors
                client_service_biases[selector] = decoded_biases
                # Service reliability affects future AMF updates but not the
                # prediction-output trust inequality; it remains raw FP32.
                client_service_errors[selector] = target_errors[None, :]
                otbc_int16_rows += compressed.compressed_rows
                otbc_fp32_rows += compressed.fallback_rows
                otbc_model_downlink += (
                    len(block_users) * bitmap_per_client + compressed.payload_bytes
                    + len(block_users) * len(transmitted_indices) * 4)
                otbc_quantization_bounds[block_start:block_stop] = compressed.bounds
                otbc_selected_bounds[block_start:block_stop] = compressed.selected_bounds
                otbc_trust_violations += int(np.count_nonzero(
                    compressed.safe_mask &
                    (compressed.selected_bounds > otbc_slack[block_users, None] + 1e-12)))
                if len(transmitted_indices):
                    cache_error = max(
                        float(np.max(np.abs(target_vectors[None, :, :] - decoded_vectors))),
                        float(np.max(np.abs(target_biases[None, :] - decoded_biases))),
                    )
                    otbc_cache_max_abs_error = max(otbc_cache_max_abs_error, cache_error)

        graph_gate_updates, graph_condition, graph_downlink = 0, 1.0, 0
        if graph_residual:
            graph_service_residual, graph_service_confidence = update_server_service_graph_state(
                graph_service_residual, graph_service_confidence,
                graph_service_counts, graph_service_residual_sums)
            graph_gram = graph_rhs = 0.0
            graph_support_count = 0
            for user in range(NUM_USERS):
                indices = np.flatnonzero(support[time_slice, user])
                if not len(indices):
                    continue
                targets = fixed_log_transform(values[time_slice, user, indices])
                base_logits = users[user] @ services[indices].T + user_biases[user] + service_biases[indices]
                signal = graph_residual_signal(
                    np.full(len(indices), user, dtype=int), indices,
                    graph_client_residual, graph_client_confidence,
                    graph_service_residual, graph_service_confidence)
                client_gram, client_rhs, client_count = graph_gate_statistics(
                    base_logits, targets, signal)
                graph_gram += client_gram
                graph_rhs += client_rhs
                graph_support_count += client_count
                graph_gate_updates += 1
            graph_gate, graph_condition = update_graph_residual_gate_from_statistics(
                graph_gate, graph_gram, graph_rhs, graph_support_count)
            graph_uplink, graph_downlink = federated_graph_residual_extra_bytes(
                participants, graph_touched_client_service_rows, int(changed.sum()))
            uplink += graph_uplink
        state_space_updates, state_space_condition = 0, 1.0
        state_space_downlink = 0
        if state_space_mf:
            state_shared_level, state_shared_velocity = advance_shared_state_from_statistics(
                state_shared_level, state_shared_velocity,
                state_service_counts, state_innovation_sums)
            state_gram = np.zeros((STATE_SPACE_GATE_DIMENSION, STATE_SPACE_GATE_DIMENSION))
            state_rhs = np.zeros(STATE_SPACE_GATE_DIMENSION)
            state_support_count = 0
            for user in range(NUM_USERS):
                indices = np.flatnonzero(support[time_slice, user])
                if not len(indices):
                    continue
                targets = fixed_log_transform(values[time_slice, user, indices])
                base_logits = users[user] @ services[indices].T + user_biases[user] + service_biases[indices]
                features = state_space_features(
                    np.full(len(indices), user, dtype=int), indices, users, services,
                    state_private_level, state_private_velocity,
                    state_shared_level, state_shared_velocity)
                client_gram, client_rhs, client_count = state_space_gate_statistics(
                    base_logits, targets, features)
                state_gram += client_gram
                state_rhs += client_rhs
                state_support_count += client_count
                state_space_updates += 1
            state_space_gates, state_space_condition = update_state_space_gates_from_statistics(
                state_space_gates, state_gram, state_rhs, state_support_count)
            state_uplink, state_space_downlink = federated_state_space_extra_bytes(
                participants, state_touched_client_service_rows, args.dimension)
            uplink += state_uplink
        dtnf_gate_updates = 0
        if dtnf_bridge:
            shared_slow = advance_shared_slow_statistics(shared_slow, counts, slow_target_sums)
            gate_gram, gate_rhs = np.zeros((2, 2)), np.zeros(2)
            for user in range(NUM_USERS):
                indices = np.flatnonzero(support[time_slice, user])
                if not len(indices):
                    continue
                targets = fixed_log_transform(values[time_slice, user, indices])
                base_logits = users[user] @ services[indices].T + user_biases[user] + service_biases[indices]
                fast_features, slow_features = temporal_factor_features(
                    private_fast[user], services[indices], shared_slow[indices])
                client_gram, client_rhs = temporal_gate_statistics(
                    base_logits, targets, fast_features, slow_features)
                gate_gram += client_gram
                gate_rhs += client_rhs
                dtnf_gate_updates += 1
            if dtnf_gate_updates:
                temporal_gates = update_temporal_factor_gates_from_statistics(
                    temporal_gates, gate_gram, gate_rhs)
            dtnf_uplink, dtnf_downlink = federated_dtnf_extra_bytes(
                participants, int(support[time_slice].sum()), NUM_SERVICES)
            uplink += dtnf_uplink
        else:
            dtnf_downlink = 0
        cttf_updates, cttf_condition = 0, 1.0
        if causal_time_factor:
            cttf_gram = np.zeros((args.dimension, args.dimension))
            cttf_rhs = np.zeros(args.dimension)
            cttf_support_count = 0
            for user in range(NUM_USERS):
                indices = np.flatnonzero(support[time_slice, user])
                if not len(indices):
                    continue
                targets = fixed_log_transform(values[time_slice, user, indices])
                base_logits = users[user] @ services[indices].T + user_biases[user] + service_biases[indices]
                if graph_residual:
                    base_logits = apply_graph_residual_head(
                        base_logits, np.full(len(indices), user, dtype=int), indices,
                        graph_client_residual, graph_client_confidence,
                        graph_service_residual, graph_service_confidence, graph_gate)
                if state_space_mf:
                    features = state_space_features(
                        np.full(len(indices), user, dtype=int), indices, users, services,
                        state_private_level, state_private_velocity,
                        state_shared_level, state_shared_velocity)
                    base_logits = apply_state_space_residual(base_logits, features, state_space_gates)
                client_gram, client_rhs, client_count = client_time_factor_statistics(
                    base_logits, targets, users[user], services[indices])
                cttf_gram += client_gram
                cttf_rhs += client_rhs
                cttf_support_count += client_count
                cttf_updates += 1
            cttf_delta, cttf_condition = update_time_factor_from_statistics(
                cttf_delta, cttf_gram, cttf_rhs, cttf_support_count)
            cttf_uplink, cttf_downlink = federated_cttf_extra_bytes(participants, args.dimension)
            uplink += cttf_uplink
        else:
            cttf_downlink = 0
        calibration_updates, dor_updates = 0, 0
        current_global_conformal_errors = []
        if calibration_parameters is not None:
            for user in range(NUM_USERS):
                indices = np.flatnonzero(support[time_slice, user])
                if not len(indices):
                    continue
                targets = fixed_log_transform(values[time_slice, user, indices])
                support_services = client_services[user] if client_state_enabled else services
                support_service_biases = (client_service_biases[user]
                                          if client_state_enabled else service_biases)
                logits = (users[user] @ support_services[indices].T + user_biases[user]
                          + support_service_biases[indices])
                if dtnf_bridge:
                    fast_features, slow_features = temporal_factor_features(
                        private_fast[user], services[indices], shared_slow[indices])
                    logits = apply_temporal_factor_head(
                        logits, fast_features, slow_features, temporal_gates)
                if graph_residual:
                    logits = apply_graph_residual_head(
                        logits, np.full(len(indices), user, dtype=int), indices,
                        graph_client_residual, graph_client_confidence,
                        graph_service_residual, graph_service_confidence, graph_gate)
                if state_space_mf:
                    features = state_space_features(
                        np.full(len(indices), user, dtype=int), indices, users, services,
                        state_private_level, state_private_velocity,
                        state_shared_level, state_shared_velocity)
                    logits = apply_state_space_residual(logits, features, state_space_gates)
                if dynamic_online_routing:
                    baseline_predictions = np.array([sigmoid(value) for value in logits])
                    temporal_logits = apply_causal_time_factor(
                        logits, users[user], services[indices], cttf_delta)
                    temporal_predictions = np.array([sigmoid(value) for value in temporal_logits])
                    dor_states[user] = update_dor_state(
                        dor_states[user], targets, baseline_predictions, temporal_predictions)
                    base_predictions = fuse_predictions(
                        dor_states[user], baseline_predictions, temporal_predictions)
                    dor_updates += 1
                else:
                    if causal_time_factor:
                        logits = apply_causal_time_factor(logits, users[user], services[indices], cttf_delta)
                    base_predictions = np.array([sigmoid(value) for value in logits])
                if conformal_mode != "off":
                    previous_normalized = apply_client_fast_calibrator(base_predictions, calibration_parameters[user])
                    previous_raw = inverse_fixed_log_transform(previous_normalized)
                    support_errors = support_absolute_errors(values[time_slice, user, indices], previous_raw)
                    if conformal_mode == "global":
                        current_global_conformal_errors.extend(support_errors.tolist())
                    else:
                        private_conformal_errors[user].extend(support_errors.tolist())
                        conformal_radii[user] = empirical_radius(np.asarray(private_conformal_errors[user]))
                if client_logit_calibration:
                    calibration_parameters[user] = fit_cplc(
                        targets, base_predictions, calibration_parameters[user])
                elif client_monotone_calibration:
                    calibration_parameters[user] = fit_cpmc(
                        targets, base_predictions, calibration_parameters[user])
                else:
                    calibration_parameters[user] = fit_client_fast_calibrator(
                        targets, base_predictions, calibration_parameters[user])
                calibration_updates += 1
        if conformal_mode == "global" and current_global_conformal_errors:
            global_conformal_errors.extend(current_global_conformal_errors)
            conformal_radii[:] = empirical_radius(np.asarray(global_conformal_errors))
        update_seconds = time.perf_counter() - started

        support_service_count = np.zeros(NUM_SERVICES, dtype=int)
        support_service_residual_sum = np.zeros(NUM_SERVICES)
        support_service_absolute_error = np.zeros(NUM_SERVICES)
        support_service_squared_error = np.zeros(NUM_SERVICES)
        support_service_target_sum = np.zeros(NUM_SERVICES)
        support_service_prediction_sum = np.zeros(NUM_SERVICES)
        support_service_normalized_prediction_sum = np.zeros(NUM_SERVICES)
        support_service_normalized_residual_sum = np.zeros(NUM_SERVICES)
        support_service_normalized_prediction_squared_sum = np.zeros(NUM_SERVICES)
        support_service_normalized_prediction_residual_sum = np.zeros(NUM_SERVICES)
        support_service_normalized_residual_squared_sum = np.zeros(NUM_SERVICES)
        if args.diagnostic_service_aggregation or service_raw_residual_calibration:
            # Read-only post-aggregation support proxy.  It is evaluated before
            # query access and is diagnostic only; the current baseline audit
            # has no optional prediction head, so this is the exact query model.
            for user in range(NUM_USERS):
                indices = np.flatnonzero(support[time_slice, user])
                if not len(indices):
                    continue
                query_services = client_services[user] if client_state_enabled else services
                query_service_biases = (client_service_biases[user]
                                        if client_state_enabled else service_biases)
                logits = (users[user] @ query_services[indices].T + user_biases[user]
                          + query_service_biases[indices])
                normalized = np.array([sigmoid(value) for value in logits])
                support_predictions = inverse_fixed_log_transform(normalized)
                support_targets = values[time_slice, user, indices]
                residual = support_targets - support_predictions
                np.add.at(support_service_count, indices, 1)
                np.add.at(support_service_residual_sum, indices, residual)
                np.add.at(support_service_absolute_error, indices, np.abs(residual))
                np.add.at(support_service_squared_error, indices, residual ** 2)
                np.add.at(support_service_target_sum, indices, support_targets)
                np.add.at(support_service_prediction_sum, indices, support_predictions)
                if prediction_conditioned_service_calibration:
                    raw_upper = float(_qos_spec(_ACTIVE_QOS)["log_upper"])
                    normalized_prediction = np.asarray(support_predictions / raw_upper, dtype=np.float32)
                    normalized_residual = np.asarray(residual / raw_upper, dtype=np.float32)
                    np.add.at(support_service_normalized_prediction_sum, indices, normalized_prediction)
                    np.add.at(support_service_normalized_residual_sum, indices, normalized_residual)
                    np.add.at(support_service_normalized_prediction_squared_sum, indices,
                              normalized_prediction * normalized_prediction)
                    np.add.at(support_service_normalized_prediction_residual_sum, indices,
                              normalized_prediction * normalized_residual)
                    np.add.at(support_service_normalized_residual_squared_sum, indices,
                              normalized_residual * normalized_residual)
                elif log_domain_service_calibration:
                    normalized_target = np.asarray(fixed_log_transform(support_targets), dtype=np.float32)
                    normalized_prediction = np.asarray(normalized, dtype=np.float32)
                    normalized_residual = np.asarray(
                        normalized_target - normalized_prediction, dtype=np.float32)
                    np.add.at(support_service_normalized_residual_sum, indices, normalized_residual)
                    np.add.at(support_service_normalized_residual_squared_sum, indices,
                              normalized_residual * normalized_residual)
        raw_service_raw_corrections = mean_service_residual(
            support_service_residual_sum, support_service_count)
        conditional_service_intercepts = np.zeros(NUM_SERVICES)
        conditional_service_slope = 0.0
        log_domain_global_residual = 0.0
        if log_domain_service_calibration:
            log_domain_service_corrections, service_residual_reliability, log_domain_global_residual = (
                log_domain_service_residual(
                    support_service_normalized_residual_sum,
                    support_service_normalized_residual_squared_sum,
                    support_service_count))
            service_raw_corrections = np.zeros(NUM_SERVICES)
            touched = support_service_count > 0
            if np.any(touched):
                mean_prediction = np.zeros(NUM_SERVICES)
                mean_prediction[touched] = (
                    support_service_prediction_sum[touched] / support_service_count[touched])
                calibrated_mean = apply_log_domain_service_residual(
                    mean_prediction[touched], np.flatnonzero(touched),
                    log_domain_service_corrections, touched,
                    _qos_spec(_ACTIVE_QOS)["log_upper"])
                service_raw_corrections[touched] = calibrated_mean - mean_prediction[touched]
        elif prediction_conditioned_service_calibration:
            conditional_service_intercepts, conditional_service_slope, service_residual_reliability = (
                prediction_conditioned_service_residual(
                    support_service_normalized_prediction_sum,
                    support_service_normalized_residual_sum,
                    support_service_normalized_prediction_squared_sum,
                    support_service_normalized_prediction_residual_sum,
                    support_service_normalized_residual_squared_sum,
                    support_service_count))
            service_raw_corrections = np.zeros(NUM_SERVICES)
            touched = support_service_count > 0
            service_raw_corrections[touched] = float(_qos_spec(_ACTIVE_QOS)["log_upper"]) * (
                conditional_service_intercepts[touched]
                + conditional_service_slope
                * support_service_normalized_prediction_sum[touched] / support_service_count[touched])
        elif service_residual_variance_aware_shrinkage:
            service_raw_corrections, service_residual_reliability = variance_aware_service_residual(
                support_service_residual_sum, support_service_squared_error,
                support_service_count)
        elif service_residual_reliability_shrinkage:
            service_raw_corrections, service_residual_reliability = reliable_service_residual(
                support_service_residual_sum, support_service_squared_error,
                support_service_count)
        else:
            service_raw_corrections = raw_service_raw_corrections
            service_residual_reliability = np.zeros(NUM_SERVICES)
            service_residual_reliability[support_service_count > 0] = 1.0

        started = time.perf_counter()
        real, predicted = [], []
        interval_covered, interval_widths, interval_scores = [], [], []
        value_bin_real = [[] for _ in range(len(RT_VALUE_BINS) - 1)]
        value_bin_predicted = [[] for _ in range(len(RT_VALUE_BINS) - 1)]
        service_query_count = np.zeros(NUM_SERVICES, dtype=int)
        service_query_residual_sum = np.zeros(NUM_SERVICES)
        service_query_absolute_error = np.zeros(NUM_SERVICES)
        service_query_squared_error = np.zeros(NUM_SERVICES)
        service_query_target_sum = np.zeros(NUM_SERVICES)
        service_query_prediction_sum = np.zeros(NUM_SERVICES)
        service_query_precal_prediction_sum = np.zeros(NUM_SERVICES)
        service_query_precal_residual_sum = np.zeros(NUM_SERVICES)
        groups = drift_groups(scores)
        group_real, group_predicted = {group: [] for group in ("low", "middle", "high")}, {group: [] for group in ("low", "middle", "high")}
        for user in range(NUM_USERS):
            indices = np.flatnonzero(available[time_slice, user] & ~support[time_slice, user])
            if len(indices):
                query_services = client_services[user] if client_state_enabled else services
                query_service_biases = (client_service_biases[user]
                                        if client_state_enabled else service_biases)
                logits = (users[user] @ query_services[indices].T + user_biases[user]
                          + query_service_biases[indices])
                if dtnf_bridge:
                    fast_features, slow_features = temporal_factor_features(
                        private_fast[user], services[indices], shared_slow[indices])
                    logits = apply_temporal_factor_head(
                        logits, fast_features, slow_features, temporal_gates)
                if graph_residual:
                    logits = apply_graph_residual_head(
                        logits, np.full(len(indices), user, dtype=int), indices,
                        graph_client_residual, graph_client_confidence,
                        graph_service_residual, graph_service_confidence, graph_gate)
                if state_space_mf:
                    features = state_space_features(
                        np.full(len(indices), user, dtype=int), indices, users, services,
                        state_private_level, state_private_velocity,
                        state_shared_level, state_shared_velocity)
                    logits = apply_state_space_residual(logits, features, state_space_gates)
                if dynamic_online_routing:
                    baseline_normalized = np.array([sigmoid(value) for value in logits])
                    temporal_logits = apply_causal_time_factor(
                        logits, users[user], services[indices], cttf_delta)
                    temporal_normalized = np.array([sigmoid(value) for value in temporal_logits])
                    normalized = fuse_predictions(
                        dor_states[user], baseline_normalized, temporal_normalized)
                else:
                    if causal_time_factor:
                        logits = apply_causal_time_factor(logits, users[user], services[indices], cttf_delta)
                    if residuals is not None:
                        reliability = residual_counts[user, indices] / (residual_counts[user, indices] + args.residual_kappa)
                        logits = logits + reliability * residuals[user, indices]
                    normalized = np.array([sigmoid(value) for value in logits])
                if calibration_parameters is not None:
                    normalized = (apply_cplc(normalized, calibration_parameters[user])
                                  if client_logit_calibration else
                                  (apply_cpmc(normalized, calibration_parameters[user])
                                   if client_monotone_calibration else
                                   apply_client_fast_calibrator(normalized, calibration_parameters[user])))
                client_real = values[time_slice, user, indices]
                client_precal_predicted = inverse_fixed_log_transform(normalized)
                client_predicted = client_precal_predicted
                if service_raw_residual_calibration:
                    if log_domain_service_calibration:
                        log_apply = (apply_dual_constrained_log_service_residual
                                     if dual_constrained_log_calibration else
                                     apply_log_domain_service_residual)
                        client_predicted = log_apply(
                            client_predicted, indices, log_domain_service_corrections,
                            support_service_count > 0,
                            _qos_spec(_ACTIVE_QOS)["log_upper"])
                    elif prediction_conditioned_service_calibration:
                        client_predicted = apply_prediction_conditioned_service_residual(
                            client_predicted, indices, conditional_service_intercepts,
                            conditional_service_slope, support_service_count > 0,
                            _qos_spec(_ACTIVE_QOS)["log_upper"])
                    else:
                        client_predicted = apply_service_raw_residual(
                            client_predicted, indices, service_raw_corrections,
                            _qos_spec(_ACTIVE_QOS)["log_upper"])
                real.append(client_real); predicted.append(client_predicted)
                client_interval = None
                if conformal_mode != "off":
                    lower, upper = prediction_interval(client_predicted, conformal_radii[user], RT_UPPER_EXCLUSIVE)
                    client_interval = interval_statistics(client_real, lower, upper)
                    interval_covered.append(client_interval["covered"])
                    interval_widths.append(client_interval["width"])
                    interval_scores.append(client_interval["score"])
                if args.diagnostic_service_aggregation:
                    absolute_error = np.abs(client_predicted - client_real)
                    np.add.at(service_query_count, indices, 1)
                    np.add.at(service_query_residual_sum, indices, client_real - client_predicted)
                    np.add.at(service_query_absolute_error, indices, absolute_error)
                    np.add.at(service_query_squared_error, indices, absolute_error ** 2)
                    np.add.at(service_query_target_sum, indices, client_real)
                    np.add.at(service_query_prediction_sum, indices, client_predicted)
                    np.add.at(service_query_precal_prediction_sum, indices, client_precal_predicted)
                    np.add.at(service_query_precal_residual_sum, indices, client_real - client_precal_predicted)
                if args.diagnostic_value_bins:
                    for bin_index, (lower, upper) in enumerate(zip(RT_VALUE_BINS[:-1], RT_VALUE_BINS[1:])):
                        selected_bin = (client_real > lower) & (client_real <= upper)
                        if np.any(selected_bin):
                            value_bin_real[bin_index].append(client_real[selected_bin])
                            value_bin_predicted[bin_index].append(client_predicted[selected_bin])
                metrics = evaluate(client_real, client_predicted)
                client_row = {"time_slice": time_slice, "client": user, "support_entries": int(support[time_slice, user].sum()),
                              "query_entries": len(client_real), "drift_score": scores[user], "drift_group": groups[user], **metrics}
                if client_interval is not None:
                    client_row.update({"interval_coverage": client_interval["coverage"],
                                       "interval_mean_width": client_interval["mean_width"],
                                       "interval_score": client_interval["interval_score"],
                                       "conformal_radius": float(conformal_radii[user])})
                client_rows.append(client_row)
                if groups[user] in group_real:
                    group_real[groups[user]].append(client_real); group_predicted[groups[user]].append(client_predicted)
        real, predicted = np.concatenate(real), np.concatenate(predicted)
        tail_safe_uplink, tail_safe_downlink = (
            federated_tail_safe_extra_bytes(int(support[time_slice].sum()), args.dimension)
            if tail_safe_aggregation else (0, 0))
        service_calibration_uplink, service_calibration_downlink = (
            (log_domain_service_residual_extra_bytes(
                int(support[time_slice].sum()), participants, NUM_SERVICES)
             if log_domain_service_calibration else
             prediction_conditioned_service_residual_extra_bytes(
                int(support[time_slice].sum()), participants, NUM_SERVICES)
             if prediction_conditioned_service_calibration else
             service_raw_residual_extra_bytes(
                int(support[time_slice].sum()), participants, NUM_SERVICES))
            if service_raw_residual_calibration else (0, 0))
        full_service_model_downlink = (participants * NUM_SERVICES * (args.dimension + 2)
                                       * communication_parameter_bytes)
        if local_mode:
            service_state_downlink = 0
        elif otbc_enabled:
            service_calibration_downlink = event_sparse_plan["chosen_extra_bytes"]
            service_state_downlink = otbc_model_downlink + service_calibration_downlink
        elif getattr(args, "event_sparse_service_downlink", False):
            # ``chosen_extra_bytes`` is the actual correction portion of the
            # selected payload; the full plan includes model rows, bitmap,
            # and any shared calibration scalar.
            service_calibration_downlink = event_sparse_plan["chosen_extra_bytes"]
            service_state_downlink = event_sparse_plan["chosen_bytes"]
        else:
            service_state_downlink = full_service_model_downlink + service_calibration_downlink
        row = {
            "time_slice": time_slice, "support_entries": int(support[time_slice].sum()), "query_entries": len(real),
            "participating_clients": participants,
            "updated_services": (0 if local_mode else int(changed.sum())),
            "update_seconds": update_seconds,
            "evaluation_seconds": time.perf_counter() - started,
            "uplink_bytes": (0 if local_mode else uplink + tail_safe_uplink + service_calibration_uplink),
            "downlink_bytes": (0 if local_mode else service_state_downlink + dtnf_downlink + cttf_downlink + state_space_downlink
                               + graph_downlink + tail_safe_downlink),
            "event_sparse_service_downlink_mode": event_sparse_plan["mode"],
            "event_sparse_service_downlink_bitmap_bytes": event_sparse_plan["bitmap_bytes"],
            "event_sparse_service_downlink_transmitted_rows": event_sparse_plan["transmitted_rows"],
            "event_sparse_service_downlink_full_bytes": event_sparse_plan["full_bytes"],
            "event_sparse_service_downlink_sparse_bytes": event_sparse_plan["sparse_bytes"],
            "event_sparse_service_downlink_saved_bytes": event_sparse_plan["saved_bytes"],
            "communication_parameter_bytes": communication_parameter_bytes,
            "otbc_int16_rows": otbc_int16_rows,
            "otbc_fp32_rows": otbc_fp32_rows,
            "otbc_model_downlink_bytes": otbc_model_downlink,
            "otbc_fp32_reference_downlink_bytes": event_sparse_plan["chosen_bytes"],
            "otbc_mean_trust_slack": (float(np.mean(otbc_slack[otbc_user_norm > 0]))
                                       if np.any(otbc_user_norm > 0) else 0.0),
            "otbc_mean_quantization_bound": (float(np.mean(otbc_quantization_bounds))
                                              if otbc_quantization_bounds.size else 0.0),
            "otbc_mean_selected_bound": (float(np.mean(otbc_selected_bounds))
                                          if otbc_selected_bounds.size else 0.0),
            "otbc_trust_violations": otbc_trust_violations,
            "otbc_cache_max_abs_error": otbc_cache_max_abs_error,
            **evaluate(real, predicted),
        }
        if state_space_mf:
            row.update({
                "state_space_updates": state_space_updates,
                "state_space_gate_norm": float(np.linalg.norm(state_space_gates)),
                "state_space_gate_max_abs": float(np.max(np.abs(state_space_gates))),
                "state_space_condition_number": state_space_condition,
                "mean_private_state_norm": float(np.linalg.norm(state_private_level, axis=1).mean()),
                "mean_shared_state_norm": float(np.linalg.norm(state_shared_level, axis=1).mean()),
            })
        if graph_residual:
            row.update({
                "graph_gate_updates": graph_gate_updates,
                "graph_gate": float(np.tanh(graph_gate)),
                "graph_condition_number": graph_condition,
                "mean_client_graph_confidence": float(np.mean(graph_client_confidence)),
                "mean_service_graph_confidence": float(np.mean(graph_service_confidence)),
            })
        if robust_amf:
            finite_scales = robust_scales[np.isfinite(robust_scales)]
            row.update({
                "robust_scale_mean": float(np.mean(finite_scales)) if len(finite_scales) else 0.0,
                "robust_scale_min": float(np.min(finite_scales)) if len(finite_scales) else 0.0,
                "robust_scale_max": float(np.max(finite_scales)) if len(finite_scales) else 0.0,
            })
        if args.aggregation == "stca":
            row.update({"mean_aggregation_entropy": float(np.mean(aggregation_entropy)) if aggregation_entropy else 0.0,
                        "mean_effective_clients": float(np.mean(effective_clients)) if effective_clients else 0.0,
                        "history_weighted_services": history_services})
        elif _is_frozen_fedavg_residual_control(args):
            # FedAvg has no temporal-coherence weighting or server history;
            # expose that fact explicitly for the paired RQ2 control.
            row.update({"history_weighted_services": 0})
        if args.stca_component in ("adaptive_evidence", "bidirectional_evidence"):
            row.update({
                "mean_service_drift_intensity": (float(np.mean(adaptive_drift_intensity))
                                                  if adaptive_drift_intensity else 0.0),
                "mean_adaptive_consensus_gate": (float(np.mean(adaptive_consensus_gate))
                                                  if adaptive_consensus_gate else 0.0),
                "mean_proposal_uncertainty": (float(np.mean(proposal_uncertainty_rows))
                                               if proposal_uncertainty_rows else 0.0),
                "mean_reactive_transfer": (float(np.mean(reactive_transfer_rows))
                                             if reactive_transfer_rows else 0.0),
                "mean_risk_suppression": (float(np.mean(risk_suppression_rows))
                                           if risk_suppression_rows else 0.0),
            })
        if selective_dual_memory:
            row.update({
                "sdm_reactive_eligible_services": sdm_reactive_eligible,
                "sdm_reactive_selected_services": sdm_reactive_selected,
                "sdm_reactive_selection_fraction": (sdm_reactive_selected / sdm_reactive_eligible
                                                      if sdm_reactive_eligible else 0.0),
                "mean_sdm_stable_support_risk": (float(np.mean(sdm_stable_risk))
                                                  if sdm_stable_risk else 0.0),
                "mean_sdm_reactive_support_risk": (float(np.mean(sdm_reactive_risk))
                                                    if sdm_reactive_risk else 0.0),
                "mean_sdm_stable_proposal_dispersion": (float(np.mean(sdm_stable_dispersion))
                                                         if sdm_stable_dispersion else 0.0),
                "mean_sdm_reactive_proposal_dispersion": (float(np.mean(sdm_reactive_dispersion))
                                                           if sdm_reactive_dispersion else 0.0),
            })
        if tail_safe_aggregation:
            row.update({
                "tail_safe_validated_services": tail_safe_validated_services,
                "tail_safe_fallback_services": tail_safe_fallback_services,
                "tail_safe_fallback_fraction": (tail_safe_fallback_services / tail_safe_validated_services
                                                 if tail_safe_validated_services else 0.0),
                "tail_safe_fedavg_support_sse": tail_safe_fedavg_support_sse,
                "tail_safe_stca_support_sse": tail_safe_stca_support_sse,
                "tail_safe_extra_uplink_bytes": tail_safe_uplink,
                "tail_safe_extra_downlink_bytes": tail_safe_downlink,
            })
        if service_raw_residual_calibration:
            touched_service_residual = support_service_count > 0
            row.update({
                "service_raw_residual_touched": int(np.count_nonzero(support_service_count)),
                "mean_abs_service_raw_correction": float(np.mean(np.abs(
                    service_raw_corrections[touched_service_residual]))),
                "mean_abs_unshrunk_service_raw_correction": float(np.mean(np.abs(
                    raw_service_raw_corrections[touched_service_residual]))),
                "mean_service_residual_reliability": float(np.mean(
                    service_residual_reliability[touched_service_residual])),
                "full_service_residual_reliability_fraction": float(np.mean(
                    service_residual_reliability[touched_service_residual] >= 1.0 - 1e-12)),
                "prediction_conditioned_service_slope": float(conditional_service_slope),
                "log_domain_global_service_residual": float(log_domain_global_residual),
                "service_raw_residual_extra_uplink_bytes": service_calibration_uplink,
                "service_raw_residual_extra_downlink_bytes": service_calibration_downlink,
            })
        if conformal_mode != "off":
            covered = np.concatenate(interval_covered); widths = np.concatenate(interval_widths); scores_interval = np.concatenate(interval_scores)
            row.update({"conformal_mode": conformal_mode,
                        "interval_coverage": float(np.mean(covered)),
                        "interval_coverage_error": float(abs(np.mean(covered) - (1.0 - CONFORMAL_ALPHA))),
                        "interval_mean_width": float(np.mean(widths)),
                        "interval_score": float(np.mean(scores_interval)),
                        "mean_conformal_radius": float(np.mean(conformal_radii))})
        if args.stca_component == "persistent":
            row.update({"persistent_drift_clients": int(np.count_nonzero(persistent_drift)),
                        "mean_persistent_drift": float(np.mean(persistent_drift)),
                        "mean_service_persistent_risk": (float(np.mean(persistent_service_risk))
                                                         if persistent_service_risk else 0.0)})
        if residuals is not None:
            query_counts = residual_counts[available[time_slice] & ~support[time_slice]]
            row.update({"private_residual_entries": int(np.count_nonzero(residual_counts)),
                        "query_seen_service_fraction": float(np.mean(query_counts > 0)),
                        "mean_query_residual_reliability": float(np.mean(query_counts / (query_counts + args.residual_kappa)))})
        if calibration_parameters is not None:
            if client_logit_calibration:
                row.update({
                    "client_calibration_updates": calibration_updates,
                    "mean_cplc_gain": float(np.mean(calibration_parameters[:, 0])),
                    "mean_abs_cplc_offset": float(np.mean(np.abs(calibration_parameters[:, 1]))),
                    "cplc_bound_fraction": float(np.mean(
                        (calibration_parameters[:, 0] == CPLC_GAIN_BOUNDS[0]) |
                        (calibration_parameters[:, 0] == CPLC_GAIN_BOUNDS[1]) |
                        (calibration_parameters[:, 1] == CPLC_OFFSET_BOUNDS[0]) |
                        (calibration_parameters[:, 1] == CPLC_OFFSET_BOUNDS[1]))),
                })
            elif client_monotone_calibration:
                row.update({
                    "client_calibration_updates": calibration_updates,
                    "mean_cpmc_lower_control": float(np.mean(calibration_parameters[:, 0])),
                    "mean_cpmc_middle_control": float(np.mean(calibration_parameters[:, 1])),
                    "mean_cpmc_upper_control": float(np.mean(calibration_parameters[:, 2])),
                    "cpmc_flat_client_fraction": float(np.mean(np.ptp(calibration_parameters, axis=1) <= 1e-12)),
                })
            else:
                row.update({
                    "client_calibration_updates": calibration_updates,
                    "mean_client_calibration_gain": float(np.mean(calibration_parameters[:, 0])),
                    "mean_abs_client_calibration_offset": float(np.mean(np.abs(calibration_parameters[:, 1]))),
                })
        if dtnf_bridge:
            row.update({
                "dtnf_gate_updates": dtnf_gate_updates,
                "dtnf_fast_gate": float(np.tanh(temporal_gates[0])),
                "dtnf_slow_gate": float(np.tanh(temporal_gates[1])),
                "mean_private_fast_norm": float(np.linalg.norm(private_fast, axis=1).mean()),
                "mean_shared_slow_norm": float(np.linalg.norm(shared_slow, axis=1).mean()),
            })
        if causal_time_factor:
            row.update({
                "cttf_updates": cttf_updates,
                "cttf_factor_norm": float(np.linalg.norm(cttf_delta)),
                "cttf_factor_min": float(np.min(cttf_delta)),
                "cttf_factor_max": float(np.max(cttf_delta)),
                "cttf_bound_fraction": float(np.mean(np.abs(cttf_delta) >= CTTF_DELTA_BOUND - 1e-12)),
                "cttf_condition_number": cttf_condition,
            })
        if dynamic_online_routing:
            participating_weights = np.array([
                router_weights(dor_states[user])[1]
                for user in range(NUM_USERS) if support[time_slice, user].any()
            ])
            row.update({
                "dor_updates": dor_updates,
                "mean_dor_temporal_weight": (float(np.mean(participating_weights))
                                               if len(participating_weights) else 0.5),
                "min_dor_temporal_weight": (float(np.min(participating_weights))
                                              if len(participating_weights) else 0.5),
                "max_dor_temporal_weight": (float(np.max(participating_weights))
                                              if len(participating_weights) else 0.5),
            })
        rows.append(row)
        if args.diagnostic_service_aggregation:
            for service in np.flatnonzero(service_query_count):
                query_count = int(service_query_count[service])
                service_time_rows.append({
                    "time_slice": time_slice, "service": int(service), "support_clients": int(counts[service]),
                    "query_entries": query_count,
                    "support_mean_residual": (float(support_service_residual_sum[service] / support_service_count[service])
                                              if support_service_count[service] else 0.0),
                    "support_mean_target": (float(support_service_target_sum[service] / support_service_count[service])
                                            if support_service_count[service] else 0.0),
                    "support_mean_prediction": (float(support_service_prediction_sum[service] / support_service_count[service])
                                                if support_service_count[service] else 0.0),
                    "support_mae": (float(support_service_absolute_error[service] / support_service_count[service])
                                    if support_service_count[service] else 0.0),
                    "support_rmse": (float(np.sqrt(support_service_squared_error[service] / support_service_count[service]))
                                     if support_service_count[service] else 0.0),
                    "query_mean_residual": float(service_query_residual_sum[service] / query_count),
                    "query_mean_target": float(service_query_target_sum[service] / query_count),
                    "query_mean_prediction": float(service_query_prediction_sum[service] / query_count),
                    "query_mean_prediction_before_srrc": float(service_query_precal_prediction_sum[service] / query_count),
                    "query_mean_residual_before_srrc": float(service_query_precal_residual_sum[service] / query_count),
                    "unshrunk_service_raw_correction": (float(raw_service_raw_corrections[service])
                                                        if service_raw_residual_calibration else 0.0),
                    "service_residual_reliability": (float(service_residual_reliability[service])
                                                     if service_raw_residual_calibration else 0.0),
                    "applied_service_raw_correction": (float(service_raw_corrections[service])
                                                       if service_raw_residual_calibration else 0.0),
                    "query_mean_effective_correction": float(
                        (service_query_prediction_sum[service] - service_query_precal_prediction_sum[service])
                        / query_count),
                    "mae": float(service_query_absolute_error[service] / query_count),
                    "rmse": float(np.sqrt(service_query_squared_error[service] / query_count)),
                })
        if args.diagnostic_value_bins:
            for bin_index, (lower, upper) in enumerate(zip(RT_VALUE_BINS[:-1], RT_VALUE_BINS[1:])):
                if value_bin_real[bin_index]:
                    bin_real = np.concatenate(value_bin_real[bin_index])
                    bin_predicted = np.concatenate(value_bin_predicted[bin_index])
                    value_bin_rows.append({"time_slice": time_slice, "lower_exclusive": lower, "upper_inclusive": upper,
                                           "query_entries": len(bin_real), **evaluate(bin_real, bin_predicted)})
        for group in group_real:
            if group_real[group]:
                selected = groups == group
                group_rows.append({"time_slice": time_slice, "drift_group": group, "eligible_clients": int(selected.sum()),
                                   "query_entries": int(sum(len(x) for x in group_real[group])), "mean_drift_score": float(np.mean(scores[selected])),
                                   **evaluate(np.concatenate(group_real[group]), np.concatenate(group_predicted[group]))})
        print(json.dumps(row))
    summary = {key: float(np.mean([row[key] for row in rows])) for key in ("mae", "rmse", "mre", "npre", "update_seconds", "evaluation_seconds", "uplink_bytes", "downlink_bytes")}
    args.read_only_audit_result = read_only_audit
    return rows, client_rows, group_rows, value_bin_rows, service_time_rows, service_aggregation_rows, summary


def _is_frozen_fedavg_residual_control(args):
    """Return whether ``args`` is the approved FedAvg residual control.

    The current-support residual calibration itself is independent of STCA,
    but the RQ2 control is intentionally narrower: it reuses the S4
    log-domain/dual-constrained calibration, keeps C-OMF output-stability
    control and ESSD's FP32 wire contract, and does not combine OTBC with this
    aggregation ablation.  Keeping this as an explicit predicate prevents a
    bare assertion removal from silently creating an unvalidated variant.
    """

    return bool(
        getattr(args, "aggregation", None) == "fedavg"
        and getattr(args, "stca_component", None) == "full"
        and getattr(args, "causal_support_output_trust_update", False)
        and getattr(args, "event_sparse_service_downlink", False)
        and int(getattr(args, "communication_parameter_bytes", -1)) == 4
        and getattr(args, "service_raw_residual_calibration", False)
        and getattr(args, "log_domain_service_calibration", False)
        and getattr(args, "dual_constrained_log_calibration", False)
        and not getattr(args, "output_trust_bounded_compression", False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qos-type", choices=("rt", "tp"), default="rt",
                        help="QoS attribute and frozen preprocessing contract (default: rt)")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--mask-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--eta", type=float, default=.8)
    parser.add_argument("--l2", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=.3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--causal-support-output-trust-update", action="store_true",
                        help="enable CSOTU support-set block-GN update with output trust control")
    # RQ2 exposes the already-implemented count-weighted FedAvg control, the
    # STCA aggregator, and a local-only C-OMF control.  Local mode keeps a
    # private service-state table per client and performs no communication.
    parser.add_argument("--aggregation", choices=("fedavg", "stca", "local"), default="stca")
    parser.add_argument("--stca-component", choices=("full", "evidence"), default="evidence")
    parser.add_argument("--event-sparse-service-downlink", action="store_true",
                        help="enable FedSRA ESSD full-vs-bitmap+changed-service-row downlink selection")
    parser.add_argument("--output-trust-bounded-compression", action="store_true",
                        help="enable real client-cache INT16/FP32 OTBC downlink after CSOTU")
    parser.add_argument("--communication-parameter-bytes", type=int, choices=(4, 8), default=8,
                        help="wire width for uncompressed model parameters (default: 8)")
    parser.add_argument("--service-raw-residual-calibration", action="store_true",
                        help="add current-support mean raw residual per service at query prediction")
    parser.add_argument("--log-domain-service-calibration", action="store_true",
                        help="aggregate float32 support-only service residuals in the fixed log1p domain")
    parser.add_argument("--dual-constrained-log-calibration", action="store_true",
                        help="limit LCSRRC by the smaller normalized log/raw-domain displacement")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--max-time-slices", type=int, default=None)
    parser.add_argument("--audit-read-only", action="store_true",
                        help="snapshot model state around current query probes and require exact read-only equality")
    parser.add_argument("--audit-future", action="store_true",
                        help="snapshot model state around next-slice future probes and require exact read-only equality")
    # Retired experiment paths stay unreachable inside this historical
    # single-file runner.  Explicit disabled defaults preserve the frozen
    # final-path numerics while removing their CLI and import dependencies.
    parser.set_defaults(
        fedprox_mu=0.0,
        support_conditioned_stability_plasticity_update=False,
        support_novelty_subspace_update=False,
        diagnostic_value_bins=False,
        diagnostic_service_aggregation=False,
        private_service_residual=False,
        client_fast_calibration=False,
        client_monotone_calibration=False,
        client_logit_calibration=False,
        dtnf_bridge=False,
        causal_time_factor=False,
        state_space_mf=False,
        graph_residual_backbone=False,
        robust_amf=False,
        robust_delta=0.1,
        robust_scale_floor=0.001,
        robust_scale_decay=0.9,
        tail_safe_aggregation=False,
        dual_scale_gradient=False,
        bounded_tail_gradient=False,
        prediction_gated_tail_gradient=False,
        target_gated_tail_gradient=False,
        service_residual_reliability_shrinkage=False,
        service_residual_variance_aware_shrinkage=False,
        prediction_conditioned_service_calibration=False,
        dynamic_online_routing=False,
        conformal_mode="off",
        residual_eta=0.1,
        residual_l2=1e-4,
        residual_decay=0.99,
        residual_kappa=3.0,
    )
    args = parser.parse_args()
    assert args.fedprox_mu >= 0
    assert args.residual_eta >= 0 and args.residual_l2 >= 0 and 0 < args.residual_decay <= 1 and args.residual_kappa > 0
    assert args.robust_delta > 0 and args.robust_scale_floor > 0 and 0 <= args.robust_scale_decay < 1
    assert args.aggregation == "fedavg" or args.fedprox_mu == 0, "STCA excludes FedProx"
    assert args.aggregation == "stca" or args.stca_component == "full", "STCA components require --aggregation stca"
    if args.aggregation == "local":
        assert args.causal_support_output_trust_update, (
            "local aggregation is the Local C-OMF control and requires --causal-support-output-trust-update")
        assert args.fedprox_mu == 0, "Local C-OMF has no server reference for FedProx"
        assert not any((
            args.support_conditioned_stability_plasticity_update,
            args.support_novelty_subspace_update,
            args.event_sparse_service_downlink,
            args.output_trust_bounded_compression,
            args.private_service_residual,
            args.client_fast_calibration,
            args.client_monotone_calibration,
            args.client_logit_calibration,
            args.dtnf_bridge,
            args.causal_time_factor,
            args.state_space_mf,
            args.graph_residual_backbone,
            args.robust_amf,
            args.tail_safe_aggregation,
            args.dual_scale_gradient,
            args.bounded_tail_gradient,
            args.prediction_gated_tail_gradient,
            args.target_gated_tail_gradient,
            args.service_raw_residual_calibration,
            args.service_residual_reliability_shrinkage,
            args.service_residual_variance_aware_shrinkage,
            args.prediction_conditioned_service_calibration,
            args.log_domain_service_calibration,
            args.dual_constrained_log_calibration,
            args.dynamic_online_routing,
            args.conformal_mode != "off",
        )), "Local C-OMF excludes server aggregation, TSD, and alternative predictor extensions"
    if args.client_fast_calibration:
        assert args.aggregation == "stca" and args.stca_component == "evidence", "client calibration requires frozen STCA-v2"
        assert args.fedprox_mu == 0, "client calibration excludes FedProx"
    if args.client_monotone_calibration:
        assert not args.client_fast_calibration, "CPMC replaces rather than stacks CTAC"
        assert args.causal_time_factor, "CPMC first-round evaluation requires frozen CTTF"
        assert args.aggregation == "stca" and args.stca_component == "evidence", "CPMC requires frozen STCA-v2"
        assert args.fedprox_mu == 0 and not args.private_service_residual and not args.dtnf_bridge, "CPMC excludes FedProx, residual and DTNF"
    if args.client_logit_calibration:
        assert not args.client_fast_calibration and not args.client_monotone_calibration, "CPLC replaces rather than stacks CTAC/CPMC"
        assert args.causal_time_factor, "CPLC first-round evaluation requires frozen CTTF"
        assert args.aggregation == "stca" and args.stca_component == "evidence", "CPLC requires frozen STCA-v2"
        assert args.fedprox_mu == 0 and not args.private_service_residual and not args.dtnf_bridge, "CPLC excludes FedProx, residual and DTNF"
    if args.dtnf_bridge:
        assert args.aggregation == "stca" and args.stca_component == "evidence", "DTNF bridge requires frozen STCA-v2"
        assert args.fedprox_mu == 0 and not args.private_service_residual, "DTNF bridge excludes FedProx and private residual"
    if args.state_space_mf:
        assert args.aggregation == "stca" and args.stca_component == "evidence", "state-space MF requires frozen STCA-v2"
        assert args.fedprox_mu == 0 and not args.private_service_residual, "state-space MF excludes FedProx and private residual"
        assert not args.dtnf_bridge, "state-space MF and DTNF are independent backbone candidates"
    if args.graph_residual_backbone:
        assert args.aggregation == "stca" and args.stca_component == "evidence", "graph residual requires frozen STCA-v2"
        assert args.fedprox_mu == 0 and not args.private_service_residual, "graph residual excludes FedProx and private residual"
        assert not args.dtnf_bridge and not args.state_space_mf, "graph residual, state-space MF and DTNF are independent backbone candidates"
    if args.robust_amf:
        assert args.fedprox_mu == 0 and not args.private_service_residual, "Robust-AMF excludes FedProx and the old private residual"
        assert not args.dtnf_bridge and not args.state_space_mf and not args.graph_residual_backbone, "Robust-AMF is an independent backbone candidate"
    if args.causal_support_output_trust_update:
        assert not args.support_conditioned_stability_plasticity_update, "CSOTU and SCSPU are mutually exclusive"
        assert not args.support_novelty_subspace_update, "CSOTU and SNSU are mutually exclusive"
        assert not any((args.private_service_residual, args.client_fast_calibration,
                        args.client_monotone_calibration, args.client_logit_calibration,
                        args.dtnf_bridge, args.causal_time_factor, args.state_space_mf,
                        args.graph_residual_backbone, args.robust_amf,
                        args.tail_safe_aggregation, args.dual_scale_gradient,
                        args.bounded_tail_gradient, args.prediction_gated_tail_gradient,
                        args.target_gated_tail_gradient, args.dynamic_online_routing)), (
                            "CSOTU is an isolated COQMF local-update candidate")
    if args.output_trust_bounded_compression:
        assert (args.causal_support_output_trust_update or
                args.support_conditioned_stability_plasticity_update or
                args.support_novelty_subspace_update), (
                    "OTBC requires CSOTU, SCSPU, or SNSU")
        assert args.event_sparse_service_downlink, "OTBC requires the frozen ESSD representation"
        assert args.communication_parameter_bytes == 4, "OTBC uses the frozen FP32 communication baseline"
        assert args.aggregation == "stca" and args.stca_component == "evidence", (
            "OTBC requires the frozen HEGSA evidence aggregation")
    if args.support_conditioned_stability_plasticity_update:
        assert not args.causal_support_output_trust_update, "SCSPU and CSOTU are mutually exclusive"
        assert not args.support_novelty_subspace_update, "SCSPU and SNSU are mutually exclusive"
        assert args.fedprox_mu == 0, "SCSPU uses the frozen COQMF local proposal without FedProx"
        assert not any((args.private_service_residual, args.client_fast_calibration,
                        args.client_monotone_calibration, args.client_logit_calibration,
                        args.dtnf_bridge, args.causal_time_factor, args.state_space_mf,
                        args.graph_residual_backbone, args.robust_amf,
                        args.tail_safe_aggregation, args.dual_scale_gradient,
                        args.bounded_tail_gradient, args.prediction_gated_tail_gradient,
                        args.target_gated_tail_gradient, args.dynamic_online_routing)), (
                            "SCSPU is an isolated COQMF local-update candidate; current FedSRA flags remain allowed")
    if args.support_novelty_subspace_update:
        assert not args.causal_support_output_trust_update, "SNSU and CSOTU are mutually exclusive"
        assert not args.support_conditioned_stability_plasticity_update, "SNSU and SCSPU are mutually exclusive"
        assert args.fedprox_mu == 0, "SNSU uses the frozen COQMF local proposal without FedProx"
        assert not any((args.private_service_residual, args.client_fast_calibration,
                        args.client_monotone_calibration, args.client_logit_calibration,
                        args.dtnf_bridge, args.causal_time_factor, args.state_space_mf,
                        args.graph_residual_backbone, args.robust_amf,
                        args.tail_safe_aggregation, args.dual_scale_gradient,
                        args.bounded_tail_gradient, args.prediction_gated_tail_gradient,
                        args.target_gated_tail_gradient, args.dynamic_online_routing)), (
                            "SNSU is an isolated COQMF local-update candidate; current FedSRA flags remain allowed")
    if args.tail_safe_aggregation:
        assert args.qos_type == "tp", "tail-safe aggregation is a TP tail-RMSE candidate"
        assert args.aggregation == "stca" and args.stca_component == "evidence", "tail-safe aggregation requires frozen STCA-v2"
        assert args.fedprox_mu == 0, "tail-safe aggregation excludes FedProx"
        assert not any((args.private_service_residual, args.client_fast_calibration,
                        args.client_monotone_calibration, args.client_logit_calibration,
                        args.dtnf_bridge, args.causal_time_factor, args.state_space_mf,
                        args.graph_residual_backbone, args.robust_amf,
                        args.dynamic_online_routing)), "tail-safe B0 is an isolated STCA-v2 extension"
    if args.dual_scale_gradient:
        assert args.qos_type == "tp", "dual-scale gradients are a TP RMSE candidate"
        assert args.aggregation == "stca" and args.stca_component == "evidence", "dual-scale gradients require frozen STCA-v2"
        assert args.fedprox_mu == 0 and not args.robust_amf and not args.tail_safe_aggregation, "dual-scale gradients are an isolated AMF candidate"
        assert not any((args.private_service_residual, args.client_fast_calibration,
                        args.client_monotone_calibration, args.client_logit_calibration,
                        args.dtnf_bridge, args.causal_time_factor, args.state_space_mf,
                        args.graph_residual_backbone, args.dynamic_online_routing)), "dual-scale B0 excludes other candidates"
    if args.bounded_tail_gradient:
        assert args.qos_type == "tp", "bounded tail gradients are a TP RMSE candidate"
        assert args.aggregation == "stca" and args.stca_component == "evidence", "bounded tail gradients require frozen STCA-v2"
        assert args.fedprox_mu == 0 and not args.robust_amf and not args.tail_safe_aggregation and not args.dual_scale_gradient, "bounded tail gradients are an isolated AMF candidate"
        assert not any((args.private_service_residual, args.client_fast_calibration,
                        args.client_monotone_calibration, args.client_logit_calibration,
                        args.dtnf_bridge, args.causal_time_factor, args.state_space_mf,
                        args.graph_residual_backbone, args.dynamic_online_routing)), "bounded tail B0 excludes other candidates"
    if args.prediction_gated_tail_gradient:
        assert args.qos_type == "tp", "prediction-gated tail gradients are a TP RMSE candidate"
        assert args.aggregation == "stca" and args.stca_component == "evidence", "prediction-gated tail gradients require frozen STCA-v2"
        assert args.fedprox_mu == 0 and not args.robust_amf and not args.tail_safe_aggregation and not args.dual_scale_gradient and not args.bounded_tail_gradient, "prediction-gated tail gradients are isolated"
        assert not any((args.private_service_residual, args.client_fast_calibration,
                        args.client_monotone_calibration, args.client_logit_calibration,
                        args.dtnf_bridge, args.causal_time_factor, args.state_space_mf,
                        args.graph_residual_backbone, args.dynamic_online_routing)), "prediction-gated tail B0 excludes other candidates"
    if args.target_gated_tail_gradient:
        assert args.qos_type == "tp", "target-gated tail gradients are a TP RMSE candidate"
        assert args.aggregation == "stca" and args.stca_component == "evidence", "target-gated tail gradients require frozen STCA-v2"
        assert args.fedprox_mu == 0 and not args.robust_amf and not args.tail_safe_aggregation and not args.dual_scale_gradient and not args.bounded_tail_gradient and not args.prediction_gated_tail_gradient, "target-gated tail gradients are isolated"
        assert not any((args.private_service_residual, args.client_fast_calibration,
                        args.client_monotone_calibration, args.client_logit_calibration,
                        args.dtnf_bridge, args.causal_time_factor, args.state_space_mf,
                        args.graph_residual_backbone, args.dynamic_online_routing)), "target-gated tail B0 excludes other candidates"
    if args.service_raw_residual_calibration:
        stca_residual_control = (
            args.aggregation == "stca"
            and args.stca_component in (
                "evidence", "adaptive_evidence", "bidirectional_evidence",
                "selective_dual_memory"))
        assert stca_residual_control or _is_frozen_fedavg_residual_control(args), (
            "service raw-residual calibration requires HEGSA or its "
            "drift-adaptive extension, or the frozen FedAvg+DCLCSRRC control")
        assert args.fedprox_mu == 0 and not args.robust_amf and not args.tail_safe_aggregation and not args.dual_scale_gradient and not args.bounded_tail_gradient and not args.prediction_gated_tail_gradient and not args.target_gated_tail_gradient, "service raw-residual calibration is isolated"
        assert not any((args.private_service_residual, args.client_fast_calibration,
                        args.client_monotone_calibration, args.client_logit_calibration,
                        args.dtnf_bridge, args.causal_time_factor, args.state_space_mf,
                        args.graph_residual_backbone, args.dynamic_online_routing)), "service raw-residual calibration excludes other candidates"
    if args.service_residual_reliability_shrinkage:
        assert args.service_raw_residual_calibration, "reliability shrinkage requires service raw-residual calibration"
    if args.service_residual_variance_aware_shrinkage:
        assert args.service_raw_residual_calibration, "variance-aware shrinkage requires service raw-residual calibration"
        assert not args.service_residual_reliability_shrinkage, "reliable-SRRC sibling flags are mutually exclusive"
    if args.prediction_conditioned_service_calibration:
        assert args.service_raw_residual_calibration, "prediction-conditioned calibration requires service raw-residual calibration"
        assert not args.service_residual_reliability_shrinkage and not args.service_residual_variance_aware_shrinkage, "conditional and scalar SRRC variants are mutually exclusive"
        assert not args.log_domain_service_calibration, "conditional residual variants are mutually exclusive"
    if args.log_domain_service_calibration:
        assert args.service_raw_residual_calibration, "log-domain calibration requires service raw-residual calibration"
        assert not args.service_residual_reliability_shrinkage and not args.service_residual_variance_aware_shrinkage, "log-domain and scalar SRRC variants are mutually exclusive"
    if args.dual_constrained_log_calibration:
        assert args.log_domain_service_calibration, "dual-constrained calibration requires log-domain calibration"
    if args.causal_time_factor:
        assert args.aggregation == "stca" and args.stca_component == "evidence", "CTTF requires frozen STCA-v2"
        assert args.fedprox_mu == 0, "CTTF excludes FedProx"
        assert not args.dtnf_bridge, "CTTF excludes DTNF"
    if args.dynamic_online_routing:
        assert args.causal_time_factor and args.client_fast_calibration, "DOR requires the frozen CTTF+CTAC mainline"
        assert args.aggregation == "stca" and args.stca_component == "evidence", "DOR requires frozen STCA-v2"
    if args.conformal_mode != "off":
        assert args.client_fast_calibration and args.causal_time_factor, "conformal intervals require frozen CTTF+CTAC point predictor"
        assert not args.dynamic_online_routing and not args.client_monotone_calibration and not args.client_logit_calibration, "conformal first round excludes alternative heads"
    spec = _qos_spec(args.qos_type)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert manifest.get("qos_type", "rt") == args.qos_type
    available, support = load_masks(args.mask_file, args.qos_type)
    values, sha256 = load_values(args.data_file, args.qos_type)
    assert sha256 == manifest["data_sha256"]
    assert np.array_equal(available, np.isfinite(values))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, client_rows, group_rows, value_bin_rows, service_time_rows, service_aggregation_rows, summary = run(args, values, available, support)
    common_sample_keys = []
    for time_slice in range(args.max_time_slices or values.shape[0]):
        for user in range(values.shape[1]):
            query_services = np.flatnonzero(available[time_slice, user] & ~support[time_slice, user])
            for service in query_services:
                common_sample_keys.append({"time_slice": int(time_slice), "user": int(user), "service": int(service)})
                if len(common_sample_keys) >= 8:
                    break
            if len(common_sample_keys) >= 8:
                break
        if len(common_sample_keys) >= 8:
            break
    with (args.run_dir / "per_time.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    for name, data in (("per_client_time.csv", client_rows), ("per_time_drift_group.csv", group_rows)):
        if data:
            with (args.run_dir / name).open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=data[0].keys())
                writer.writeheader(); writer.writerows(data)
    if value_bin_rows:
        with (args.run_dir / "per_time_value_bin.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=value_bin_rows[0].keys())
            writer.writeheader(); writer.writerows(value_bin_rows)
    for name, data in (("per_service_time.csv", service_time_rows),
                       ("per_service_aggregation.csv", service_aggregation_rows)):
        if data:
            with (args.run_dir / name).open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=data[0].keys())
                writer.writeheader(); writer.writerows(data)
    config = {key: getattr(args, key) for key in ("qos_type", "dimension", "eta", "l2", "beta", "fedprox_mu", "local_epochs", "causal_support_output_trust_update", "support_conditioned_stability_plasticity_update", "support_novelty_subspace_update", "aggregation", "stca_component", "event_sparse_service_downlink", "output_trust_bounded_compression", "communication_parameter_bytes", "diagnostic_value_bins", "diagnostic_service_aggregation", "private_service_residual", "client_fast_calibration", "client_monotone_calibration", "client_logit_calibration", "dtnf_bridge", "causal_time_factor", "state_space_mf", "graph_residual_backbone", "robust_amf", "robust_delta", "robust_scale_floor", "robust_scale_decay", "tail_safe_aggregation", "dual_scale_gradient", "bounded_tail_gradient", "prediction_gated_tail_gradient", "target_gated_tail_gradient", "service_raw_residual_calibration", "service_residual_reliability_shrinkage", "service_residual_variance_aware_shrinkage", "prediction_conditioned_service_calibration", "log_domain_service_calibration", "dual_constrained_log_calibration", "dynamic_online_routing", "conformal_mode", "residual_eta", "residual_l2", "residual_decay", "residual_kappa", "seed", "max_time_slices")}
    config["qos_contract"] = {
        "name": spec["qos_name"], "unit": spec["qos_unit"],
        "validity_upper_exclusive": spec["valid_upper"],
        "fixed_log1p_upper_bound": spec["log_upper"],
        "value_bins": list(spec["value_bins"]),
    }
    if args.causal_support_output_trust_update:
        config["csotu_constants"] = {
            "candidate_step": "AMF-reliability-weighted block Gauss-Newton",
            "trust_budget": "median support absolute residual times sqrt(n/(n+1))",
            "trust_metric": "actual support-prediction RMS change",
            "step_selection": "deterministic halving backtracking",
            "new_tunable_hyperparameters": "none",
        }
    if args.output_trust_bounded_compression:
        config["otbc_constants"] = {
            "quantizer": "deterministic per-service-row symmetric signed INT16 delta",
            "fallback": "FP32 vector-and-bias delta",
            "trust_bound": "||p_u||_2||epsilon_v||_2+|epsilon_b| <= max(tau_u-D_u,0)",
            "error_feedback": "next delta is exact server target minus decoded client cache",
            "service_error": "FP32",
            "trust_certificate": "two FP32 scalars per participating client",
            "query_or_future_use": "none",
        }
    if args.support_conditioned_stability_plasticity_update:
        config["scspu_constants"] = {
            "drift_signal": "median abs(r-h)/(r+h+eps) over the pre-update current support",
            "relative_error": "abs(pre_update_prediction-target)/target",
            "history_error": "(pre-update user AMF error + service AMF error)/2",
            "plasticity_factor": "rho=1/(2-d), bounded in [0.5,1]",
            "projection": "pre-update parameter anchor plus rho times the original COQMF proposal displacement",
            "projected_parameters": "user/service vectors and biases only",
            "error_state": "original COQMF proposal retained without projection",
            "new_tunable_hyperparameters": "none",
            "query_or_future_use": "none",
            "communication_increment": {"uplink_bytes": 0, "downlink_bytes": 0},
        }
    if args.support_novelty_subspace_update:
        config["support_novelty_subspace_constants"] = {
            "history_state": "client-private per-user vector of dimension d+1",
            "history_update": "accumulate only the accepted support-update displacement after rotation",
            "first_update": "zero history accepts the original COQMF proposal exactly",
            "decomposition": "delta_parallel=<delta,h>h/(||h||^2+eps); delta_novel=delta-delta_parallel",
            "novelty_ratio": "nu=||delta_novel||/(||delta||+eps), clipped only for roundoff to [0,1]",
            "rotation": "z=delta_parallel+(1+nu)delta_novel; delta_star=||delta||z/||z||",
            "norm_constraint": "accepted joint user-vector/user-bias displacement preserves original COQMF norm",
            "projected_parameters": "user vector and user bias only",
            "unchanged_parameters": "service vectors, service biases, user AMF error and service AMF errors",
            "query_or_future_use": "none",
            "communication_increment": {"uplink_bytes": 0, "downlink_bytes": 0},
            "new_tunable_hyperparameters": "none",
        }
    if args.client_fast_calibration:
        config["client_calibration_ridge"] = CLIENT_CALIBRATION_RIDGE
    if args.client_monotone_calibration:
        config["cpmc_constants"] = {"ridge": CPMC_RIDGE, "degree": 2, "projection": "equal-weight PAVA"}
    if args.client_logit_calibration:
        config["cplc_constants"] = {"ridge": CPLC_RIDGE, "eps": CPLC_EPS,
                                    "gain_bounds": list(CPLC_GAIN_BOUNDS), "offset_bounds": list(CPLC_OFFSET_BOUNDS)}
    if args.dtnf_bridge:
        config["dtnf_constants"] = {
            "fast_rate": DTNF_FAST_RATE, "slow_rate": DTNF_SLOW_RATE,
            "slow_kappa": DTNF_SLOW_KAPPA, "gate_ridge": DTNF_GATE_RIDGE,
        }
    if args.causal_time_factor:
        config["cttf_constants"] = {
            "ridge": CTTF_RIDGE, "target_eps": CTTF_TARGET_EPS,
            "delta_bound": CTTF_DELTA_BOUND, "objective_reduction": "mean",
        }
    if args.state_space_mf:
        config["state_space_constants"] = {
            "gate_ridge": STATE_SPACE_GATE_RIDGE,
            "gate_bound": STATE_SPACE_GATE_BOUND,
            "gate_dimension": STATE_SPACE_GATE_DIMENSION,
        }
    if args.graph_residual_backbone:
        config["graph_residual_constants"] = {
            "gate_ridge": GRAPH_GATE_RIDGE,
            "gate_bound": GRAPH_GATE_BOUND,
        }
    stca_method = {"full": "Fed-AMF-STCA", "consensus": "Fed-AMF-STCA-consensus-only",
                   "history": "Fed-AMF-STCA-history-only",
                   "evidence": "Fed-AMF-STCA-evidence-gated",
                   "adaptive_evidence": "Fed-AMF-DA-HEGSA",
                   "bidirectional_evidence": "Fed-AMF-DUA-HEGSA",
                   "selective_dual_memory": "Fed-AMF-SDM-HEGSA",
                   "persistent": "Fed-AMF-PDN-STCA"}[args.stca_component]
    stca_description = {
        "full": "per-service coordinate-median consensus coherence times age-attenuated historical update coherence",
        "consensus": "diagnostic per-service coordinate-median current-consensus coherence only",
        "history": "diagnostic per-service age-attenuated historical update coherence only",
        "evidence": "per-service historical-evidence entropy gate times current-consensus coherence",
        "adaptive_evidence": "service-local drift-adaptive transfer from historical coherence to current-consensus coherence",
        "bidirectional_evidence": "service-local drift-uncertainty split between coherent current reaction and uncertain-change history protection",
        "selective_dual_memory": "per-service Pareto routing between accepted-update stable memory and proposal-median reactive memory",
        "persistent": "STCA-v2 weights continuously neutralized toward uniform by client persistent support drift",
    }[args.stca_component]
    resolved = {
        "method": (("Fed-RSRRC-AMF-STCA-v2" if args.service_residual_reliability_shrinkage else
                    ("Fed-SRRC-AMF-STCA-v2" if args.service_raw_residual_calibration else
                    ("Fed-TGBLR-AMF-STCA-v2" if args.target_gated_tail_gradient else
                    ("Fed-PGBLR-AMF-STCA-v2" if args.prediction_gated_tail_gradient else
                    ("Fed-BLR-AMF-STCA-v2" if args.bounded_tail_gradient else
                    ("Fed-DSG-AMF-STCA-v2" if args.dual_scale_gradient else
                    ("Fed-AMF-STCA-v2-SRG" if args.tail_safe_aggregation else
                    ("Fed-AMF-CTTF-STCA-v2-CTAC-CPOC" if args.conformal_mode == "private" else
                    ("Fed-AMF-CTTF-STCA-v2-CTAC-GOC" if args.conformal_mode == "global" else
                    ("Fed-AMF-CTTF-STCA-v2-CPLC" if args.client_logit_calibration else
                    ("Fed-AMF-CTTF-STCA-v2-CPMC" if args.client_monotone_calibration else
                    ("Fed-AMF-CTTF-DOR-STCA-v2-CTAC" if args.dynamic_online_routing else
                    ("Fed-AMF-CTTF-STCA-v2-CTAC" if args.causal_time_factor and args.client_fast_calibration else
                    ("Fed-AMF-CTTF-STCA-v2" if args.causal_time_factor else
                    ("Fed-AMF-DTNF-STCA-v2-CTAC" if args.dtnf_bridge and args.client_fast_calibration else
                    ("Fed-AMF-DTNF-STCA-v2" if args.dtnf_bridge else
                    ("Fed-AMF-STCA-v2-CTAC" if args.client_fast_calibration else stca_method))
                    )))))))))))))))
                   if args.aggregation == "stca" else
                   ("Fed-AMF-private-service-residual" if args.private_service_residual else
                    ("FedProx-AMF-minimal" if args.fedprox_mu else "Fed-AMF-minimal"))),
        "qos_type": spec["qos_type"], "qos_name": spec["qos_name"], "qos_unit": spec["qos_unit"],
        "qos_contract": config["qos_contract"],
        "config": config, "mask_file": str(args.mask_file.resolve()),
        "mask_manifest": str(args.manifest.resolve()), "data_file": str(args.data_file.resolve()), "data_sha256": sha256,
        "common_state_digest": getattr(args, "common_state_digest", None),
        "common_sample_keys": common_sample_keys,
        "protocol": manifest["task"], "preprocessing": manifest["preprocessing"],
        "aggregation": ((stca_description + "; current-support original-scale SSE safety selection against matched FedAvg")
                        if args.tail_safe_aggregation else
                        (stca_description if args.aggregation == "stca" else "per-service count-weighted average of local AMF updates")),
        "dual_scale_gradient": ({
            "objectives": "equal mean-magnitude balance of normalized and fixed-inverse-log original-scale support gradients",
            "normalizer": "client-local mean absolute gradient; recomputed at each local epoch",
            "query_or_future_use": "none",
        } if args.dual_scale_gradient else "disabled"),
        "bounded_tail_gradient": ({
            "risk": "log1p(raw support squared error) / log1p(10000^2)",
            "gradient_multiplier": "1 + risk, fixed in [1,2]",
            "query_or_future_use": "none",
        } if args.bounded_tail_gradient else "disabled"),
        "prediction_gated_tail_gradient": ({
            "risk": "log1p(raw support squared error) / log1p(10000^2)",
            "gradient_multiplier": "1 + normalized_prediction * risk, fixed in [1,2]",
            "query_or_future_use": "none",
        } if args.prediction_gated_tail_gradient else "disabled"),
        "target_gated_tail_gradient": ({
            "risk": "log1p(raw support squared error) / log1p(10000^2)",
            "gradient_multiplier": "1 + normalized_support_target * risk, fixed in [1,2]",
            "query_or_future_use": "none",
        } if args.target_gated_tail_gradient else "disabled"),
        "service_raw_residual_calibration": ({
            "fit": ("empirical-Bayes current-support log-domain residual per service after aggregation"
                    if args.log_domain_service_calibration else
                    "mean current-support raw residual per service after aggregation"),
            "prediction": (("log-domain correction followed by the frozen inverse transform; "
                            if args.log_domain_service_calibration else
                            ("alpha * service mean residual, alpha=c^2/(c^2+v/n); "
                            if args.service_residual_variance_aware_shrinkage else
                            ("alpha * service mean residual, alpha=R^2/(R^2+Q); "
                             if args.service_residual_reliability_shrinkage else ""))) +
                           f"clip(raw prediction + service correction, 0, {spec['log_upper']})"),
            "qos_scope": "shared RT/TP formula with the frozen attribute-specific upper bound",
            "query_or_future_use": "none",
        } if args.service_raw_residual_calibration else "disabled"),
        "service_residual_reliability_shrinkage": ({
            "statistics": "current-support raw residual sum R, squared sum Q and count n per service",
            "reliability": "R^2 / (R^2 + Q), or zero when the denominator is zero",
            "correction": "reliability * R / n",
            "parameters": "none; identical formula for RT and TP",
            "communication_increment_vs_srrc": {"uplink_bytes": 0, "downlink_bytes": 0},
            "query_or_future_use": "none",
        } if args.service_residual_reliability_shrinkage else "disabled"),
        "service_residual_variance_aware_shrinkage": ({
            "statistics": "current-support raw residual mean c, population variance v and count n per service",
            "reliability": "c^2/(c^2+v/n) for n>=2; one-half for a nonzero singleton; zero for no support or exact zero",
            "correction": "reliability * c",
            "parameters": "none; identical formula for RT and TP",
            "communication_increment_vs_srrc": {"uplink_bytes": 0, "downlink_bytes": 0},
            "query_or_future_use": "none",
        } if args.service_residual_variance_aware_shrinkage else "disabled"),
        "prediction_conditioned_service_calibration": ({
            "fit": "support-only normalized residual = service intercept + one pooled within-service prediction slope",
            "support_uncertainty": "empirical-Bayes service intercept shrinkage toward the current global residual level",
            "uplink_payload": "float32 normalized prediction and residual per support entry",
            "downlink_payload": "float32 service intercept vector plus one float32 shared slope per participant",
            "fallback": "untouched services receive exactly zero correction",
            "parameters": "no tuned threshold; identical formula for RT and TP",
            "query_or_future_use": "none",
        } if args.prediction_conditioned_service_calibration else "disabled"),
        "log_domain_service_calibration": ({
            "fit": "empirical-Bayes shrinkage of float32 current-support service residuals in the frozen normalized log1p domain",
            "prediction": "add service correction in normalized log1p space, clip to [0,1], then apply the frozen inverse transform",
            "support_uncertainty": "pooled within-service variance versus between-service variance; singleton fallback shrinks toward the current global residual",
            "uplink_payload": "one float32 normalized log-domain residual per support entry",
            "downlink_payload": "one float32 normalized log-domain correction per service and participant",
            "fallback": "untouched services and exact zero corrections are exact no-ops",
            "parameters": "no tuned threshold; identical formula for RT and TP",
            "query_or_future_use": "none",
        } if args.log_domain_service_calibration else "disabled"),
        "dual_constrained_log_calibration": ({
            "prediction": "compute normalized log-domain and normalized raw-domain candidates, then retain the smaller absolute raw displacement",
            "tail_guard": "absolute raw movement is bounded by raw_upper_bound times the absolute float32 service correction",
            "parameters": "none; identical formula for RT and TP",
            "communication_increment_vs_lcsrrc": {"uplink_bytes": 0, "downlink_bytes": 0},
            "query_usage": "query prediction only; no query or future target",
        } if args.dual_constrained_log_calibration else "disabled"),
        "stca_component": args.stca_component if args.aggregation == "stca" else "disabled",
        "drift_adaptive_hegsa": ({
            "drift_signal": "bounded normalized change between the current per-service coordinate-median proposal delta and the previous accepted service delta",
            "history_retention": "raise age-attenuated historical reliability to one minus the service drift intensity",
            "current_evidence_transfer": "interpolate the historical-information gate toward one by the same service drift intensity",
            "state": "server-side previous accepted service delta and update age only",
            "parameters": "none; identical formula for RT and TP",
            "communication_increment": {"uplink_bytes": 0, "downlink_bytes": 0},
            "query_usage": "none; current proposals and prior server state only",
        } if args.stca_component == "adaptive_evidence" else "disabled"),
        "drift_uncertainty_adaptive_hegsa": ({
            "drift_signal": "bounded normalized change between current coordinate-median service delta and previous accepted service delta",
            "proposal_uncertainty": "bounded relative dispersion of current client service proposals",
            "three_way_gate": "weights one-minus-drift on the original HEGSA gate, drift-times-one-minus-uncertainty on current reaction, and drift-times-uncertainty on history protection",
            "history_retention": "historical reliability is relaxed only by the coherent reactive-transfer mass",
            "state": "server-side current proposals, previous accepted service delta, and update age only",
            "parameters": "none; identical formula for RT and TP",
            "communication_increment": {"uplink_bytes": 0, "downlink_bytes": 0},
            "query_usage": "none; query is diagnostic/evaluation only",
        } if args.stca_component == "bidirectional_evidence" else "disabled"),
        "selective_dual_memory_hegsa": ({
            "stable_memory": "previous accepted per-service aggregate update and server-side age",
            "reactive_memory": "previous per-service coordinate-median client proposal update and server-side age",
            "experts": "the unchanged HEGSA evidence formula applied independently to stable and reactive memory",
            "routing": "select reactive only when its support-derived candidate-error risk and weighted proposal dispersion Pareto-dominate the stable expert; stable wins ties",
            "parameters": "no tuned threshold; only machine-precision comparison tolerance; identical formula for RT and TP",
            "state": "one extra service update vector and timestamp at the server",
            "communication_increment": {"uplink_bytes": 0, "downlink_bytes": 0},
            "query_usage": "none; current support-derived proposals and prior server states only",
        } if args.stca_component == "selective_dual_memory" else "disabled"),
        "value_bin_diagnostic": ({"selection": f"true query {spec['qos_name']} after prediction; never used for update or model selection",
                                  "intervals": list(spec["value_bins"])} if args.diagnostic_value_bins else "disabled"),
        "service_aggregation_diagnostic": ("support-only server aggregation features joined after prediction with per-service query errors; diagnostic only"
                                           if args.diagnostic_service_aggregation else "disabled"),
        "federated_optimizer": ("STCA on touched service rows" if args.aggregation == "stca" else ("FedProx on shared service vectors and biases" if args.fedprox_mu else "FedAvg-style fixed aggregation")),
        "aggregation_privacy": ("server additionally receives support-derived CTTF symmetric Gram/RHS sufficient statistics; no query labels or private user vectors are uploaded, but support information may leak and no formal privacy is claimed"
                                if args.causal_time_factor else
                                ("server additionally receives support-derived slow sum/count rows and gate sufficient statistics; no query labels or private user vectors are uploaded, but singleton slow rows may leak support values and no formal privacy is claimed"
                                if args.dtnf_bridge else
                                ("server additionally receives one support-derived drift scalar per participating client; raw QoS remains local"
                                if args.stca_component == "persistent" else
                                ("support clients receive matched FedAvg/STCA candidate service rows and return two raw-scale squared-loss scalars per touched service; query/future labels are excluded"
                                 if args.tail_safe_aggregation else
                                 "server receives the unchanged shared service proposal payload; STCA uses proposals and server-side service history only")))),
        "privacy_boundary": ("query/future labels, raw QoS and private user states remain local; support-derived Gram/RHS are uploaded and may leak support information; no DP or secure aggregation is claimed"
                             if args.causal_time_factor else
                             ("query/future labels and raw client states remain local; support-derived normalized sum/count and gate Gram/RHS are uploaded and may reveal support values; no DP or secure aggregation is claimed"
                             if args.dtnf_bridge else
                             ("raw support QoS and user states remain client-side inputs to the two-candidate loss evaluation; only two loss scalars are returned; no DP or secure aggregation is claimed"
                              if args.tail_safe_aggregation else
                              "raw QoS, U_i, p_i, and user error state remain client-private"))),
        "private_residual": "client-private per-user-service scalar residual; updated only from support labels and never uploaded" if args.private_service_residual else "disabled",
        "client_fast_calibration": ({
            "form": "private client-wide affine calibration of normalized predictions",
            "update": ("post-CTTF post-aggregation current-support ridge update toward previous private state"
                       if args.causal_time_factor else
                       ("post-DTNF post-aggregation current-support ridge update toward previous private state"
                       if args.dtnf_bridge else
                       "post-aggregation current-support ridge update toward previous private state")),
            "fit_input": ("support-only normalized AMF+CTTF predictions" if args.causal_time_factor else
                          ("support-only normalized AMF+DTNF predictions" if args.dtnf_bridge else
                          "support-only normalized AMF predictions")),
            "query_order": ("AMF -> CTTF -> sigmoid -> private CTAC" if args.causal_time_factor else
                            ("AMF -> DTNF -> sigmoid -> private CTAC" if args.dtnf_bridge else
                            "AMF -> sigmoid -> private CTAC")),
            "ridge": CLIENT_CALIBRATION_RIDGE,
            "upload": "none",
        } if args.client_fast_calibration else "disabled"),
        "client_monotone_calibration": ({
            "form": "client-private degree-2 Bernstein calibration with nondecreasing bounded controls",
            "update": "post-CTTF current-support ridge toward previous private controls, then equal-weight PAVA",
            "fit_input": "support-only normalized AMF+CTTF predictions",
            "query_order": "AMF -> CTTF -> sigmoid -> private CPMC -> inverse transform",
            "ridge": CPMC_RIDGE,
            "identity_controls": CPMC_IDENTITY.tolist(),
            "upload": "none",
        } if args.client_monotone_calibration else "disabled"),
        "client_logit_calibration": ({
            "form": "client-private bounded affine calibration in logit space",
            "update": "post-CTTF current-support logit ridge toward previous private gain/offset",
            "query_order": "AMF -> CTTF -> sigmoid -> private CPLC -> inverse transform",
            "ridge": CPLC_RIDGE, "eps": CPLC_EPS,
            "gain_bounds": list(CPLC_GAIN_BOUNDS), "offset_bounds": list(CPLC_OFFSET_BOUNDS),
            "upload": "none",
        } if args.client_logit_calibration else "disabled"),
        "dtnf_state_boundary": ({
            "client_private": "one fast vector per client; support-only; never uploaded",
            "shared": "two slow evidence scalars per service and two bounded global gate parameters",
            "slow_upload": "per-touched-service normalized-target sum and count; explicitly included in communication",
            "gate_upload": "one 2x2 Gram matrix and one 2-vector per participating client; support-only",
            "query_usage": "prediction and post-prediction evaluation only; no state update",
        } if args.dtnf_bridge else "disabled"),
        "state_space_boundary": ({
            "client_private": "one level vector and one velocity vector per client; support-only; never uploaded",
            "shared": "touched-service level/velocity vectors and four bounded global gates",
            "uplink": "per-client touched-service count/innovation sufficient statistics plus a 4x4 symmetric Gram, RHS and count",
            "downlink": "touched-service level/velocity rows and four global gates",
            "query_usage": "read-only state-space logit correction; query/future labels never update state",
            "privacy_note": "support-derived sufficient statistics may leak support information; no DP or secure aggregation is claimed",
        } if args.state_space_mf else "disabled"),
        "graph_residual_boundary": ({
            "client_private": "one residual and confidence scalar per client; support-only; never uploaded",
            "shared": "one residual and confidence scalar per touched service plus one bounded global gate",
            "uplink": "per-client touched-service count/residual-sum rows plus scalar gate Gram/RHS/count",
            "downlink": "touched-service residual/confidence rows and one global gate",
            "query_usage": "read-only graph residual logit correction; query/future labels never update state",
            "privacy_note": "support-derived sufficient statistics may leak support information; no DP or secure aggregation is claimed",
        } if args.graph_residual_backbone else "disabled"),
        "robust_amf_boundary": ({
            "loss": "pseudo-Huber derivative on normalized support residuals",
            "scale": "one support-only adaptive MAD/EMA scalar per client",
            "client_private": "robust scale is never uploaded",
            "communication_increment": {"uplink_bytes": 0, "downlink_bytes": 0},
            "query_usage": "query/future labels never update loss scale or model state",
        } if args.robust_amf else "disabled"),
        "cttf_state_boundary": ({
            "client_private": "user vector, bias and error state; raw support/query labels are not uploaded",
            "shared": "one d-dimensional causal time factor with a previous-factor ridge prior",
            "uplink": "upper triangle of support-only X^T X and support-only X^T residual per participating client",
            "downlink": "one d-dimensional time factor per participating client after the support statistics update",
            "query_usage": ("AMF -> CTTF -> sigmoid -> private CPLC; query/future labels never update state"
                            if args.client_logit_calibration else
                            ("AMF -> CTTF -> sigmoid -> private CPMC; query/future labels never update state"
                            if args.client_monotone_calibration else
                            ("AMF -> CTTF -> sigmoid -> private CTAC; query/future labels never update state"
                            if args.client_fast_calibration else
                            "AMF -> CTTF logit correction -> prediction; query/future labels never update state"),
                            ))
        } if args.causal_time_factor else "disabled"),
        "dynamic_online_routing": ({
            "experts": ["post-STCA AMF normalized prediction", "post-STCA AMF+CTTF normalized prediction"],
            "update": "client-private current-support mean squared losses with parameter-free cumulative Hedge",
            "learning_rate": "sqrt(2*log(2)/max(1,support-bearing rounds))",
            "query_order": "AMF/CTTF -> sigmoid -> private DOR -> private CTAC -> inverse transform",
            "upload": "none",
            "query_usage": "read-only prediction and post-prediction evaluation; query/future labels never update routing",
        } if args.dynamic_online_routing else "disabled"),
        "online_conformal": ({
            "mode": args.conformal_mode, "target_coverage": 1.0 - CONFORMAL_ALPHA,
            "radius": "higher empirical quantile of pre-current-CTAC support absolute errors",
            "state": ("client-private histories and radii; no upload" if args.conformal_mode == "private" else
                      "pooled support error history at server; baseline only"),
            "query_usage": "interval construction and post-prediction evaluation only",
        } if args.conformal_mode != "off" else "disabled"),
        "causal_drift_audit": "32-bin JS(current support, prior support history); min 20 entries; per-time eligible-client quartiles",
        "query_read_only_audit": bool(getattr(args, "read_only_audit_result", {}).get("query_read_only", False)),
        "future_query_read_only_audit": bool(getattr(args, "read_only_audit_result", {}).get("future_read_only", False)),
        "causal_read_only_audit": getattr(args, "read_only_audit_result", {"checked": False}),
    }
    if args.service_residual_variance_aware_shrinkage:
        resolved["method"] = "Fed-VSRRC-AMF-STCA-v2"
    if args.prediction_conditioned_service_calibration:
        resolved["method"] = "Fed-PCSRRC-AMF-STCA-v2"
    if args.log_domain_service_calibration:
        resolved["method"] = "Fed-LCSRRC-AMF-STCA-v2"
    if args.dual_constrained_log_calibration:
        resolved["method"] = ({
            "adaptive_evidence": "Fed-DCLCSRRC-AMF-DA-HEGSA",
            "bidirectional_evidence": "Fed-DCLCSRRC-AMF-DUA-HEGSA",
            "selective_dual_memory": "Fed-DCLCSRRC-AMF-SDM-HEGSA",
        }.get(args.stca_component, "Fed-DCLCSRRC-AMF-STCA-v2"))
    if args.state_space_mf:
        if args.conformal_mode == "private":
            resolved["method"] = "Fed-SSMF-CTTF-STCA-v2-CTAC-CPOC"
        elif args.conformal_mode == "global":
            resolved["method"] = "Fed-SSMF-CTTF-STCA-v2-CTAC-GOC"
        elif args.causal_time_factor and args.client_fast_calibration:
            resolved["method"] = "Fed-SSMF-CTTF-STCA-v2-CTAC"
        elif args.causal_time_factor:
            resolved["method"] = "Fed-SSMF-CTTF-STCA-v2"
        elif args.client_fast_calibration:
            resolved["method"] = "Fed-SSMF-STCA-v2-CTAC"
        else:
            resolved["method"] = "Fed-SSMF-STCA-v2"
    if args.graph_residual_backbone:
        if args.conformal_mode == "private":
            resolved["method"] = "Fed-GCR-AMF-CTTF-STCA-v2-CTAC-CPOC"
        elif args.conformal_mode == "global":
            resolved["method"] = "Fed-GCR-AMF-CTTF-STCA-v2-CTAC-GOC"
        elif args.causal_time_factor and args.client_fast_calibration:
            resolved["method"] = "Fed-GCR-AMF-CTTF-STCA-v2-CTAC"
        elif args.causal_time_factor:
            resolved["method"] = "Fed-GCR-AMF-CTTF-STCA-v2"
        elif args.client_fast_calibration:
            resolved["method"] = "Fed-GCR-AMF-STCA-v2-CTAC"
        else:
            resolved["method"] = "Fed-GCR-AMF-STCA-v2"
    if args.robust_amf:
        if args.conformal_mode == "private":
            resolved["method"] = "Fed-Robust-AMF-CTTF-STCA-v2-CTAC-CPOC"
        elif args.conformal_mode == "global":
            resolved["method"] = "Fed-Robust-AMF-CTTF-STCA-v2-CTAC-GOC"
        elif args.causal_time_factor and args.client_fast_calibration:
            resolved["method"] = "Fed-Robust-AMF-CTTF-STCA-v2-CTAC"
        elif args.causal_time_factor:
            resolved["method"] = "Fed-Robust-AMF-CTTF-STCA-v2"
        elif args.client_fast_calibration:
            resolved["method"] = "Fed-Robust-AMF-STCA-v2-CTAC"
        elif args.aggregation == "stca":
            resolved["method"] = "Fed-Robust-AMF-STCA-v2"
        else:
            resolved["method"] = "Fed-Robust-AMF-minimal"
    if args.private_service_residual and args.aggregation == "stca":
        if args.conformal_mode == "private":
            resolved["method"] = "Fed-AMF-PSR-CTTF-STCA-v2-CTAC-CPOC"
        elif args.conformal_mode == "global":
            resolved["method"] = "Fed-AMF-PSR-CTTF-STCA-v2-CTAC-GOC"
        elif args.causal_time_factor and args.client_fast_calibration:
            resolved["method"] = "Fed-AMF-PSR-CTTF-STCA-v2-CTAC"
        elif args.causal_time_factor:
            resolved["method"] = "Fed-AMF-PSR-CTTF-STCA-v2"
        elif args.client_fast_calibration:
            resolved["method"] = "Fed-AMF-PSR-STCA-v2-CTAC"
        else:
            resolved["method"] = "Fed-AMF-PSR-STCA-v2"
    resolved["essd"] = ({
        "name": "ESSD",
        "full_name": "Event-triggered Sparse Service Downlink",
        "parent_aggregator": "FedSRA",
        "trigger": "current support-touched service update event (changed = counts > 0)",
        "payload": "per-client fixed bitmap plus changed service rows; DCLCSRRC correction shares the same bitmap",
        "selection": "per-slice min(full shared service block, bitmap plus changed rows)",
        "uplink": "unchanged; all client proposals and DCLCSRRC support residual payloads are preserved",
        "causal_boundary": "current support/update state only; query and future labels are never used",
        "privacy_boundary": "bitmap reveals the touched-service pattern; no raw query labels, DP, or secure aggregation claim",
        "numerical_invariance": "server aggregation, model state, and prediction path are unchanged",
    } if args.event_sparse_service_downlink else "disabled")
    resolved["communication_semantics"] = (
        "point-to-point per-client full-or-event-sparse service downlink; sparse mode uses a fixed little-endian bitmap and changed rows"
        if args.event_sparse_service_downlink else
        "point-to-point full shared service-state downlink; fixed-layout row indices implicit"
    )
    resolved["communication_parameter_bytes"] = args.communication_parameter_bytes
    resolved["communication_compression"] = ({
        "name": "OTBC",
        "full_name": "Output-Trust-Bounded Compression",
        "server_state": "exact uncompressed aggregation state",
        "client_state": "decoded per-client service cache used by current query prediction and next local update",
        "compressed_block": "cache-relative service-vector and service-bias delta, signed INT16 with one FP32 scale and one flag per row",
        "fallback": "FP32 vector-and-bias delta",
        "uncompressed_fields": "FP32 AMF service error and FP32 DCLCSRRC correction",
        "trust_certificate": "FP32 support-only online-update remaining slack and FP32 private-user norm; 8 uplink bytes per participant",
        "trust_test": "||p_u||_2||epsilon_v||_2+|epsilon_b| <= max(tau_u-D_u,0)",
        "error_feedback": "target-minus-decoded-cache delta reinjects the complete prior residual",
        "causal_boundary": "current support, private client state, and current server model only",
        "query_or_future_use": "none",
    } if args.output_trust_bounded_compression else "disabled")
    resolved["online_update"] = ({
        "name": "CSOTU",
        "full_name": "Causal Support-Set Output-Trust Update",
        "parent_predictor": "COQMF",
        "support_unit": "current client support set",
        "curvature": "AMF-reliability-weighted block Gauss-Newton",
        "trust_region": "actual support-prediction RMS change under a support-only budget",
        "error_state": "same-snapshot batch update; support-order invariant",
        "query_or_future_use": "none",
        "extra_uplink_bytes": 0,
        "extra_downlink_bytes": 0,
    } if args.causal_support_output_trust_update else ({
        "name": "SCSPU",
        "full_name": "Support-Conditioned Stability-Plasticity Update",
        "parent_predictor": "COQMF",
        "support_unit": "current client support set",
        "drift_signal": "median abs(r-h)/(r+h+eps), computed before the local update",
        "plasticity_factor": "rho=1/(2-d), bounded in [0.5,1]",
        "projection": "pre-update user/service vectors and biases plus rho times the original COQMF proposal displacement",
        "error_state": "original COQMF error-state proposal retained without projection",
        "query_or_future_use": "none",
        "extra_uplink_bytes": 0,
        "extra_downlink_bytes": 0,
    } if args.support_conditioned_stability_plasticity_update else ({
        "name": "SNSU",
        "full_name": "Support-Novelty Subspace Update",
        "parent_predictor": "COQMF",
        "support_unit": "current client support set",
        "history_state": "client-private per-user direction of dimension d+1",
        "history_update": "accumulate only previously accepted support-update displacements",
        "first_update": "zero history accepts the original COQMF displacement exactly",
        "decomposition": "delta_parallel=<delta,h>h/(||h||^2+eps); delta_novel=delta-delta_parallel",
        "novelty_ratio": "nu=||delta_novel||/(||delta||+eps), bounded in [0,1]",
        "rotation": "z=delta_parallel+(1+nu)delta_novel; delta_star=||delta||z/||z||",
        "norm_constraint": "joint user-vector/user-bias displacement norm is preserved",
        "projected_parameters": "user vector and user bias only",
        "unchanged_parameters": "service proposals and user/service AMF error states",
        "query_or_future_use": "none",
        "extra_uplink_bytes": 0,
        "extra_downlink_bytes": 0,
    } if args.support_novelty_subspace_update else "original per-sample COQMF SGD")))
    if args.event_sparse_service_downlink:
        resolved["method"] = "COQMF + FedSRA"
        resolved["aggregator"] = "FedSRA"
        resolved["fed_sra_components"] = ["HEGSA", "DCLCSRRC", "ESSD"]
    if args.causal_support_output_trust_update:
        resolved["method"] = "FedOSR-CSOTU"
        resolved["prediction_model"] = "COQMF-CSOTU"
    if args.output_trust_bounded_compression:
        resolved["method"] = "FedOSR-CSOTU-OTBC"
        resolved["prediction_model"] = "COQMF-CSOTU"
        resolved["aggregator"] = "FedSRA-OTBC"
    if args.support_conditioned_stability_plasticity_update:
        resolved["method"] = ("FedOSR-SCSPU-OTBC" if args.output_trust_bounded_compression
                              else "FedOSR-SCSPU")
        resolved["prediction_model"] = "COQMF-SCSPU"
    if args.support_novelty_subspace_update:
        resolved["method"] = ("FedOSR-SNSU-OTBC" if args.output_trust_bounded_compression
                              else "FedOSR-SNSU")
        resolved["prediction_model"] = "COQMF-SNSU"
    if args.aggregation == "local":
        # The local control is intentionally explicit in the artifact rather
        # than being represented as a degenerate federated aggregator.  This
        # makes the no-upload/no-download boundary auditable in RQ2 outputs.
        resolved.update({
            "method": "Local C-OMF",
            "prediction_model": "C-OMF",
            "aggregator": "none (client-local service state)",
            "fed_sra_components": [],
            "aggregation": "each client updates its private service state from its own current support; no server aggregation",
            "federated_optimizer": "none; client-private online AMF update only",
            "aggregation_privacy": "no support records, model proposals, or state are uploaded to a server",
            "privacy_boundary": "user state and service state remain client-private; query and future labels remain local",
            "communication_semantics": "none; no client-to-server upload and no server-to-client downlink",
            "communication_compression": "disabled",
            "essd": "disabled",
            "local_state": {
                "user_state": "one private user vector, bias, and error state per client",
                "service_state": "one private service-vector, bias, and error table per client",
                "initialization": "all client tables copy the common initialization once",
                "persistence": "each client's service table persists across time slices",
                "support_update": "only the owning client's current support rows are updated",
                "query_update": "query and future reads are state-free",
                "server_interaction": "none",
            },
        })
    if _is_frozen_fedavg_residual_control(args):
        # ESSD is only the representation of the server-to-client payload in
        # this control; it must not make a FedAvg run look like FedSRA in the
        # resolved artifact.  Keep this override scoped to the new validated
        # control so historical configurations retain their existing metadata.
        resolved.update({
            "method": "C-OMF + FedAvg + Residual Calibration",
            "prediction_model": "COQMF-CSOTU",
            "aggregator": "FedAvg",
            "fed_sra_components": [],
        })
        if isinstance(resolved.get("essd"), dict):
            resolved["essd"]["parent_aggregator"] = "FedAvg"
    (args.run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    (args.run_dir / "summary.json").write_text(json.dumps({"time_mean": summary, "elapsed_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
