"""Current-support service calibration in the original QoS scale."""

from __future__ import annotations

import numpy as np


def mean_service_residual(residual_sums, counts):
    residual_sums = np.asarray(residual_sums, dtype=float)
    counts = np.asarray(counts, dtype=int)
    if residual_sums.shape != counts.shape or np.any(counts < 0):
        raise ValueError("service residual sums and nonnegative counts must align")
    if not np.all(np.isfinite(residual_sums)):
        raise ValueError("service residual sums must be finite")
    result = np.zeros_like(residual_sums)
    touched = counts > 0
    result[touched] = residual_sums[touched] / counts[touched]
    return result


def reliable_service_residual(residual_sums, squared_residual_sums, counts):
    """Shrink a service correction when its current-support evidence disagrees.

    For a touched service, let R be the sum of raw residuals, Q the sum of
    squared raw residuals and n the support count.  The parameter-free
    reliability is R**2 / (R**2 + Q), and the returned correction is
    reliability * R / n.  The ratio is invariant to the raw QoS unit, so the
    same rule is used for RT and TP.  Untouched or exactly fitted services are
    exact no-ops.
    """
    residual_sums = np.asarray(residual_sums, dtype=float)
    squared_residual_sums = np.asarray(squared_residual_sums, dtype=float)
    counts = np.asarray(counts, dtype=int)
    if (residual_sums.shape != squared_residual_sums.shape or
            residual_sums.shape != counts.shape or np.any(counts < 0)):
        raise ValueError("service residual statistics and nonnegative counts must align")
    if (not np.all(np.isfinite(residual_sums)) or
            not np.all(np.isfinite(squared_residual_sums)) or
            np.any(squared_residual_sums < 0)):
        raise ValueError("service residual statistics must be finite and squared sums nonnegative")

    reliability = np.zeros_like(residual_sums)
    correction = np.zeros_like(residual_sums)
    touched = counts > 0
    signal = residual_sums[touched] ** 2
    denominator = signal + squared_residual_sums[touched]
    valid = denominator > 0
    touched_reliability = np.zeros_like(signal)
    touched_reliability[valid] = signal[valid] / denominator[valid]
    reliability[touched] = touched_reliability
    correction[touched] = touched_reliability * residual_sums[touched] / counts[touched]
    return correction, reliability


def variance_aware_service_residual(residual_sums, squared_residual_sums, counts):
    """Shrink SRRC by the estimated standard error of support residuals.

    For n >= 2, reliability is mean**2 / (mean**2 + variance / n),
    using population variance.  Perfectly agreeing support therefore recovers
    ordinary SRRC.  A nonzero singleton is fixed at one-half reliability.
    """
    residual_sums = np.asarray(residual_sums, dtype=float)
    squared_residual_sums = np.asarray(squared_residual_sums, dtype=float)
    counts = np.asarray(counts, dtype=int)
    if (residual_sums.shape != squared_residual_sums.shape or
            residual_sums.shape != counts.shape or np.any(counts < 0)):
        raise ValueError("service residual statistics and nonnegative counts must align")
    if (not np.all(np.isfinite(residual_sums)) or
            not np.all(np.isfinite(squared_residual_sums)) or
            np.any(squared_residual_sums < 0)):
        raise ValueError("service residual statistics must be finite and squared sums nonnegative")

    correction = np.zeros_like(residual_sums)
    reliability = np.zeros_like(residual_sums)
    touched = counts > 0
    means = np.zeros_like(residual_sums)
    means[touched] = residual_sums[touched] / counts[touched]
    singleton = counts == 1
    reliability[singleton & (means != 0)] = 0.5
    multiple = counts >= 2
    variance = np.zeros_like(residual_sums)
    variance[multiple] = np.maximum(
        squared_residual_sums[multiple] / counts[multiple] - means[multiple] ** 2,
        0.0)
    signal = means[multiple] ** 2
    denominator = signal + variance[multiple] / counts[multiple]
    valid = denominator > 0
    multiple_reliability = np.zeros_like(signal)
    multiple_reliability[valid] = signal[valid] / denominator[valid]
    reliability[multiple] = multiple_reliability
    correction[touched] = reliability[touched] * means[touched]
    return correction, reliability


def prediction_conditioned_service_residual(prediction_sums, residual_sums,
                                             squared_prediction_sums,
                                             prediction_residual_sums,
                                             squared_residual_sums, counts):
    """Fit a support-only hierarchical affine residual in normalized QoS scale.

    The correction is ``intercept[service] + slope * prediction``. A shared
    within-service slope is pooled across touched services, while service
    intercepts are empirical-Bayes shrunk toward the current global residual
    level according to their support uncertainty. Untouched services are exact
    no-ops. Inputs are sufficient statistics of float32-quantized normalized
    support predictions and residuals.
    """
    arrays = [np.asarray(values, dtype=float) for values in (
        prediction_sums, residual_sums, squared_prediction_sums,
        prediction_residual_sums, squared_residual_sums)]
    counts = np.asarray(counts, dtype=int)
    if any(values.shape != counts.shape for values in arrays) or np.any(counts < 0):
        raise ValueError("conditional service statistics and nonnegative counts must align")
    if (not all(np.all(np.isfinite(values)) for values in arrays) or
            np.any(arrays[2] < 0) or np.any(arrays[4] < 0)):
        raise ValueError("conditional service statistics must be finite with nonnegative squares")

    prediction_sums, residual_sums, squared_prediction_sums, prediction_residual_sums, squared_residual_sums = arrays
    touched = counts > 0
    intercepts = np.zeros_like(residual_sums)
    reliability = np.zeros_like(residual_sums)
    if not np.any(touched):
        return intercepts, 0.0, reliability

    n = counts[touched].astype(float)
    z_sum = prediction_sums[touched]
    r_sum = residual_sums[touched]
    centered_z2 = np.maximum(squared_prediction_sums[touched] - z_sum ** 2 / n, 0.0)
    centered_zr = prediction_residual_sums[touched] - z_sum * r_sum / n
    slope_denominator = float(np.sum(centered_z2))
    slope = (float(np.sum(centered_zr)) / slope_denominator
             if slope_denominator > 1e-12 else 0.0)
    slope = float(np.clip(slope, -1.0, 1.0))

    adjusted_sum = r_sum - slope * z_sum
    adjusted_squared_sum = np.maximum(
        squared_residual_sums[touched]
        - 2.0 * slope * prediction_residual_sums[touched]
        + slope ** 2 * squared_prediction_sums[touched], 0.0)
    service_means = adjusted_sum / n
    global_mean = float(np.sum(adjusted_sum) / np.sum(n))

    within_sse = np.maximum(adjusted_squared_sum - adjusted_sum ** 2 / n, 0.0)
    within_df = float(np.sum(np.maximum(n - 1.0, 0.0)))
    if within_df > 0:
        within_variance = float(np.sum(within_sse) / within_df)
        between_variance = float(np.average((service_means - global_mean) ** 2, weights=n))
        mean_sampling_variance = within_variance * float(np.average(1.0 / n, weights=n))
        prior_variance = max(between_variance - mean_sampling_variance, 0.0)
        denominator = prior_variance + within_variance / n
        touched_reliability = np.ones_like(n)
        valid = denominator > 1e-15
        touched_reliability[valid] = prior_variance / denominator[valid]
    else:
        touched_reliability = n / (n + 1.0)

    intercepts[touched] = global_mean + touched_reliability * (service_means - global_mean)
    reliability[touched] = touched_reliability
    return intercepts, slope, reliability


def log_domain_service_residual(residual_sums, squared_residual_sums, counts):
    """Shrink support-only service residuals in the fixed log1p QoS domain.

    Each touched service mean is empirically shrunk toward the current global
    support-residual mean.  The shrinkage is determined only by pooled within-
    service uncertainty and between-service variation; it has no tuned
    threshold and is shared by RT and TP.  Inputs are sufficient statistics of
    float32-quantized log-domain residuals.  Untouched services are exact
    no-ops.
    """
    residual_sums = np.asarray(residual_sums, dtype=float)
    squared_residual_sums = np.asarray(squared_residual_sums, dtype=float)
    counts = np.asarray(counts, dtype=int)
    if (residual_sums.shape != squared_residual_sums.shape or
            residual_sums.shape != counts.shape or np.any(counts < 0)):
        raise ValueError("log-domain service statistics and nonnegative counts must align")
    if (not np.all(np.isfinite(residual_sums)) or
            not np.all(np.isfinite(squared_residual_sums)) or
            np.any(squared_residual_sums < 0)):
        raise ValueError("log-domain service statistics must be finite with nonnegative squares")

    corrections = np.zeros_like(residual_sums)
    reliability = np.zeros_like(residual_sums)
    touched = counts > 0
    if not np.any(touched):
        return corrections, reliability, 0.0

    n = counts[touched].astype(float)
    sums = residual_sums[touched]
    means = sums / n
    global_mean = float(np.sum(sums) / np.sum(n))
    within_sse = np.maximum(
        squared_residual_sums[touched] - sums ** 2 / n, 0.0)
    within_df = float(np.sum(np.maximum(n - 1.0, 0.0)))
    if within_df > 0:
        within_variance = float(np.sum(within_sse) / within_df)
        between_variance = float(np.average(
            (means - global_mean) ** 2, weights=n))
        sampling_variance = within_variance * float(
            np.average(1.0 / n, weights=n))
        prior_variance = max(between_variance - sampling_variance, 0.0)
        denominator = prior_variance + within_variance / n
        touched_reliability = np.ones_like(n)
        valid = denominator > 1e-15
        touched_reliability[valid] = prior_variance / denominator[valid]
    else:
        touched_reliability = n / (n + 1.0)

    corrections[touched] = (
        global_mean + touched_reliability * (means - global_mean))
    reliability[touched] = touched_reliability
    return corrections, reliability, global_mean


def apply_service_raw_residual(predictions, services, corrections, raw_upper_bound):
    predictions = np.asarray(predictions, dtype=float)
    services = np.asarray(services, dtype=int)
    corrections = np.asarray(corrections, dtype=float)
    if predictions.shape != services.shape or np.any(services < 0) or np.any(services >= len(corrections)):
        raise ValueError("predictions and valid service indices must align")
    if raw_upper_bound <= 0 or not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(corrections)):
        raise ValueError("calibration inputs must be finite with a positive upper bound")
    return np.clip(predictions + corrections[services], 0.0, float(raw_upper_bound))


def apply_prediction_conditioned_service_residual(predictions, services, intercepts,
                                                   slope, touched, raw_upper_bound):
    """Apply a float32-quantized affine correction to touched services only."""
    predictions = np.asarray(predictions, dtype=float)
    services = np.asarray(services, dtype=int)
    intercepts = np.asarray(intercepts, dtype=np.float32)
    touched = np.asarray(touched, dtype=bool)
    slope = float(np.float32(slope))
    if (predictions.shape != services.shape or intercepts.shape != touched.shape or
            np.any(services < 0) or np.any(services >= len(intercepts))):
        raise ValueError("conditional predictions and valid service state must align")
    if raw_upper_bound <= 0 or not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(intercepts)):
        raise ValueError("conditional calibration inputs must be finite with a positive upper bound")
    correction = np.zeros_like(predictions)
    active = touched[services]
    normalized_prediction = predictions[active] / float(raw_upper_bound)
    correction[active] = float(raw_upper_bound) * (
        intercepts[services[active]].astype(float) + slope * normalized_prediction)
    return np.clip(predictions + correction, 0.0, float(raw_upper_bound))


def apply_log_domain_service_residual(predictions, services, corrections,
                                      touched, raw_upper_bound):
    """Apply float32 service residuals in the fixed normalized log1p domain."""
    predictions = np.asarray(predictions, dtype=float)
    services = np.asarray(services, dtype=int)
    corrections = np.asarray(corrections, dtype=np.float32)
    touched = np.asarray(touched, dtype=bool)
    if (predictions.shape != services.shape or corrections.shape != touched.shape or
            np.any(services < 0) or np.any(services >= len(corrections))):
        raise ValueError("log-domain predictions and valid service state must align")
    if (raw_upper_bound <= 0 or not np.all(np.isfinite(predictions)) or
            not np.all(np.isfinite(corrections))):
        raise ValueError("log-domain calibration inputs must be finite with a positive upper bound")
    result = predictions.copy()
    selected_corrections = corrections[services].astype(float)
    active = touched[services] & (selected_corrections != 0.0)
    if np.any(active):
        log_upper = np.log1p(float(raw_upper_bound))
        normalized = np.log1p(np.clip(predictions[active], 0.0, raw_upper_bound)) / log_upper
        calibrated = np.clip(normalized + selected_corrections[active], 0.0, 1.0)
        result[active] = np.expm1(calibrated * log_upper)
    return np.clip(result, 0.0, float(raw_upper_bound))


def apply_dual_constrained_log_service_residual(predictions, services, corrections,
                                                 touched, raw_upper_bound):
    """Apply the smaller raw displacement implied by log and raw domains.

    A normalized log-domain correction can be amplified by the inverse log1p
    transform for a large prediction.  This parameter-free guard interprets
    the same correction in both the normalized log domain and normalized raw
    domain, then keeps the candidate with the smaller absolute raw movement.
    It uses only the query prediction and the already transmitted service
    correction; no query target, extra state, or extra communication is needed.
    """
    predictions = np.asarray(predictions, dtype=float)
    services = np.asarray(services, dtype=int)
    corrections = np.asarray(corrections, dtype=np.float32)
    touched = np.asarray(touched, dtype=bool)
    if (predictions.shape != services.shape or corrections.shape != touched.shape or
            np.any(services < 0) or np.any(services >= len(corrections))):
        raise ValueError("dual-constrained predictions and valid service state must align")
    if (raw_upper_bound <= 0 or not np.all(np.isfinite(predictions)) or
            not np.all(np.isfinite(corrections))):
        raise ValueError("dual-constrained calibration inputs must be finite with a positive upper bound")

    log_candidate = apply_log_domain_service_residual(
        predictions, services, corrections, touched, raw_upper_bound)
    result = predictions.copy()
    selected_corrections = corrections[services].astype(float)
    active = touched[services] & (selected_corrections != 0.0)
    if np.any(active):
        raw_candidate = np.clip(
            predictions[active] + float(raw_upper_bound) * selected_corrections[active],
            0.0, float(raw_upper_bound))
        log_displacement = log_candidate[active] - predictions[active]
        raw_displacement = raw_candidate - predictions[active]
        use_log = np.abs(log_displacement) <= np.abs(raw_displacement)
        result[active] = np.where(use_log, log_candidate[active], raw_candidate)
    return np.clip(result, 0.0, float(raw_upper_bound))


def service_raw_residual_extra_bytes(support_entries, participants, services, value_bytes=8):
    if min(support_entries, participants, services) < 0 or value_bytes <= 0:
        raise ValueError("communication counts must be nonnegative")
    return (int(support_entries) * int(value_bytes),
            int(participants) * int(services) * int(value_bytes))


def prediction_conditioned_service_residual_extra_bytes(support_entries, participants,
                                                         services, value_bytes=4):
    """Count two float32 support values up and intercepts plus one slope down."""
    if min(support_entries, participants, services) < 0 or value_bytes <= 0:
        raise ValueError("communication counts must be nonnegative")
    return (2 * int(support_entries) * int(value_bytes),
            int(participants) * (int(services) + 1) * int(value_bytes))


def log_domain_service_residual_extra_bytes(support_entries, participants,
                                            services, value_bytes=4):
    """Count one float32 residual up and one service correction down."""
    if min(support_entries, participants, services) < 0 or value_bytes <= 0:
        raise ValueError("communication counts must be nonnegative")
    return (int(support_entries) * int(value_bytes),
            int(participants) * int(services) * int(value_bytes))


__all__ = ["apply_dual_constrained_log_service_residual",
           "apply_log_domain_service_residual",
           "apply_prediction_conditioned_service_residual",
           "apply_service_raw_residual", "mean_service_residual",
           "log_domain_service_residual",
           "log_domain_service_residual_extra_bytes",
           "prediction_conditioned_service_residual",
           "prediction_conditioned_service_residual_extra_bytes",
           "reliable_service_residual",
           "variance_aware_service_residual",
           "service_raw_residual_extra_bytes"]
