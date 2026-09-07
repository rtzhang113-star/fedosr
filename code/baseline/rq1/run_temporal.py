"""Run one causal FL-LFT stream and write audited minimal artifacts.

The protocol owns preprocessing: support targets and query predictions use the
manifest's fixed ``log1p`` transform and its matching inverse.  The model is
initialized once from the first support slice only; query and future labels
are never passed to model construction or updates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FED_CODE = ROOT / "code" / "fed_online_qos"
sys.path.insert(0, str(FED_CODE))

from run_fed_amf import evaluate, load_masks, load_values  # noqa: E402
from methods.fl_lft import FederatedLatentTensorFactorization  # noqa: E402

FIELDS = (
    "time_slice", "support_entries", "query_entries", "mae", "rmse", "mre", "npre",
    "support_target_mean", "support_loss_before", "support_loss_after",
    "support_loss_reduction", "support_prediction_mean_before",
    "support_prediction_mean_after", "support_convergence_passed",
    "prediction_zero_fraction", "prediction_finite", "update_seconds",
    "evaluation_seconds", "uplink_bytes", "downlink_bytes", "query_read_only",
)
SUMMARY_METRICS = ("mae", "rmse", "mre", "npre", "update_seconds", "evaluation_seconds")
SUPPORT_AUDIT_METRICS = (
    "support_target_mean", "support_loss_before", "support_loss_after",
    "support_loss_reduction", "support_prediction_mean_before",
    "support_prediction_mean_after", "prediction_zero_fraction",
)
_SUPPORT_CONVERGENCE_ABS_TOL = 1e-12
_SUPPORT_CONVERGENCE_REL_TOL = 1e-9


def fixed_log_transform(values, upper_bound: float) -> np.ndarray:
    """Apply the manifest's fixed ``log1p`` normalization.

    ``upper_bound`` comes from the manifest and is never fitted from loaded
    values.  Rejecting values outside the fixed domain prevents a linear
    ``value / upper_bound`` path from silently returning.
    """
    values = np.asarray(values, dtype=np.float64)
    upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("upper_bound must be positive and finite")
    if not np.all(np.isfinite(values)) or not np.all(
        (values > 0.0) & (values < upper_bound)
    ):
        raise ValueError("QoS values must be finite and strictly inside fixed_log1p domain")
    return np.log1p(values) / np.log1p(upper_bound)


def inverse_fixed_log_transform(values, upper_bound: float) -> np.ndarray:
    """Invert fixed ``log1p`` predictions after enforcing its output domain."""
    values = np.asarray(values, dtype=np.float64)
    upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("upper_bound must be positive and finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("normalized predictions must be finite")
    return np.expm1(np.clip(values, 0.0, 1.0) * np.log1p(upper_bound))


def _support_converged(before: float, after: float) -> bool:
    """Return the per-slice support-only convergence gate decision."""
    tolerance = max(_SUPPORT_CONVERGENCE_ABS_TOL, _SUPPORT_CONVERGENCE_REL_TOL * abs(before))
    return bool(after <= before + tolerance)


def run_arrays(args, values, available, support, upper_bound: float, density: float = 0.0):
    """Run the causal stream on arrays, exposing a leakage-testable entry point.

    Only ``values[time_slice][support[time_slice]]`` is sent to
    ``update_client``.  Query values are consumed solely by ``evaluate`` after
    prediction, and future values are not read until their own slice.
    """
    values = np.asarray(values, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if values.shape != available.shape or values.shape != support.shape or values.ndim != 3:
        raise ValueError("values, available and support must be aligned 3-D tensors")
    if np.any(support & ~available):
        raise ValueError("support cannot contain unavailable entries")
    upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("upper_bound must be positive and finite")

    num_times, num_users, num_services = values.shape
    requested_slices = int(getattr(args, "slices", num_times))
    if requested_slices <= 0:
        raise ValueError("slices must be positive")
    slices = min(requested_slices, num_times)
    rank = int(getattr(args, "rank", 10))
    learning_rate = float(getattr(args, "learning_rate", 0.00038))
    regularization = float(getattr(args, "regularization", 0.001))
    local_epochs = int(getattr(args, "local_epochs", 1))
    seed = int(getattr(args, "seed", 42))
    qos = str(getattr(args, "qos", "unknown"))
    if rank <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("rank must be positive and learning_rate must be finite/positive")
    if regularization < 0.0 or not np.isfinite(regularization) or local_epochs <= 0:
        raise ValueError("regularization must be finite/non-negative and local_epochs positive")

    model = FederatedLatentTensorFactorization(
        num_users, num_services, num_times, rank=rank,
        learning_rate=learning_rate, regularization=regularization, seed=seed,
    )

    # The only scale statistic is the first slice's support mean.  Initializing
    # all three factor tables once gives every time row a usable multiplicative
    # scale while remaining independent of query and future labels.
    first_users, first_services = np.nonzero(support[0])
    first_targets = fixed_log_transform(
        values[0, first_users, first_services], upper_bound
    )
    initialization = model.initialize_from_support_targets(first_targets)

    rows: list[dict] = []
    wall_start = time.perf_counter()
    for time_slice in range(slices):
        support_users, support_services = np.nonzero(support[time_slice])
        normalized = fixed_log_transform(
            values[time_slice, support_users, support_services], upper_bound
        )
        time_indices = np.full(len(support_users), time_slice, dtype=np.int64)
        support_before = model.support_reconstruction_metrics(
            support_users, support_services, time_indices, normalized,
        )

        update_start = time.perf_counter()
        uplink_bytes = 0
        downlink_bytes = 0
        for user in np.unique(support_users):
            selected = support_users == user
            services = support_services[selected]
            targets = normalized[selected]
            result = model.update_client(
                int(user), services,
                np.full(len(services), time_slice, dtype=np.int64),
                targets, epochs=local_epochs,
            )
            uplink_bytes += int(result["uploaded_bytes"])
            downlink_bytes += int(result["downloaded_bytes"])
        update_seconds = time.perf_counter() - update_start

        support_after = model.support_reconstruction_metrics(
            support_users, support_services, time_indices, normalized,
        )
        support_loss_reduction = float(
            support_before["support_loss"] - support_after["support_loss"]
        )
        support_convergence_passed = _support_converged(
            support_before["support_loss"], support_after["support_loss"]
        )

        query_users, query_services = np.nonzero(available[time_slice] & ~support[time_slice])
        if len(query_users) == 0:
            raise RuntimeError(f"slice {time_slice} has no query entries")
        before = (
            model.private_user_factors.copy(), model.shared_service_factors.copy(),
            model.shared_time_factors.copy(),
        )
        evaluation_start = time.perf_counter()
        normalized_predicted = model.predict(
            query_users, query_services,
            np.full(len(query_users), time_slice, dtype=np.int64),
        )
        prediction_finite = bool(np.all(np.isfinite(normalized_predicted)))
        if not prediction_finite:
            raise RuntimeError(f"slice {time_slice} produced non-finite normalized predictions")
        predicted = np.clip(
            inverse_fixed_log_transform(normalized_predicted, upper_bound),
            0.0, upper_bound,
        )
        prediction_finite = bool(prediction_finite and np.all(np.isfinite(predicted)))
        if not prediction_finite:
            raise RuntimeError(f"slice {time_slice} produced non-finite raw predictions")
        prediction_zero_fraction = float(np.mean(predicted <= 0.0))
        metrics = evaluate(values[time_slice, query_users, query_services], predicted)
        evaluation_seconds = time.perf_counter() - evaluation_start
        query_read_only = all(
            np.array_equal(left, right) for left, right in zip(
                before,
                (model.private_user_factors, model.shared_service_factors, model.shared_time_factors),
            )
        )
        row = {
            "time_slice": int(time_slice),
            "support_entries": int(len(support_users)),
            "query_entries": int(len(query_users)),
            **metrics,
            "support_target_mean": float(support_before["support_target_mean"]),
            "support_loss_before": float(support_before["support_loss"]),
            "support_loss_after": float(support_after["support_loss"]),
            "support_loss_reduction": support_loss_reduction,
            "support_prediction_mean_before": float(
                support_before["support_prediction_mean"]
            ),
            "support_prediction_mean_after": float(
                support_after["support_prediction_mean"]
            ),
            "support_convergence_passed": support_convergence_passed,
            "prediction_zero_fraction": prediction_zero_fraction,
            "prediction_finite": prediction_finite,
            "update_seconds": float(update_seconds),
            "evaluation_seconds": float(evaluation_seconds),
            "uplink_bytes": uplink_bytes,
            "downlink_bytes": downlink_bytes,
            "query_read_only": query_read_only,
        }
        if not query_read_only:
            raise RuntimeError(f"slice {time_slice} query read-only audit failed")
        if not support_convergence_passed:
            raise RuntimeError(
                f"slice {time_slice} support-only convergence gate failed: "
                f"{support_before['support_loss']:.17g} -> {support_after['support_loss']:.17g}"
            )
        if prediction_zero_fraction >= 1.0:
            raise RuntimeError(f"slice {time_slice} prediction-zero collapse gate failed")
        if not all(
            math.isfinite(float(row[key]))
            for key in SUMMARY_METRICS + SUPPORT_AUDIT_METRICS
        ):
            raise RuntimeError(f"slice {time_slice} failed finite metric audit")
        rows.append(row)

    summary = {
        "method": "FL-LFT",
        "protocol": "causal_streaming_current_slice_completion",
        "qos": qos,
        "density": float(density),
        "seed": seed,
        "time_slices": int(slices),
        "rank": rank,
        "learning_rate": learning_rate,
        "regularization": regularization,
        "local_epochs": local_epochs,
        "preprocessing": "fixed_log1p",
        "normalization_formula": "log1p(values) / log1p(manifest.preprocessing.upper_bound)",
        "inverse_normalization": "fixed_expm1",
        "inverse_normalization_formula": "expm1(normalized * log1p(manifest.preprocessing.upper_bound))",
        "support_scale_initialization": initialization,
        "support_scale_initialization_count": model.support_scale_initialization_count,
        "support_convergence_gate": "support MSE after update <= before update per slice",
        "support_convergence_passed": bool(all(row["support_convergence_passed"] for row in rows)),
        "support_loss_reduction_mean": float(
            np.mean([row["support_loss_reduction"] for row in rows])
        ),
        "support_loss_before_mean": float(
            np.mean([row["support_loss_before"] for row in rows])
        ),
        "support_loss_after_mean": float(
            np.mean([row["support_loss_after"] for row in rows])
        ),
        "prediction_zero_fraction_mean": float(
            np.mean([row["prediction_zero_fraction"] for row in rows])
        ),
        "prediction_zero_fraction_max": float(
            np.max([row["prediction_zero_fraction"] for row in rows])
        ),
        "prediction_finite_all": bool(all(row["prediction_finite"] for row in rows)),
        "query_read_only_all": bool(all(row["query_read_only"] for row in rows)),
        **{key: float(np.mean([row[key] for row in rows])) for key in SUMMARY_METRICS},
        "support_entries": int(sum(row["support_entries"] for row in rows)),
        "query_entries": int(sum(row["query_entries"] for row in rows)),
        "uplink_bytes": int(sum(row["uplink_bytes"] for row in rows)),
        "downlink_bytes": int(sum(row["downlink_bytes"] for row in rows)),
        "wall_seconds": float(time.perf_counter() - wall_start),
    }
    return rows, summary


def run_stream(args) -> tuple[list[dict], dict]:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    if manifest.get("qos_type") != args.qos or int(manifest.get("seed", -1)) != int(args.seed):
        raise RuntimeError("manifest identity does not match command")
    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, dict) or preprocessing.get("name") != "fixed_log1p":
        raise RuntimeError(
            "FL-LFT requires manifest.preprocessing.name == 'fixed_log1p'"
        )
    try:
        upper_bound = float(preprocessing["upper_bound"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("fixed_log1p manifest must provide a finite upper_bound") from exc
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise RuntimeError("fixed_log1p manifest upper_bound must be positive and finite")
    density = float(manifest["density"])
    values, data_sha256 = load_values(Path(args.data), args.qos)
    if data_sha256 != manifest["data_sha256"]:
        raise RuntimeError("dataset hash does not match manifest")
    available, support = load_masks(Path(args.mask), args.qos)
    if values.shape != available.shape or available.shape != support.shape:
        raise RuntimeError("data/mask shape mismatch")
    return run_arrays(args, values, available, support, upper_bound, density)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qos", choices=("rt", "tp"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--slices", type=int, default=64)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.00038)
    parser.add_argument("--regularization", type=float, default=0.001)
    parser.add_argument("--local-epochs", type=int, default=1)
    args = parser.parse_args()
    rows, summary = run_stream(args)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_time.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
