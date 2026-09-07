"""Run causal FMF/EFMF on a frozen WS-DREAM Dataset 3 protocol."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FED_CODE = ROOT / "code" / "fed_online_qos"
sys.path.insert(0, str(FED_CODE))

from run_fed_amf import evaluate, load_masks, load_values  # noqa: E402
from methods.efmf import EfficientFederatedMatrixFactorization  # noqa: E402

FIELDS = (
    "time_slice", "support_entries", "query_entries", "mae", "rmse", "mre", "npre",
    "support_target_mean", "support_loss_before", "support_loss_after",
    "support_prediction_mean_before", "support_prediction_mean_after",
    "update_seconds", "evaluation_seconds", "participating_clients",
    "transmitted_service_vectors", "delayed_service_vectors", "uplink_bytes",
    "downlink_bytes", "query_read_only",
)
SUMMARY_METRICS = (
    "mae", "rmse", "mre", "npre", "update_seconds", "evaluation_seconds",
    "support_target_mean", "support_loss_before", "support_loss_after",
    "support_prediction_mean_before", "support_prediction_mean_after",
)
PAPER_ALPHA = {"rt": -0.007, "tp": -0.005}
PAPER_UPPER_BOUND = {"rt": 20.0, "tp": 1000.0}


def fixed_log_transform(values, upper_bound: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(upper_bound) or upper_bound <= 0:
        raise ValueError("upper_bound must be positive and finite")
    if not np.all(np.isfinite(values)) or not np.all((values > 0) & (values < upper_bound)):
        raise ValueError("QoS values must lie strictly inside the frozen domain")
    return (np.log1p(values) / np.log1p(float(upper_bound))).astype(np.float32)


def inverse_fixed_log_transform(values, upper_bound: float) -> np.ndarray:
    return np.expm1(
        np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
        * np.log1p(float(upper_bound))
    )


def _box_cox(values, alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("Box-Cox requires finite positive values")
    if abs(alpha) < 1e-12:
        return np.log(values)
    return np.expm1(alpha * np.log(values)) / alpha


def paper_boxcox_transform(values, alpha: float, lower_bound: float, upper_bound: float):
    if not 0 < lower_bound < upper_bound:
        raise ValueError("paper Box-Cox bounds must satisfy 0 < lower < upper")
    values = np.asarray(values, dtype=np.float64)
    if not np.all((values >= lower_bound) & (values <= upper_bound)):
        raise ValueError("values fall outside frozen paper Box-Cox bounds")
    low = float(_box_cox([lower_bound], alpha)[0])
    high = float(_box_cox([upper_bound], alpha)[0])
    return ((_box_cox(values, alpha) - low) / (high - low)).astype(np.float32)


def inverse_paper_boxcox_transform(values, alpha: float, lower_bound: float, upper_bound: float):
    low = float(_box_cox([lower_bound], alpha)[0])
    high = float(_box_cox([upper_bound], alpha)[0])
    transformed = low + np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0) * (high - low)
    if abs(alpha) < 1e-12:
        raw = np.exp(transformed)
    else:
        base = 1.0 + alpha * transformed
        if np.any(base <= 0):
            raise ValueError("invalid inverse Box-Cox domain")
        raw = np.exp(np.log(base) / alpha)
    return np.clip(raw, lower_bound, upper_bound)


def _transform(values, args, upper_bound):
    if args.preprocessing == "manifest_fixed_log1p":
        return fixed_log_transform(values, upper_bound)
    return paper_boxcox_transform(
        values, args.boxcox_alpha, args.boxcox_lower_bound, args.boxcox_upper_bound
    )


def _inverse(values, args, upper_bound):
    if args.preprocessing == "manifest_fixed_log1p":
        return inverse_fixed_log_transform(values, upper_bound)
    return inverse_paper_boxcox_transform(
        values, args.boxcox_alpha, args.boxcox_lower_bound, args.boxcox_upper_bound
    )


def run_arrays(args, values, available, support, upper_bound: float, density: float = 0.0):
    values = np.asarray(values)
    available = np.asarray(available, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if values.shape != available.shape or values.shape != support.shape or values.ndim != 3:
        raise ValueError("values, available and support must be aligned time-user-service tensors")
    if np.any(support & ~available):
        raise ValueError("support cannot contain unavailable entries")
    if int(args.slices) <= 0:
        raise ValueError("slices must be positive")
    slices = min(int(args.slices), values.shape[0])
    model = EfficientFederatedMatrixFactorization(
        values.shape[1], values.shape[2], rank=args.rank,
        learning_rate=args.learning_rate, regularization=args.regularization,
        seed=args.seed, mode=args.mode, omit_fraction=args.omit_fraction,
        quantization_bits=args.quantization_bits,
        communication_threshold=args.communication_threshold,
        communication_enabled=args.communication_enabled,
    )
    rows = []
    wall_start = time.perf_counter()
    for time_slice in range(slices):
        users, services = np.nonzero(support[time_slice])
        targets = _transform(values[time_slice, users, services], args, upper_bound)
        # The only scale statistic is the normalized label mean from the first
        # support slice.  It is applied before the first audit/update and never
        # recomputed from later slices or query values.
        if time_slice == 0:
            model.initialize_from_support_targets(targets)
        support_before = model.support_reconstruction_metrics(users, services, targets)
        update_start = time.perf_counter()
        update = model.update_round(users, services, targets, epochs=args.local_epochs)
        update_seconds = time.perf_counter() - update_start
        support_after = model.support_reconstruction_metrics(users, services, targets)

        query_users, query_services = np.nonzero(available[time_slice] & ~support[time_slice])
        snapshot = model.state_copy()
        evaluation_start = time.perf_counter()
        normalized = model.predict(query_users, query_services)
        predicted = _inverse(normalized, args, upper_bound)
        metrics = evaluate(values[time_slice, query_users, query_services], predicted)
        evaluation_seconds = time.perf_counter() - evaluation_start
        query_read_only = model.state_unchanged(snapshot)
        row = {
            "time_slice": time_slice,
            "support_entries": int(len(users)),
            "query_entries": int(len(query_users)),
            **metrics,
            "support_target_mean": float(support_before["support_target_mean"]),
            "support_loss_before": float(support_before["support_loss"]),
            "support_loss_after": float(support_after["support_loss"]),
            "support_prediction_mean_before": float(support_before["support_prediction_mean"]),
            "support_prediction_mean_after": float(support_after["support_prediction_mean"]),
            "update_seconds": float(update_seconds),
            "evaluation_seconds": float(evaluation_seconds),
            **update.as_dict(),
            "query_read_only": bool(query_read_only),
        }
        # update.as_dict contains the same support_entries by construction.
        if not query_read_only or not all(math.isfinite(float(row[key])) for key in SUMMARY_METRICS):
            raise RuntimeError(f"slice {time_slice} failed finite/read-only audit")
        rows.append(row)
    summary = {
        "method": args.mode.upper(),
        "qos": args.qos,
        "density": float(density),
        "seed": int(args.seed),
        "time_slices": int(slices),
        "preprocessing": args.preprocessing,
        "paper_alpha": float(args.boxcox_alpha),
        "paper_boxcox_lower_bound": float(args.boxcox_lower_bound),
        "paper_boxcox_upper_bound": float(args.boxcox_upper_bound),
        "support_scale_initialization": model.support_scale_initialization_info(),
        "support_scale_initialization_count": model.support_scale_initialization_count,
        "support_loss_reduction_mean": float(
            np.mean([row["support_loss_before"] - row["support_loss_after"] for row in rows])
        ),
        "privacy_cryptographic_layer_implemented": False,
        **{key: float(np.mean([row[key] for row in rows])) for key in SUMMARY_METRICS},
        "uplink_bytes": int(sum(row["uplink_bytes"] for row in rows)),
        "downlink_bytes": int(sum(row["downlink_bytes"] for row in rows)),
        "transmitted_service_vectors": int(sum(row["transmitted_service_vectors"] for row in rows)),
        "delayed_service_vectors": int(sum(row["delayed_service_vectors"] for row in rows)),
        "wall_seconds": float(time.perf_counter() - wall_start),
    }
    return rows, summary


def run_stream(args):
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    if manifest["qos_type"] != args.qos or int(manifest["seed"]) != args.seed:
        raise RuntimeError("manifest identity does not match command")
    preprocessing = manifest.get("preprocessing", {})
    if args.preprocessing == "manifest_fixed_log1p" and preprocessing.get("name") != "fixed_log1p":
        raise RuntimeError("manifest_fixed_log1p requires a fixed_log1p manifest")
    upper_bound = float(preprocessing["upper_bound"])
    values, data_sha256 = load_values(Path(args.data), args.qos)
    if data_sha256 != manifest["data_sha256"]:
        raise RuntimeError("dataset hash does not match manifest")
    available, support = load_masks(Path(args.mask), args.qos)
    return run_arrays(args, values, available, support, upper_bound, float(manifest["density"]))


def write_artifacts(output: Path, rows, summary):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_time.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qos", choices=("rt", "tp"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--slices", type=int, default=64)
    parser.add_argument("--mode", choices=("fmf", "efmf"), default="efmf")
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=0.001)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--omit-fraction", type=float, default=0.2)
    parser.add_argument("--quantization-bits", type=int, default=8)
    parser.add_argument("--communication-threshold", type=float, default=0.0)
    parser.add_argument("--no-communication", dest="communication_enabled", action="store_false")
    parser.set_defaults(communication_enabled=True)
    parser.add_argument(
        "--preprocessing", choices=("manifest_fixed_log1p", "paper_boxcox"),
        default="manifest_fixed_log1p",
    )
    parser.add_argument("--boxcox-alpha", type=float)
    parser.add_argument("--boxcox-lower-bound", type=float, default=1e-6)
    parser.add_argument("--boxcox-upper-bound", type=float)
    args = parser.parse_args()
    if args.boxcox_alpha is None:
        args.boxcox_alpha = PAPER_ALPHA[args.qos]
    if args.boxcox_upper_bound is None:
        args.boxcox_upper_bound = PAPER_UPPER_BOUND[args.qos]
    rows, summary = run_stream(args)
    write_artifacts(args.output, rows, summary)


if __name__ == "__main__":
    main()
