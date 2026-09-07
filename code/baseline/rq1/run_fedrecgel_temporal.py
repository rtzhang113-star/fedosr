"""Run the independent FedRecGEL-QoS causal adaptation.

The original FedRecGEL repository trains a binary recommender with sampled
negative items.  This runner keeps the item-centred private/shared/SAM
mechanism while replacing the task head with continuous QoS regression.  Each
time slice follows the frozen project protocol:

``history state + support[t] -> local/SAM update + shared aggregation -> query[t]``.

Query and future labels are used only by the metric calculation.  No
checkpoint selection or parameter update reads them.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FED_CODE = ROOT / "code" / "fed_online_qos"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(FED_CODE) not in sys.path:
    sys.path.insert(0, str(FED_CODE))

from run_fed_amf import evaluate, load_masks, load_values  # noqa: E402
from methods.fedrecgel import FedRecGELQoS  # noqa: E402


FIELDS = (
    "time_slice", "support_entries", "query_entries", "participating_clients",
    "active_services", "mae", "rmse", "mre", "npre", "support_target_mean",
    "support_loss_before", "support_loss_after", "support_prediction_mean_before",
    "support_prediction_mean_after", "support_prediction_min_before",
    "support_prediction_max_before", "support_prediction_boundary_fraction_before",
    "support_prediction_min_after", "support_prediction_max_after",
    "support_prediction_boundary_fraction_after", "support_scale_initialized_this_slice",
    "update_seconds", "evaluation_seconds",
    "uplink_bytes", "downlink_bytes", "shared_gradient_clients", "query_read_only",
)
SUMMARY_METRICS = (
    "mae", "rmse", "mre", "npre", "update_seconds", "evaluation_seconds",
    "support_target_mean", "support_loss_before", "support_loss_after",
    "support_prediction_mean_before", "support_prediction_mean_after",
    "support_prediction_min_before", "support_prediction_max_before",
    "support_prediction_boundary_fraction_before", "support_prediction_min_after",
    "support_prediction_max_after", "support_prediction_boundary_fraction_after",
)


def fixed_log_transform(values: np.ndarray, upper_bound: float) -> np.ndarray:
    """Apply the frozen ex-ante ``log1p`` normalization."""
    values = np.asarray(values, dtype=np.float64)
    upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("upper_bound must be positive and finite")
    # ``np.all`` deliberately accepts an empty support vector: an empty
    # support round is a valid no-op update and must remain causally safe.
    if not np.all(np.isfinite(values)) or not np.all((values > 0.0) & (values < upper_bound)):
        raise ValueError("QoS values must lie strictly inside the frozen domain")
    return (np.log1p(values) / np.log1p(upper_bound)).astype(np.float32)


def inverse_fixed_log_transform(values: np.ndarray, upper_bound: float) -> np.ndarray:
    upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("upper_bound must be positive and finite")
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("normalized predictions must be finite")
    return np.expm1(np.clip(values, 0.0, 1.0) * np.log1p(upper_bound))


def _arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _copy_state(model: FedRecGELQoS) -> dict:
    return model.state_copy()


def _state_unchanged(before: dict, model: FedRecGELQoS) -> bool:
    return model.state_unchanged(before)


def run_arrays(
    args: Any,
    values: np.ndarray,
    available: np.ndarray,
    support: np.ndarray,
    upper_bound: float,
    density: float = 0.0,
) -> tuple[list[dict], dict]:
    """Run FedRecGEL-QoS on an in-memory causal tensor.

    This function intentionally accepts the full values tensor only because
    the runner must score query rows.  The model receives support values for
    the current slice and never receives query/future labels.
    """
    values = np.asarray(values, dtype=np.float32)
    available = np.asarray(available, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if values.ndim != 3 or values.shape != available.shape or values.shape != support.shape:
        raise ValueError("values, available and support must be aligned [time, user, service] tensors")
    if np.any(support & ~available):
        raise ValueError("support cannot contain unavailable entries")
    if not np.isfinite(float(upper_bound)) or float(upper_bound) <= 0.0:
        raise ValueError("upper_bound must be positive and finite")

    num_times, num_users, num_services = values.shape
    slices = min(int(_arg(args, "slices", num_times)), num_times)
    if slices <= 0:
        raise ValueError("slices must be positive")

    model = FedRecGELQoS(
        num_users=num_users,
        num_services=num_services,
        factor_num=int(_arg(args, "rank", 32)),
        num_layers=int(_arg(args, "num_layers", 3)),
        local_lr=float(_arg(args, "local_lr", 0.01)),
        global_lr=float(_arg(args, "global_lr", 0.001)),
        rho_sh=float(_arg(args, "rho_sh", 0.02)),
        rho_ns=float(_arg(args, "rho_ns", 0.02)),
        local_epochs=int(_arg(args, "local_epochs", 1)),
        client_fraction=float(_arg(args, "client_fraction", 0.1)),
        seed=int(_arg(args, "seed", 42)),
        communication_enabled=bool(_arg(args, "communication_enabled", True)),
    )

    rows: list[dict] = []
    wall_start = time.perf_counter()
    for time_slice in range(slices):
        support_users, support_services = np.nonzero(support[time_slice])
        support_targets = fixed_log_transform(
            values[time_slice, support_users, support_services], upper_bound
        )
        scale_initialized_this_slice = model.initialize_output_scale(support_targets)
        support_before = model.support_reconstruction_metrics(
            support_users, support_services, support_targets
        )
        update_start = time.perf_counter()
        update = model.update_round(
            support_users,
            support_services,
            support_targets,
            epochs=int(_arg(args, "local_epochs", 1)),
        )
        update_seconds = time.perf_counter() - update_start
        support_after = model.support_reconstruction_metrics(
            support_users, support_services, support_targets
        )

        query_users, query_services = np.nonzero(available[time_slice] & ~support[time_slice])
        if query_users.size == 0:
            raise RuntimeError(f"slice {time_slice} has no query entries")
        before = _copy_state(model)
        evaluation_start = time.perf_counter()
        normalized_prediction = model.predict(query_users, query_services)
        predicted = inverse_fixed_log_transform(normalized_prediction, upper_bound)
        metrics = evaluate(values[time_slice, query_users, query_services], predicted)
        evaluation_seconds = time.perf_counter() - evaluation_start
        query_read_only = _state_unchanged(before, model)

        row = {
            "time_slice": int(time_slice),
            "support_entries": int(support_users.size),
            "query_entries": int(query_users.size),
            **update.as_dict(),
            **metrics,
            "support_target_mean": float(support_before["support_target_mean"]),
            "support_loss_before": float(support_before["support_loss"]),
            "support_loss_after": float(support_after["support_loss"]),
            "support_prediction_mean_before": float(support_before["support_prediction_mean"]),
            "support_prediction_mean_after": float(support_after["support_prediction_mean"]),
            "support_prediction_min_before": float(support_before["support_prediction_min"]),
            "support_prediction_max_before": float(support_before["support_prediction_max"]),
            "support_prediction_boundary_fraction_before": float(
                support_before["support_prediction_boundary_fraction"]
            ),
            "support_prediction_min_after": float(support_after["support_prediction_min"]),
            "support_prediction_max_after": float(support_after["support_prediction_max"]),
            "support_prediction_boundary_fraction_after": float(
                support_after["support_prediction_boundary_fraction"]
            ),
            "support_scale_initialized_this_slice": bool(scale_initialized_this_slice),
            "update_seconds": float(update_seconds),
            "evaluation_seconds": float(evaluation_seconds),
            "query_read_only": bool(query_read_only),
        }
        if not query_read_only or not all(math.isfinite(float(row[key])) for key in SUMMARY_METRICS):
            raise RuntimeError(f"slice {time_slice} failed finite/read-only audit")
        rows.append(row)

    summary = {
        "method": "FedRecGEL-QoS",
        "model_variant": "GMF+MLP+regression-head",
        "causal_protocol": "history+support[t]->query[t]",
        "qos": str(_arg(args, "qos", "rt")),
        "density": float(density),
        "seed": int(_arg(args, "seed", 42)),
        "time_slices": int(slices),
        "preprocessing": "fixed_log1p",
        "upper_bound": float(upper_bound),
        "factor_num": int(model.factor_num),
        "num_layers": int(model.num_layers),
        "local_lr": float(model.local_lr),
        "global_lr": float(model.global_lr),
        "rho_sh": float(model.rho_sh),
        "rho_ns": float(model.rho_ns),
        "local_epochs": int(model.local_epochs),
        "client_fraction": float(model.client_fraction),
        "source_experiment_defaults": {
            "global_lr": 1.5,
            "local_lr": 0.1,
            "client_fraction": 0.1,
            "rho_sh": 0.02,
            "rho_ns": 0.02,
            "local_epochs": 1,
        },
        "qos_regression_defaults": {
            "global_lr": 0.001,
            "local_lr": 0.01,
            "reason": "bounded continuous-QoS regression stability",
        },
        "output_activation": "sigmoid",
        "support_scale_initialization_count": int(model.support_scale_initialization_count),
        "support_scale_initialization": model.support_scale_info,
        "shared_parameter_count": int(model.shared_parameter_count),
        "shared_parameter_bytes": int(model.shared_parameter_bytes),
        "query_read_only_all": bool(all(row["query_read_only"] for row in rows)),
        "privacy_cryptographic_layer_implemented": False,
        **{key: float(np.mean([row[key] for row in rows])) for key in SUMMARY_METRICS},
        "support_entries": int(sum(row["support_entries"] for row in rows)),
        "query_entries": int(sum(row["query_entries"] for row in rows)),
        "participating_clients": int(sum(row["participating_clients"] for row in rows)),
        "active_services": int(sum(row["active_services"] for row in rows)),
        "shared_gradient_clients": int(sum(row["shared_gradient_clients"] for row in rows)),
        "uplink_bytes": int(sum(row["uplink_bytes"] for row in rows)),
        "downlink_bytes": int(sum(row["downlink_bytes"] for row in rows)),
        "wall_seconds": float(time.perf_counter() - wall_start),
    }
    return rows, summary


def run_stream(args: Any) -> tuple[list[dict], dict]:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    if manifest["qos_type"] != args.qos or int(manifest["seed"]) != int(args.seed):
        raise RuntimeError("manifest identity does not match command")
    preprocessing = manifest.get("preprocessing", {})
    if preprocessing.get("name") != "fixed_log1p":
        raise RuntimeError("FedRecGEL-QoS requires a fixed_log1p manifest")
    values, data_sha256 = load_values(Path(args.data), args.qos)
    if data_sha256 != manifest["data_sha256"]:
        raise RuntimeError("dataset hash does not match manifest")
    available, support = load_masks(Path(args.mask), args.qos)
    upper_bound = float(preprocessing["upper_bound"])
    return run_arrays(args, values, available, support, upper_bound, float(manifest["density"]))


def write_artifacts(output: Path, rows: list[dict], summary: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_time.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal FedRecGEL-QoS baseline")
    parser.add_argument("--qos", choices=("rt", "tp"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--slices", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--local-lr", type=float, default=0.01)
    parser.add_argument("--global-lr", type=float, default=0.001)
    parser.add_argument("--rho-sh", type=float, default=0.02)
    parser.add_argument("--rho-ns", type=float, default=0.02)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--client-fraction", type=float, default=0.1)
    parser.add_argument("--no-communication", dest="communication_enabled", action="store_false")
    parser.set_defaults(communication_enabled=True)
    args = parser.parse_args()
    rows, summary = run_stream(args)
    write_artifacts(args.output, rows, summary)


if __name__ == "__main__":
    main()
