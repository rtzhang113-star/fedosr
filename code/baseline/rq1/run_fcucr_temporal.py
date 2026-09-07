"""Run the independent FCUCR-QoS causal stream on WS-DREAM masks."""
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
if str(FED_CODE) not in sys.path:
    sys.path.insert(0, str(FED_CODE))

from run_fed_amf import evaluate, load_masks, load_values  # noqa: E402
from methods.fcucr import (
    FCUCRQoS,
    MIN_TASD_EPOCHS,
    SCALE_INITIALIZATION_STRATEGY,
    TASD_STRATEGY,
)  # noqa: E402


UPSTREAM_REPOSITORY = "https://github.com/Poizoner/code4FCUCR_www2026"
UPSTREAM_COMMIT = "1ba1f054d94eac15a6a19965cf8e45db78a94782"

FIELDS = (
    "time_slice",
    "support_entries",
    "query_entries",
    "participating_clients",
    "kb_entries_before",
    "kb_entries_after",
    "retrieved_prototypes",
    "mae",
    "rmse",
    "mre",
    "npre",
    "support_loss_before",
    "support_loss_after",
    "distillation_loss",
    "update_seconds",
    "evaluation_seconds",
    "uplink_bytes",
    "downlink_bytes",
    "query_read_only",
)
SUMMARY_METRICS = (
    "mae",
    "rmse",
    "mre",
    "npre",
    "support_loss_before",
    "support_loss_after",
    "distillation_loss",
    "update_seconds",
    "evaluation_seconds",
)


def fixed_log_transform(values, upper_bound: float) -> np.ndarray:
    """Apply the frozen log1p transform without fitting on any stream data."""
    values = np.asarray(values, dtype=np.float64)
    upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0:
        raise ValueError("upper_bound must be positive and finite")
    if not np.all(np.isfinite(values)) or not np.all(
        (values > 0.0) & (values < upper_bound)
    ):
        raise ValueError("QoS values must lie strictly inside the frozen domain")
    return (np.log1p(values) / np.log1p(upper_bound)).astype(np.float32)


def inverse_fixed_log_transform(values, upper_bound: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0:
        raise ValueError("upper_bound must be positive and finite")
    return np.expm1(np.clip(values, 0.0, 1.0) * np.log1p(upper_bound))


def _arg(args, name, default):
    return getattr(args, name, default)


def _empty_metrics() -> dict:
    # A slice with no query observations has no estimand.  Keeping zeroes
    # allows the artifact schema to stay finite; query_entries records this
    # explicitly and such slices should be excluded from aggregate accuracy.
    return {"mae": 0.0, "rmse": 0.0, "mre": 0.0, "npre": 0.0}


def _evaluate(real, predicted) -> dict:
    real = np.asarray(real)
    predicted = np.asarray(predicted)
    if real.size == 0:
        return _empty_metrics()
    return evaluate(real, predicted)


def run_arrays(args, values, available, support, upper_bound: float, density: float = 0.0):
    """Run FCUCR-QoS on already loaded arrays.

    ``values`` is indexed as ``[time, user, service]``.  Only values selected
    by ``support`` enter model updates; query labels are read solely by the
    final metrics call.  This function is intentionally silent so tests and
    formal queues do not inherit unbounded client logging.
    """
    values = np.asarray(values)
    available = np.asarray(available, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if values.ndim != 3 or values.shape != available.shape or values.shape != support.shape:
        raise ValueError("values, available and support must be aligned time-user-service tensors")
    if np.any(support & ~available):
        raise ValueError("support cannot contain unavailable entries")
    slices_requested = int(_arg(args, "slices", values.shape[0]))
    if slices_requested <= 0:
        raise ValueError("slices must be positive")
    slices = min(slices_requested, int(values.shape[0]))

    rank = int(_arg(args, "rank", 10))
    learning_rate = float(_arg(args, "learning_rate", 0.01))
    regularization = float(_arg(args, "regularization", 0.001))
    distillation_weight = float(_arg(args, "distillation_weight", 1.0))
    distil_cap = float(_arg(args, "distil_cap", 10.0))
    top_k = int(_arg(args, "top_k", _arg(args, "kb_len", 20)))
    kb_window = int(_arg(args, "kb_window", 0))
    aggregation_mode = str(_arg(args, "aggregation_mode", "equal"))
    local_epochs = int(_arg(args, "local_epochs", 4))
    seed = int(_arg(args, "seed", 0))
    model = FCUCRQoS(
        values.shape[1],
        values.shape[2],
        values.shape[0],
        rank=rank,
        learning_rate=learning_rate,
        regularization=regularization,
        distillation_weight=distillation_weight,
        distil_cap=distil_cap,
        top_k=top_k,
        kb_window=kb_window,
        aggregation_mode=aggregation_mode,
        seed=seed,
    )

    rows = []
    wall_start = time.perf_counter()
    for time_slice in range(slices):
        query_users, query_services = np.nonzero(
            available[time_slice] & ~support[time_slice]
        )
        if len(query_users) == 0:
            # Reject before updating the model so an invalid slice cannot
            # leave behind a partially advanced causal state.  This matches
            # the existing FedAvg runner protocol: every evaluated slice must
            # have a non-empty query estimand.
            raise ValueError(f"time slice {time_slice} has no query observations")

        support_users, support_services = np.nonzero(support[time_slice])
        support_targets = fixed_log_transform(
            values[time_slice, support_users, support_services], upper_bound
        )
        update_start = time.perf_counter()
        update = model.update_slice(
            support_users,
            support_services,
            time_slice,
            support_targets,
            epochs=local_epochs,
        )
        model.assert_finite_state()
        update_seconds = time.perf_counter() - update_start

        snapshot = model.state_copy()
        evaluation_start = time.perf_counter()
        normalized_prediction = model.predict(
            query_users,
            query_services,
            np.full(len(query_users), time_slice, dtype=np.int64),
        )
        predicted = inverse_fixed_log_transform(normalized_prediction, upper_bound)
        metrics = _evaluate(
            values[time_slice, query_users, query_services], predicted
        )
        evaluation_seconds = time.perf_counter() - evaluation_start
        query_read_only = model.state_unchanged(snapshot)

        row = {
            "time_slice": int(time_slice),
            "support_entries": int(len(support_users)),
            "query_entries": int(len(query_users)),
            **update.as_dict(),
            **metrics,
            "update_seconds": float(update_seconds),
            "evaluation_seconds": float(evaluation_seconds),
            "query_read_only": bool(query_read_only),
        }
        if not query_read_only or not all(
            math.isfinite(float(row[key])) for key in SUMMARY_METRICS
        ):
            raise RuntimeError(f"slice {time_slice} failed finite/read-only audit")
        rows.append(row)

    if not rows:
        raise RuntimeError("no slices were executed")
    tasd_enabled = bool(distillation_weight > 0.0 and distil_cap > 0.0)
    tasd_effective = bool(
        tasd_enabled
        and local_epochs >= MIN_TASD_EPOCHS
        and any(
            int(row["time_slice"]) > 0
            and int(row["participating_clients"]) > 0
            and float(row["distillation_loss"]) > 0.0
            for row in rows
        )
    )
    summary = {
        "method": "FCUCR-QoS",
        "qos": getattr(args, "qos", None),
        "density": float(density),
        "seed": seed,
        "time_slices": int(slices),
        "preprocessing": "fixed_log1p",
        "rank": rank,
        "learning_rate": learning_rate,
        "regularization": regularization,
        "local_epochs": local_epochs,
        "distillation_weight": distillation_weight,
        "distil_cap": distil_cap,
        "scale_initialization_strategy": SCALE_INITIALIZATION_STRATEGY,
        "scale_initialization_support_only": True,
        "scale_initialized_clients": int(model.scale_initialization_count),
        "tasd_strategy": TASD_STRATEGY,
        "tasd_enabled": tasd_enabled,
        "tasd_effective": tasd_effective,
        "tasd_min_local_epochs": int(MIN_TASD_EPOCHS),
        "top_k": top_k,
        "kb_window": kb_window,
        "aggregation_mode": aggregation_mode,
        "shared_parameter_bytes": int(model.shared_parameter_bytes),
        "prototype_bytes": int(model.prototype_bytes),
        "prototype_fusion_location": "server",
        "prototype_download": "one_fused_prototype_per_client",
        "communication_accounting": "existing_symmetric_shared_plus_fused_prototype",
        "independent_reconstruction": True,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_license_declared": False,
        "privacy_cryptographic_layer_implemented": False,
        "query_read_only": bool(all(row["query_read_only"] for row in rows)),
        "query_read_only_all": bool(all(row["query_read_only"] for row in rows)),
        "support_loss_reduction_mean": float(
            np.mean(
                [row["support_loss_before"] - row["support_loss_after"] for row in rows]
            )
        ),
        **{key: float(np.mean([row[key] for row in rows])) for key in SUMMARY_METRICS},
        "uplink_bytes": int(sum(row["uplink_bytes"] for row in rows)),
        "downlink_bytes": int(sum(row["downlink_bytes"] for row in rows)),
        "wall_seconds": float(time.perf_counter() - wall_start),
    }
    return rows, summary


def run_stream(args):
    """Load a frozen manifest/data/mask triple and execute the stream."""
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    if manifest.get("qos_type") != args.qos or int(manifest.get("seed")) != int(args.seed):
        raise RuntimeError("manifest identity does not match command")
    preprocessing = manifest.get("preprocessing", {})
    if preprocessing.get("name") != "fixed_log1p":
        raise RuntimeError("FCUCR-QoS requires a fixed_log1p manifest")
    upper_bound = float(preprocessing["upper_bound"])
    values, data_sha256 = load_values(Path(args.data), args.qos)
    if data_sha256 != manifest.get("data_sha256"):
        raise RuntimeError("dataset hash does not match manifest")
    available, support = load_masks(Path(args.mask), args.qos)
    return run_arrays(
        args,
        values,
        available,
        support,
        upper_bound,
        float(manifest.get("density", 0.0)),
    )


def write_artifacts(output: Path, rows, summary):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_time.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal FCUCR-QoS baseline")
    parser.add_argument("--qos", choices=("rt", "tp"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--slices", type=int, default=64)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=0.001)
    parser.add_argument("--local-epochs", type=int, default=4)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--distil-cap", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--kb-window", type=int, default=0)
    parser.add_argument(
        "--aggregation-mode",
        choices=("equal", "sample_weighted"),
        default="equal",
    )
    args = parser.parse_args()
    rows, summary = run_stream(args)
    write_artifacts(args.output, rows, summary)


if __name__ == "__main__":
    main()
