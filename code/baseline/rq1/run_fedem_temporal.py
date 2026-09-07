"""Run the causal FedEM-QoS baseline on one frozen WSDream stream.

The runner adapts a FedEM-style personalized model-merging mechanism to the
project's online protocol.  Each user is a client.  A time slice consumes
only its support entries, then predicts the disjoint query entries read-only.
The method implementation is in :mod:`methods.fedem`; this file owns data
provenance, fixed-log1p preprocessing, metrics, causal ordering, and artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FED_CODE = ROOT / "code" / "fed_online_qos"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(FED_CODE) not in sys.path:
    sys.path.insert(0, str(FED_CODE))

from methods.fedem import FedEMQoSModel  # noqa: E402
from run_fed_amf import evaluate, load_masks, load_values  # noqa: E402


FIELDS = (
    "time_slice", "support_entries", "query_entries", "active_clients", "aggregation_clients",
    "mae", "rmse", "mre", "npre", "update_seconds", "evaluation_seconds",
    "uplink_bytes", "downlink_bytes", "query_read_only",
)
SUMMARY_METRICS = (
    "mae", "rmse", "mre", "npre", "update_seconds", "evaluation_seconds",
)


def _arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def fixed_log_transform(values: Any, upper_bound: float) -> np.ndarray:
    """Map raw positive QoS values to the frozen fixed-log1p domain."""

    values = np.asarray(values, dtype=np.float32)
    upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("upper_bound must be finite and positive")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values >= upper_bound):
        raise ValueError("QoS values must be finite and strictly inside the upper bound")
    return (np.log1p(values.astype(np.float64)) / np.log1p(upper_bound)).astype(np.float32)


def inverse_fixed_log_transform(values: Any, upper_bound: float) -> np.ndarray:
    """Map normalized predictions back to raw QoS units."""

    values = np.asarray(values, dtype=np.float64)
    upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("upper_bound must be finite and positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("normalized predictions must be finite")
    return np.expm1(np.clip(values, 0.0, 1.0) * np.log1p(upper_bound))


def _state_unchanged(before: Any, model: FedEMQoSModel) -> bool:
    """Audit arrays plus round and private adapter state after query prediction."""

    return model.state_unchanged(before)


def _make_model(args: Any, num_users: int, num_services: int, num_times: int) -> FedEMQoSModel:
    return FedEMQoSModel(
        num_users=num_users,
        num_services=num_services,
        num_times=num_times,
        rank=int(_arg(args, "rank", _arg(args, "dimension", 16))),
        learning_rate=float(_arg(args, "learning_rate", _arg(args, "eta", 0.1))),
        regularization=float(_arg(args, "regularization", 0.0003)),
        clients_sample_ratio=float(_arg(args, "clients_sample_ratio", 1.0)),
        agg_clients_ratio=float(_arg(args, "agg_clients_ratio", 0.1)),
        agg_mode=str(_arg(args, "agg_mode", "both")),
        sim_mode=str(_arg(args, "sim_mode", "l2")),
        similarity_source=str(_arg(args, "similarity_source", "support_signature")),
        alpha=float(_arg(args, "alpha", 0.9)),
        adapter_learning_rate=float(_arg(args, "adapter_learning_rate", 0.1)),
        adapter_regularization=float(_arg(args, "adapter_regularization", 0.0003)),
        local_epochs=int(_arg(args, "local_epochs", 1)),
        batch_size=int(_arg(args, "batch_size", 2147483647)),
        seed=int(_arg(args, "seed", 42)),
    )


def run_arrays(
    args: Any,
    values: Any,
    available: Any,
    support: Any,
    upper_bound: float,
    density: float = 0.0,
) -> tuple[list[dict], dict]:
    """Run FedEM-QoS on already-loaded arrays.

    This pure-array entry point is intentionally used by semantic tests: the
    values outside support can be perturbed without changing any online action.
    """

    values = np.asarray(values, dtype=np.float32)
    available = np.asarray(available, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if values.ndim != 3 or values.shape != available.shape or values.shape != support.shape:
        raise ValueError("values, available and support must be aligned [time, user, service] tensors")
    if np.any(support & ~available):
        raise ValueError("support must be a subset of available")
    if not np.isfinite(float(upper_bound)) or float(upper_bound) <= 0.0:
        raise ValueError("upper_bound must be finite and positive")
    requested_slices = int(_arg(args, "slices", values.shape[0]))
    slices = min(requested_slices, values.shape[0])
    if slices <= 0:
        raise ValueError("at least one time slice is required")
    local_epochs = int(_arg(args, "local_epochs", 1))
    if local_epochs <= 0:
        raise ValueError("local_epochs must be positive")
    batch_size = int(_arg(args, "batch_size", 2147483647))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    num_times, num_users, num_services = values.shape
    model = _make_model(args, num_users, num_services, num_times)
    rows: list[dict] = []
    wall_start = time.perf_counter()
    for time_slice in range(slices):
        update_start = time.perf_counter()
        support_users, support_services = np.nonzero(support[time_slice])
        batches = []
        for user in np.unique(support_users):
            selected = support_users == user
            services = support_services[selected]
            if np.any(services[1:] < services[:-1]):
                raise RuntimeError("support services are not in deterministic ascending order")
            targets = fixed_log_transform(values[time_slice, user, services], upper_bound)
            users = np.full(len(services), int(user), dtype=np.int64)
            times = np.full(len(services), int(time_slice), dtype=np.int64)
            batches.append((int(user), users, services.astype(np.int64), times, targets))
        if batches:
            update = model.sync_round(batches, epochs=local_epochs)
            uplink_bytes = int(update["uplink_bytes"])
            downlink_bytes = int(update["downlink_bytes"])
            active_clients = int(update["participating_clients"])
            aggregation_clients = int(update["aggregation_clients"])
        else:
            uplink_bytes = 0
            downlink_bytes = 0
            active_clients = 0
            aggregation_clients = 0
        update_seconds = time.perf_counter() - update_start

        query_users, query_services = np.nonzero(available[time_slice] & ~support[time_slice])
        if len(query_users) == 0:
            raise RuntimeError(f"slice {time_slice} has no query entries")
        before = model.state_copy()
        evaluation_start = time.perf_counter()
        normalized = model.predict(
            query_users.astype(np.int64),
            query_services.astype(np.int64),
            np.full(len(query_users), int(time_slice), dtype=np.int64),
        )
        predicted = np.clip(
            inverse_fixed_log_transform(normalized, upper_bound),
            0.0,
            float(upper_bound),
        )
        metrics = evaluate(values[time_slice, query_users, query_services], predicted)
        evaluation_seconds = time.perf_counter() - evaluation_start
        query_read_only = _state_unchanged(before, model)
        row = {
            "time_slice": int(time_slice),
            "support_entries": int(len(support_users)),
            "query_entries": int(len(query_users)),
            "active_clients": active_clients,
            "aggregation_clients": aggregation_clients,
            **metrics,
            "update_seconds": float(update_seconds),
            "evaluation_seconds": float(evaluation_seconds),
            "uplink_bytes": uplink_bytes,
            "downlink_bytes": downlink_bytes,
            "query_read_only": bool(query_read_only),
        }
        if not query_read_only or not all(
            math.isfinite(float(row[key])) for key in SUMMARY_METRICS
        ):
            raise RuntimeError(f"slice {time_slice} failed finite/read-only audit")
        rows.append(row)

    summary = {
        "method": "FedEM-QoS",
        "protocol": "causal_streaming_current_slice_completion",
        "adaptation": "support_only_user_level_personalized_service_merging",
        "qos": _arg(args, "qos", "unknown"),
        "density": float(density),
        "seed": int(_arg(args, "seed", 42)),
        "time_slices": int(slices),
        "num_users": int(num_users),
        "num_services": int(num_services),
        "num_times": int(num_times),
        "rank": int(model.rank),
        "learning_rate": float(model.learning_rate),
        "regularization": float(model.regularization),
        "batch_size": int(model.batch_size),
        "clients_sample_ratio": float(model.clients_sample_ratio),
        "agg_clients_ratio": float(model.agg_clients_ratio),
        "agg_mode": model.agg_mode,
        "sim_mode": model.sim_mode,
        "alpha": float(model.alpha),
        "adapter_learning_rate": float(model.adapter_learning_rate),
        "adapter_regularization": float(model.adapter_regularization),
        "adapter_input_dim": int(model.adapter_input_dim),
        "adapter_layer_sizes": list(model.adapter_layer_sizes),
        "adapter_parameter_count_per_client": int(model.adapter_parameter_count),
        "local_epochs": int(local_epochs),
        "client_mode": "one_user_per_client",
        "normalization": "fixed_log1p",
        "normalization_formula": "log1p(values) / log1p(manifest.preprocessing.upper_bound)",
        "inverse_normalization": "fixed_expm1",
        "similarity_source": (
            "current_support_residual_service_signature_only"
            if model.similarity_source == "support_signature"
            else "complete_persistent_service_factor_table_pca_first_component_l2_or_cos"
        ),
        "adaptive_merge": "official MLP omega=sigmoid(MLP([q_local;q_server-q_local])); same omega merges service factor and bias",
        "aggregation": "PCA-first-component similarity with official avg/sim/both modes; both uses simplex projection of p + alpha*similarity",
        "communication_accounting": "official dense QoS adaptation accounting; service factors/biases plus participant adapter state upload, dense service model download, user state excluded",
        "private_state": "user_factor_and_user_bias_and_client_adapter",
        "persistent_local_state": "per_user_service_factor_and_service_bias",
        "query_read_only_all": bool(all(row["query_read_only"] for row in rows)),
        **{key: float(np.mean([row[key] for row in rows])) for key in SUMMARY_METRICS},
        "support_entries": int(sum(row["support_entries"] for row in rows)),
        "query_entries": int(sum(row["query_entries"] for row in rows)),
        "uplink_bytes": int(sum(row["uplink_bytes"] for row in rows)),
        "downlink_bytes": int(sum(row["downlink_bytes"] for row in rows)),
        "wall_seconds": float(time.perf_counter() - wall_start),
    }
    return rows, summary


def run_stream(args: Any) -> tuple[list[dict], dict]:
    """Load one frozen manifest/data/mask triple and run the stream."""

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    if manifest.get("qos_type") != args.qos or int(manifest.get("seed", -1)) != int(args.seed):
        raise RuntimeError("manifest identity does not match command")
    preprocessing = manifest.get("preprocessing", {})
    if preprocessing.get("name") != "fixed_log1p":
        raise RuntimeError("FedEM-QoS requires manifest.preprocessing.name == 'fixed_log1p'")
    upper_bound = float(preprocessing["upper_bound"])
    values, data_sha256 = load_values(Path(args.data), args.qos)
    if data_sha256 != manifest.get("data_sha256"):
        raise RuntimeError("dataset hash does not match manifest")
    available, support = load_masks(Path(args.mask), args.qos)
    if values.shape != available.shape or available.shape != support.shape:
        raise RuntimeError("data/mask shape mismatch")
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
    parser = argparse.ArgumentParser(description="Run the causal FedEM-QoS baseline.")
    parser.add_argument("--qos", choices=("rt", "tp"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slices", type=int, default=64)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--regularization", type=float, default=0.0003)
    parser.add_argument("--clients-sample-ratio", type=float, default=1.0)
    parser.add_argument("--agg-clients-ratio", type=float, default=0.1)
    parser.add_argument("--agg-mode", choices=("avg", "sim", "both"), default="both")
    parser.add_argument("--sim-mode", choices=("l2", "cos"), default="l2")
    parser.add_argument(
        "--similarity-source", choices=("support_signature", "persistent_pca"),
        default="support_signature",
    )
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--adapter-learning-rate", type=float, default=0.1)
    parser.add_argument("--adapter-regularization", type=float, default=0.0003)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2147483647)
    args = parser.parse_args()
    rows, summary = run_stream(args)
    write_artifacts(args.output, rows, summary)


if __name__ == "__main__":
    main()
