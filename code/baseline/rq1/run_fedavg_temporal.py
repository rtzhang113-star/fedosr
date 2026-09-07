"""Run one causal full-model FedAvg stream and write minimal artifacts."""
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(FED_CODE) not in sys.path:
    sys.path.insert(0, str(FED_CODE))

from methods.fedavg import (FullAMFModel, aggregate_sparse_deltas,
                            client_sgd_delta)  # noqa: E402
from run_fed_amf import evaluate, load_masks, load_values  # noqa: E402


FIELDS = (
    "time_slice", "support_entries", "query_entries", "mae", "rmse", "mre", "npre",
    "update_seconds", "evaluation_seconds", "uplink_bytes", "downlink_bytes", "query_read_only",
)
SUMMARY_METRICS = ("mae", "rmse", "mre", "npre", "update_seconds", "evaluation_seconds")


def fixed_log_transform(values, upper_bound: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)) or not np.all((values > 0) & (values < upper_bound)):
        raise ValueError("QoS values must be finite and strictly inside the fixed upper bound")
    return (np.log1p(values.astype(np.float64)) / np.log1p(float(upper_bound))).astype(np.float32)


def inverse_fixed_log_transform(values, upper_bound: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.expm1(np.clip(values, 0.0, 1.0) * np.log1p(float(upper_bound)))


def _state_unchanged(before: tuple[np.ndarray, ...], model: FullAMFModel) -> bool:
    return all(np.array_equal(left, right) for left, right in zip(before, model.state_arrays()))


def run_arrays(args, values, available, support, upper_bound: float, density: float = 0.0):
    """Run the stream on already-loaded arrays.

    Keeping this pure-array entry point makes the protocol audits able to
    perturb future labels without touching the dataset loader or writing a
    temporary copy of the 160-MB tensor.
    """
    values = np.asarray(values, dtype=np.float32)
    available = np.asarray(available, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if values.shape != available.shape or values.shape != support.shape:
        raise ValueError("data, availability, and support shapes must match")
    if np.any(support & ~available):
        raise ValueError("support must be a subset of available")
    if not np.all(np.isfinite(upper_bound)) or upper_bound <= 0:
        raise ValueError("upper_bound must be positive and finite")

    num_times, num_users, num_services = values.shape
    slices = min(int(args.slices), num_times)
    if slices <= 0:
        raise ValueError("at least one time slice is required")
    model = FullAMFModel.random(num_users, num_services, int(args.dimension), int(args.seed))
    rows: list[dict] = []
    wall_start = time.perf_counter()
    for time_slice in range(slices):
        update_start = time.perf_counter()
        support_users, support_services = np.nonzero(support[time_slice])
        client_deltas = []
        for user in np.unique(support_users):
            selected = support_users == user
            services = support_services[selected]
            # np.nonzero is row-major, and this explicit check is part of the
            # frozen deterministic protocol contract.
            if np.any(services[1:] < services[:-1]):
                raise RuntimeError("support service order is not ascending")
            targets = fixed_log_transform(values[time_slice, user, services], upper_bound)
            client_deltas.append(client_sgd_delta(
                model, int(user), services, targets,
                eta=float(args.eta), l2=float(args.l2), local_epochs=int(args.local_epochs),
            ))
        model = aggregate_sparse_deltas(model, client_deltas) if client_deltas else model
        update_seconds = time.perf_counter() - update_start

        query_users, query_services = np.nonzero(available[time_slice] & ~support[time_slice])
        if len(query_users) == 0:
            raise RuntimeError(f"slice {time_slice} has no query entries")
        before = tuple(np.array(array, copy=True) for array in model.state_arrays())
        evaluation_start = time.perf_counter()
        normalized_prediction = model.predict_normalized(query_users, query_services)
        predicted = np.clip(
            inverse_fixed_log_transform(normalized_prediction, upper_bound),
            0.0, float(upper_bound),
        )
        metrics = evaluate(values[time_slice, query_users, query_services], predicted)
        evaluation_seconds = time.perf_counter() - evaluation_start
        query_read_only = _state_unchanged(before, model)
        row = {
            "time_slice": int(time_slice),
            "support_entries": int(len(support_users)),
            "query_entries": int(len(query_users)),
            **metrics,
            "update_seconds": float(update_seconds),
            "evaluation_seconds": float(evaluation_seconds),
            "uplink_bytes": int(len(client_deltas) * model.parameter_count * 4),
            "downlink_bytes": int(len(client_deltas) * model.parameter_count * 4),
            "query_read_only": bool(query_read_only),
        }
        if not query_read_only or not all(math.isfinite(float(row[key])) for key in SUMMARY_METRICS):
            raise RuntimeError(f"slice {time_slice} failed finite/read-only audit")
        rows.append(row)

    summary = {
        "qos": args.qos,
        "density": float(density),
        "seed": int(args.seed),
        "time_slices": int(slices),
        **{key: float(np.mean([row[key] for row in rows])) for key in SUMMARY_METRICS},
        "uplink_bytes": int(sum(row["uplink_bytes"] for row in rows)),
        "downlink_bytes": int(sum(row["downlink_bytes"] for row in rows)),
        "wall_seconds": float(time.perf_counter() - wall_start),
    }
    return rows, summary


def run_stream(args):
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest["qos_type"] != args.qos or int(manifest["seed"]) != int(args.seed):
        raise RuntimeError("manifest identity does not match command")
    density = float(manifest["density"])
    upper_bound = float(manifest["preprocessing"]["upper_bound"])
    values, data_sha256 = load_values(Path(args.data), args.qos)
    if data_sha256 != manifest["data_sha256"]:
        raise RuntimeError("dataset hash does not match manifest")
    available, support = load_masks(Path(args.mask), args.qos)
    return run_arrays(args, values, available, support, upper_bound, density)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--qos", choices=("rt", "tp"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--slices", type=int, default=64)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.8)
    parser.add_argument("--l2", type=float, default=3e-4)
    parser.add_argument("--local-epochs", type=int, default=1)
    args = parser.parse_args()
    rows, summary = run_stream(args)
    write_artifacts(args.output, rows, summary)


if __name__ == "__main__":
    main()
