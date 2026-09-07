"""Run the causal PerFed tensor-factorization adaptation.

The Li et al. (2022) paper evaluates static tensor completion after random
deletions.  This runner deliberately keeps that model separate from the
project's frozen causal protocol: one time slice is updated from ``support``
only and the disjoint ``query`` entries are read only for prediction and
evaluation.  The implementation in :mod:`methods.perfed` owns the numerical
CP/EASGD update.  The runner owns data/mask provenance, deterministic client
partitioning, causal ordering, metrics, and artifacts.

The preferred method interface is::

    model = PersonalizedFederatedTensorFactorization(...)
    model.sync_round(client_batches, epochs=local_epochs)
    model.predict(users, services, time_slices, client_ids=client_ids)
    model.state_arrays()

``client_batches`` is a list of ``(client_id, users, services, time_slices,
normalized_values)`` records.  For compatibility with a small independently
reconstructed method, ``update_round`` and a sequential ``update_client``
fallback are also accepted; the summary records which path was used.  Query
labels never enter either update path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FED_CODE = ROOT / "code" / "fed_online_qos"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(FED_CODE) not in sys.path:
    sys.path.insert(0, str(FED_CODE))

from run_fed_amf import evaluate, load_masks, load_values  # noqa: E402


FIELDS = (
    "time_slice", "support_entries", "query_entries", "active_clients",
    "mae", "rmse", "mre", "npre", "update_seconds", "evaluation_seconds",
    "uplink_bytes", "downlink_bytes", "query_read_only",
)
SUMMARY_METRICS = ("mae", "rmse", "mre", "npre", "update_seconds", "evaluation_seconds")
MODEL_CLASS_NAMES = (
    "PersonalizedFederatedTensorFactorization",
    "PerFedTensorFactorization",
    "FederatedPersonalizedTensorFactorization",
)


def _arg(args: Any, name: str, default: Any) -> Any:
    """Read an argument from argparse or a small audit ``Namespace``."""
    return getattr(args, name, default)


def _partition_users(num_users: int, num_edge_servers: int, seed: int) -> np.ndarray:
    """Return a deterministic IID user-to-edge-server assignment.

    The assignment uses only the public user count and the configured seed. It
    does not inspect values, masks, query labels, or any future slice.
    """
    if int(num_users) <= 0 or int(num_edge_servers) <= 0:
        raise ValueError("num_users and num_edge_servers must be positive")
    rng = np.random.default_rng(int(seed))
    assignment = np.empty(int(num_users), dtype=np.int64)
    permutation = rng.permutation(int(num_users))
    for client_id, users in enumerate(np.array_split(permutation, int(num_edge_servers))):
        assignment[users] = int(client_id)
    return assignment


def _select_constructor_kwargs(
    constructor: Any,
    *,
    num_users: int,
    num_services: int,
    num_times: int,
    num_edge_servers: int,
    rank: int,
    learning_rate: float,
    gamma: float,
    seed: int,
    user_to_client: np.ndarray,
) -> dict[str, Any]:
    """Map the runner's frozen names to the method's explicit signature.

    The alias handling keeps the runner usable with the independently written
    method while still failing clearly if required model dimensions are not
    accepted.  It does not silently pass an unsupported optimization option.
    """
    try:
        parameters = inspect.signature(constructor).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    values = {
        "num_users": num_users,
        "num_services": num_services,
        "num_times": num_times,
        "num_edge_servers": num_edge_servers,
        "num_clients": num_edge_servers,
        "rank": rank,
        "learning_rate": learning_rate,
        "lr": learning_rate,
        "gamma": gamma,
        "coupling": gamma,
        "seed": seed,
        "user_to_client": user_to_client,
        "user_clients": user_to_client,
        "client_assignment": user_to_client,
        "client_of_user": user_to_client,
    }
    aliases = {
        "num_users": ("num_users", "users"),
        "num_services": ("num_services", "services"),
        "num_times": ("num_times", "times"),
        "num_edge_servers": ("num_edge_servers", "num_clients", "num_servers", "clients"),
        "rank": ("rank", "dimension"),
        "learning_rate": ("learning_rate", "lr", "eta"),
        "gamma": ("gamma", "coupling", "regularization"),
        "seed": ("seed", "random_seed"),
        "user_to_client": (
            "user_to_client", "user_clients", "client_assignment", "client_of_user",
            "user_partition",
        ),
    }
    selected: dict[str, Any] = {}
    for canonical, names in aliases.items():
        chosen = next((name for name in names if accepts_kwargs or name in parameters), None)
        if chosen is not None:
            selected[chosen] = values[chosen]
    required = ("num_users", "num_services", "num_times", "rank")
    missing = [name for name in required if not accepts_kwargs and not any(name in parameters for name in aliases[name])]
    if missing:
        raise TypeError(
            "PerFed constructor must expose tensor dimensions/rank; missing " + ", ".join(missing)
        )
    return selected


def _load_perfed_class() -> Any:
    try:
        from methods import perfed as module  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised before method installation
        raise ImportError(
            "code/baseline/rq1/methods/perfed.py is required by the PerFed runner"
        ) from exc
    for name in MODEL_CLASS_NAMES:
        candidate = getattr(module, name, None)
        if candidate is not None:
            return candidate
    raise AttributeError(
        "methods.perfed must define one of: " + ", ".join(MODEL_CLASS_NAMES)
    )


def _make_model(
    num_users: int,
    num_services: int,
    num_times: int,
    num_edge_servers: int,
    rank: int,
    learning_rate: float,
    gamma: float,
    seed: int,
    user_to_client: np.ndarray,
) -> Any:
    constructor = _load_perfed_class()
    kwargs = _select_constructor_kwargs(
        constructor,
        num_users=num_users,
        num_services=num_services,
        num_times=num_times,
        num_edge_servers=num_edge_servers,
        rank=rank,
        learning_rate=learning_rate,
        gamma=gamma,
        seed=seed,
        user_to_client=user_to_client,
    )
    model = constructor(**kwargs)
    # A method may prefer to initialize its own partition.  If it accepts a
    # setter, enforce the runner's deterministic assignment; otherwise the
    # constructor receives it through one of the aliases above.
    setter = getattr(model, "set_user_to_client", None)
    if callable(setter):
        setter(np.asarray(user_to_client, dtype=np.int64).copy())
    elif hasattr(model, "user_to_client"):
        setattr(model, "user_to_client", np.asarray(user_to_client, dtype=np.int64).copy())
    return model


def fixed_log_transform(values: np.ndarray, upper_bound: float) -> np.ndarray:
    """Map raw QoS values into the frozen normalized ``fixed_log1p`` domain.

    The preprocessing is part of the experiment protocol, rather than a
    method-specific choice.  In particular, TP values must not be normalized
    by ``value / 10000``: the CP factors operate in the same log1p space as
    the other baselines.
    """
    values = np.asarray(values, dtype=np.float32)
    upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("upper_bound must be positive and finite")
    if not np.all(np.isfinite(values)) or not np.all((values > 0.0) & (values < upper_bound)):
        raise ValueError("QoS values must be finite and strictly inside the fixed upper bound")
    return (np.log1p(values.astype(np.float64)) / np.log1p(upper_bound)).astype(np.float32)


def inverse_fixed_log_transform(values: np.ndarray, upper_bound: float) -> np.ndarray:
    """Map normalized PerFed predictions back to the raw QoS scale."""
    values = np.asarray(values, dtype=np.float64)
    upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("upper_bound must be positive and finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("normalized predictions must be finite")
    return np.expm1(np.clip(values, 0.0, 1.0) * np.log1p(upper_bound))


def _state_arrays(model: Any) -> tuple[np.ndarray, ...]:
    """Collect mutable factor arrays for the query read-only audit."""
    state_method = getattr(model, "state_arrays", None)
    if callable(state_method):
        arrays = tuple(np.asarray(item) for item in state_method())
        if arrays:
            return arrays
    # The explicit fallback supports a small method implementation while
    # keeping the audit strict.  Duplicate references are included once.
    names = (
        "private_user_factors", "user_factors", "local_user_factors",
        "local_service_factors", "local_time_factors",
        "service_factors", "time_factors", "shared_service_factors",
        "shared_time_factors", "global_service_factors", "global_time_factors",
    )
    arrays: list[np.ndarray] = []
    seen: set[int] = set()
    for name in names:
        item = getattr(model, name, None)
        if isinstance(item, np.ndarray) and id(item) not in seen:
            arrays.append(item)
            seen.add(id(item))
    if not arrays:
        raise AttributeError("PerFed model must expose state_arrays() or factor ndarrays")
    return tuple(arrays)


def _state_unchanged(before: Sequence[np.ndarray], model: Any) -> bool:
    after = _state_arrays(model)
    return len(before) == len(after) and all(np.array_equal(left, right) for left, right in zip(before, after))


def _parameter_count(model: Any, num_services: int, num_times: int, rank: int) -> int:
    value = getattr(model, "public_parameter_count", None)
    if callable(value):
        value = value()
    if value is not None:
        count = int(value)
        if count > 0:
            return count
    # One upload/download consists of the local/global public service and time
    # factors.  This is the paper's wire contract; private user factors stay at
    # the edge server.
    return int((int(num_services) + int(num_times)) * int(rank))


def _dtype_bytes(model: Any) -> int:
    for array in _state_arrays(model):
        if np.issubdtype(array.dtype, np.floating):
            return int(array.dtype.itemsize)
    # Historical PerFed smoke used NumPy's default float64 initialization.
    return 8


def _result_value(result: Any, names: Sequence[str], default: int = 0) -> int:
    if isinstance(result, Mapping):
        for name in names:
            if name in result and result[name] is not None:
                return int(result[name])
        for nested_name in ("client_results", "results", "updates"):
            nested = result.get(nested_name)
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
                return int(sum(_result_value(item, names, 0) for item in nested))
    for name in names:
        value = getattr(result, name, None)
        if value is not None:
            return int(value)
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        return int(sum(_result_value(item, names, 0) for item in result))
    return int(default)


def _call_update_round(model: Any, batches: list[dict], local_epochs: int) -> tuple[int, int, str]:
    """Update all participating edge servers and return communication bytes."""
    sync_round = getattr(model, "sync_round", None)
    if callable(sync_round):
        try:
            parameters = inspect.signature(sync_round).parameters
        except (TypeError, ValueError):
            parameters = {}
        kwargs: dict[str, Any] = {}
        if "epochs" in parameters or not parameters:
            kwargs["epochs"] = int(local_epochs)
        elif "local_epochs" in parameters:
            kwargs["local_epochs"] = int(local_epochs)
        records = [
            (
                int(batch["client_id"]), batch["users"], batch["services"],
                batch["time_slices"], batch["values"],
            )
            for batch in batches
        ]
        result = sync_round(records, **kwargs)
        return (
            _result_value(result, ("uploaded_bytes", "uplink_bytes")),
            _result_value(result, ("downloaded_bytes", "downlink_bytes")),
            "sync_round",
        )

    update_round = getattr(model, "update_round", None)
    if callable(update_round):
        try:
            parameters = inspect.signature(update_round).parameters
        except (TypeError, ValueError):
            parameters = {}
        kwargs: dict[str, Any] = {}
        if "epochs" in parameters or not parameters:
            kwargs["epochs"] = int(local_epochs)
        elif "local_epochs" in parameters:
            kwargs["local_epochs"] = int(local_epochs)
        result = update_round(batches, **kwargs)
        return (
            _result_value(result, ("uploaded_bytes", "uplink_bytes")),
            _result_value(result, ("downloaded_bytes", "downlink_bytes")),
            "update_round",
        )

    update_client = getattr(model, "update_client", None)
    if not callable(update_client):
        raise AttributeError("PerFed model must define sync_round(), update_round(), or update_client()")
    uploaded = 0
    downloaded = 0
    for batch in batches:
        try:
            parameters = inspect.signature(update_client).parameters
        except (TypeError, ValueError):
            parameters = {}
        kwargs: dict[str, Any] = {}
        if "epochs" in parameters or not parameters:
            kwargs["epochs"] = int(local_epochs)
        elif "local_epochs" in parameters:
            kwargs["local_epochs"] = int(local_epochs)
        result = update_client(
            int(batch["client_id"]), batch["users"], batch["services"],
            batch["time_slices"], batch["values"], **kwargs,
        )
        uploaded += _result_value(result, ("uploaded_bytes", "uplink_bytes"))
        downloaded += _result_value(result, ("downloaded_bytes", "downlink_bytes"))
    aggregate = getattr(model, "aggregate", None)
    if callable(aggregate):
        # ``update_client`` returns public payloads and records the participating
        # clients internally.  The aggregate call is the required Eq. (18)
        # center step and does not add another factor transmission.
        aggregate()
        return uploaded, downloaded, "update_client+aggregate_fallback"
    return uploaded, downloaded, "update_client_sequential_fallback"


def _call_predict(model: Any, client_id: int, users, services, time_slices) -> np.ndarray:
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise AttributeError("PerFed model must define predict()")
    try:
        parameters = inspect.signature(predict).parameters
    except (TypeError, ValueError):
        parameters = {}
    names = tuple(parameters)
    if "client_ids" in parameters:
        result = predict(
            users, services, time_slices,
            client_ids=np.full(len(np.asarray(users)), int(client_id), dtype=np.int64),
        )
    elif names and names[0] in {"client", "client_id", "server", "server_id"}:
        result = predict(int(client_id), users, services, time_slices)
    elif "client_id" in parameters:
        result = predict(users, services, time_slices, client_id=int(client_id))
    else:
        result = predict(users, services, time_slices)
    prediction = np.asarray(result, dtype=np.float64)
    if prediction.shape != np.asarray(users).shape:
        raise ValueError("PerFed predict() must return one normalized value per query entry")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("PerFed predictions must be finite")
    return prediction


def run_arrays(args: Any, values, available, support, upper_bound: float, density: float = 0.0):
    """Run the causal PerFed protocol on pre-loaded arrays.

    This entry point is intentionally independent of disk loading.  Audit
    tests can perturb query/future labels in ``values`` and verify that model
    state and predictions stay unchanged because only support values are ever
    sent to the update method.
    """
    values = np.asarray(values, dtype=np.float32)
    available = np.asarray(available, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if values.shape != available.shape or values.shape != support.shape:
        raise ValueError("data, availability, and support shapes must match")
    if values.ndim != 3:
        raise ValueError("values must have shape [time, user, service]")
    if np.any(support & ~available):
        raise ValueError("support must be a subset of available")
    if not np.isfinite(float(upper_bound)) or float(upper_bound) <= 0.0:
        raise ValueError("upper_bound must be positive and finite")

    num_times, num_users, num_services = values.shape
    requested_slices = int(_arg(args, "slices", num_times))
    slices = min(requested_slices, num_times)
    if slices <= 0:
        raise ValueError("at least one time slice is required")
    seed = int(_arg(args, "seed", 42))
    rank = int(_arg(args, "rank", 5))
    learning_rate = float(_arg(args, "learning_rate", 0.001))
    gamma = float(_arg(args, "gamma", 0.01))
    local_epochs = int(_arg(args, "local_epochs", 1))
    num_edge_servers = int(_arg(args, "num_edge_servers", 6))
    if rank <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("rank must be positive and learning_rate must be finite/positive")
    if not np.isfinite(gamma) or gamma < 0.0 or local_epochs <= 0:
        raise ValueError("gamma must be finite/non-negative and local_epochs positive")
    if num_edge_servers <= 0:
        raise ValueError("num_edge_servers must be positive")

    user_to_client = _partition_users(num_users, num_edge_servers, seed)
    model = _make_model(
        num_users, num_services, num_times, num_edge_servers, rank,
        learning_rate, gamma, seed, user_to_client,
    )
    rows: list[dict] = []
    update_paths: set[str] = set()
    wall_start = time.perf_counter()
    for time_slice in range(slices):
        update_start = time.perf_counter()
        support_users, support_services = np.nonzero(support[time_slice])
        batches: list[dict] = []
        for client_id in range(num_edge_servers):
            selected = user_to_client[support_users] == int(client_id)
            if not np.any(selected):
                continue
            users = np.asarray(support_users[selected], dtype=np.int64)
            services = np.asarray(support_services[selected], dtype=np.int64)
            if np.any(users[1:] < users[:-1]) or (
                len(users) > 1 and np.any((users[1:] == users[:-1]) & (services[1:] < services[:-1]))
            ):
                raise RuntimeError("support rows are not in deterministic user/service order")
            targets = fixed_log_transform(values[time_slice, users, services], upper_bound)
            batches.append({
                "client_id": int(client_id),
                "users": users,
                "services": services,
                "time_slices": np.full(len(users), time_slice, dtype=np.int64),
                "values": targets,
            })
        if batches:
            uploaded, downloaded, update_path = _call_update_round(model, batches, local_epochs)
            update_paths.add(update_path)
        else:
            uploaded = downloaded = 0
            update_paths.add("no_support")
        update_seconds = time.perf_counter() - update_start

        query_users, query_services = np.nonzero(available[time_slice] & ~support[time_slice])
        if len(query_users) == 0:
            raise RuntimeError(f"slice {time_slice} has no query entries")
        before = tuple(np.array(array, copy=True) for array in _state_arrays(model))
        evaluation_start = time.perf_counter()
        predicted_normalized = np.empty(len(query_users), dtype=np.float64)
        for client_id in range(num_edge_servers):
            selected = user_to_client[query_users] == int(client_id)
            if np.any(selected):
                predicted_normalized[selected] = _call_predict(
                    model, client_id, query_users[selected], query_services[selected],
                    np.full(int(np.sum(selected)), time_slice, dtype=np.int64),
                )
        if not np.all(np.isfinite(predicted_normalized)):
            raise RuntimeError(f"slice {time_slice} produced non-finite predictions")
        predicted = np.clip(
            inverse_fixed_log_transform(predicted_normalized, upper_bound),
            0.0,
            float(upper_bound),
        )
        metrics = evaluate(values[time_slice, query_users, query_services], predicted)
        evaluation_seconds = time.perf_counter() - evaluation_start
        query_read_only = _state_unchanged(before, model)
        if uploaded == 0 and downloaded == 0 and batches:
            public_count = _parameter_count(model, num_services, num_times, rank)
            bytes_per_scalar = _dtype_bytes(model)
            fallback_bytes = int(len(batches) * public_count * bytes_per_scalar)
            uploaded = downloaded = fallback_bytes
        row = {
            "time_slice": int(time_slice),
            "support_entries": int(len(support_users)),
            "query_entries": int(len(query_users)),
            "active_clients": int(len(batches)),
            **metrics,
            "update_seconds": float(update_seconds),
            "evaluation_seconds": float(evaluation_seconds),
            "uplink_bytes": int(uploaded),
            "downlink_bytes": int(downloaded),
            "query_read_only": bool(query_read_only),
        }
        if not query_read_only or not all(math.isfinite(float(row[key])) for key in SUMMARY_METRICS):
            raise RuntimeError(f"slice {time_slice} failed finite/read-only audit")
        rows.append(row)

    assignment_hash = hashlib.sha256(user_to_client.astype(np.int64).tobytes()).hexdigest()
    summary = {
        "method": "PerFed",
        "protocol": "causal_streaming_current_slice_completion",
        "adaptation": "causal_online_support_query",
        "qos": _arg(args, "qos", "unknown"),
        "density": float(density),
        "seed": seed,
        "time_slices": int(slices),
        "num_users": int(num_users),
        "num_services": int(num_services),
        "num_times": int(num_times),
        "num_edge_servers": num_edge_servers,
        "rank": rank,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "local_epochs": local_epochs,
        "client_partition": "iid_random_user_split",
        "client_assignment_sha256": assignment_hash,
        "normalization": "fixed_log1p",
        "normalization_formula": "log1p(values) / log1p(manifest.preprocessing.upper_bound)",
        "inverse_normalization": "fixed_expm1",
        "inverse_normalization_formula": "expm1(normalized * log1p(manifest.preprocessing.upper_bound))",
        "initialization": "rank_scaled_positive_uniform",
        "initialization_is_reproduction_assumption": True,
        "update_paths": sorted(update_paths),
        "gamma_is_reproduction_assumption": True,
        "learning_rate_is_reproduction_assumption": True,
        **{key: float(np.mean([row[key] for row in rows])) for key in SUMMARY_METRICS},
        "support_entries": int(sum(row["support_entries"] for row in rows)),
        "query_entries": int(sum(row["query_entries"] for row in rows)),
        "uplink_bytes": int(sum(row["uplink_bytes"] for row in rows)),
        "downlink_bytes": int(sum(row["downlink_bytes"] for row in rows)),
        "wall_seconds": float(time.perf_counter() - wall_start),
    }
    return rows, summary


def run_stream(args: Any) -> tuple[list[dict], dict]:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest["qos_type"] != args.qos or int(manifest["seed"]) != int(args.seed):
        raise RuntimeError("manifest identity does not match command")
    density = float(manifest["density"])
    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, Mapping) or preprocessing.get("name") != "fixed_log1p":
        raise RuntimeError(
            "PerFed requires manifest.preprocessing.name == 'fixed_log1p'"
        )
    upper_bound = float(preprocessing["upper_bound"])
    values, data_sha256 = load_values(Path(args.data), args.qos)
    if data_sha256 != manifest["data_sha256"]:
        raise RuntimeError("dataset hash does not match manifest")
    available, support = load_masks(Path(args.mask), args.qos)
    if values.shape != available.shape or available.shape != support.shape:
        raise RuntimeError("data/mask shape mismatch")
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
    parser = argparse.ArgumentParser(
        description="Run the causal PerFed tensor-factorization baseline adaptation."
    )
    parser.add_argument("--qos", choices=("rt", "tp"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slices", type=int, default=64)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument(
        "--learning-rate", type=float, default=0.001,
        help="reconstruction assumption; the paper does not report a final learning rate",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.01,
        help="reconstruction assumption; the paper sweeps gamma but does not report a final value",
    )
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument(
        "--num-edge-servers", type=int, default=6,
        help="causal adaptation default; the paper's static study uses K=3,6,10",
    )
    args = parser.parse_args()
    rows, summary = run_stream(args)
    write_artifacts(args.output, rows, summary)


if __name__ == "__main__":
    main()
