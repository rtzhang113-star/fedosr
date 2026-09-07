"""Generate reproducible TP support/query masks for the causal protocol.

The WS-DREAM ``tpRate`` file is stored as ``user service time value``.  TP
uses an ex-ante fixed ``0 < TP < 10000`` validity rule and a fixed
``log1p(x) / log1p(10000)`` transform.  Raw files contain a small number of
duplicate ``(user, service, time)`` keys; the protocol counts each location
once and applies the AMF-compatible last-record-wins rule.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


NUM_USERS, NUM_SERVICES, NUM_TIMES = 142, 4500, 64
TP_UPPER_EXCLUSIVE = 10000.0
TP_LOG_UPPER = 10000.0


def fixed_log_transform(values, upper_bound=TP_LOG_UPPER):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or not np.all((values > 0) & (values < upper_bound)):
        raise ValueError("TP values must be finite and lie strictly inside the fixed upper bound")
    return np.log1p(values) / np.log1p(upper_bound)


def inverse_fixed_log_transform(values, upper_bound=TP_LOG_UPPER):
    return np.expm1(np.asarray(values, dtype=float) * np.log1p(upper_bound))


class CausalDriftScorer:
    """Per-client JS drift from current support versus prior support history."""

    def __init__(self, users, bins=32, min_entries=20):
        self.edges = np.linspace(0.0, 1.0, bins + 1)
        self.history = np.zeros((users, bins), dtype=np.int64)
        self.min_entries = min_entries

    def update_and_score(self, user, normalized_values):
        current_counts = np.histogram(normalized_values, bins=self.edges)[0]
        current, previous = current_counts.astype(float), self.history[user].astype(float)
        score = np.nan
        if current.sum() >= self.min_entries and previous.sum() >= self.min_entries:
            current /= current.sum(); previous /= previous.sum()
            mean = .5 * (current + previous)
            current_nonzero, previous_nonzero = current > 0, previous > 0
            score = .5 * (np.sum(current[current_nonzero] * np.log(current[current_nonzero] / mean[current_nonzero])) +
                          np.sum(previous[previous_nonzero] * np.log(previous[previous_nonzero] / mean[previous_nonzero])))
        self.history[user] += current_counts
        return float(score)


def drift_groups(scores):
    groups = np.full(len(scores), "unscored", dtype=object)
    eligible = np.isfinite(scores)
    if eligible.any():
        low, high = np.quantile(scores[eligible], (.25, .75))
        groups[eligible] = "middle"
        groups[eligible & (scores <= low)] = "low"
        groups[eligible & (scores >= high)] = "high"
    return groups


def _scan(data_file):
    """Load the TP availability mask and audit duplicate/raw-record counts."""
    available = np.zeros((NUM_TIMES, NUM_USERS, NUM_SERVICES), dtype=bool)
    seen = np.zeros_like(available)
    # 0 = no duplicate seen, 1 = duplicate seen and values agree,
    # 2 = duplicate seen with at least one conflicting value.
    duplicate_marked = np.zeros_like(available, dtype=np.uint8)
    first_values = np.full(available.shape, np.nan, dtype=np.float32)
    digest, invalid, raw_records = hashlib.sha256(), 0, 0
    duplicate_records = 0
    duplicate_keys = 0
    conflicting_keys = 0
    with data_file.open("rb") as file:
        for line in file:
            raw_records += 1
            digest.update(line)
            user, service, time_slice, raw_value = line.split()
            user, service, time_slice = int(user), int(service), int(time_slice)
            value = float(raw_value)
            if not (0 <= user < NUM_USERS and 0 <= service < NUM_SERVICES and 0 <= time_slice < NUM_TIMES):
                raise ValueError(f"TP key out of range: {(user, service, time_slice)}")
            if seen[time_slice, user, service]:
                duplicate_records += 1
                key_index = (time_slice, user, service)
                # Match the runner's float32 storage semantics. A direct
                # float64-to-float32 comparison falsely marks identical
                # decimal rows as conflicts due only to representation.
                differs = np.float32(value) != first_values[key_index]
                if duplicate_marked[key_index] == 0:
                    duplicate_keys += 1
                    duplicate_marked[key_index] = 2 if differs else 1
                    if differs:
                        conflicting_keys += 1
                elif differs and duplicate_marked[key_index] != 2:
                    duplicate_marked[key_index] = 2
                    conflicting_keys += 1
            else:
                seen[time_slice, user, service] = True
                first_values[time_slice, user, service] = value
            # Assignment on every row is intentional: invalid duplicate rows
            # clear an earlier valid value under last-record-wins semantics.
            is_valid = np.isfinite(value) and 0 < value < TP_UPPER_EXCLUSIVE
            available[time_slice, user, service] = is_valid
            if not is_valid:
                invalid += 1
    return available, invalid, digest.hexdigest(), {
        "raw_records": raw_records,
        "unique_raw_keys": int(seen.sum()),
        "duplicate_raw_records": duplicate_records,
        "duplicate_key_count": duplicate_keys,
        "duplicate_conflicting_keys": conflicting_keys,
        "duplicate_policy": "last-record-wins; one unique (user, service, time) location",
    }


def load_available_entries(data_file):
    available, invalid, sha256, _ = _scan(data_file)
    return available, invalid, sha256


def make_support_mask(available, density, seed):
    support = np.zeros_like(available)
    for time_slice in range(NUM_TIMES):
        for user in range(NUM_USERS):
            services = np.flatnonzero(available[time_slice, user])
            size = int(density * len(services))
            if size:
                rng = np.random.default_rng(np.random.SeedSequence([seed, time_slice, user]))
                support[time_slice, user, rng.choice(services, size, replace=False)] = True
    return support


def verify(available, support, density):
    query = available & ~support
    assert not np.any(support & ~available)
    assert np.array_equal(support.sum(2), (density * available.sum(2)).astype(int))
    assert available.sum() == support.sum() + query.sum()
    return query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        values = np.array([0.001, 1.0, 6726.8335])
        assert np.allclose(values, inverse_fixed_log_transform(fixed_log_transform(values)))
        print("self-test passed")
        return

    available, invalid, sha256, duplicate_stats = _scan(args.data_file)
    support = make_support_mask(available, args.density, args.seed)
    query = verify(available, support, args.density)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"tp_density_{args.density:.2f}_seed_{args.seed}"
    np.savez_compressed(
        args.output_dir / f"{tag}_masks.npz",
        shape=np.array(available.shape, dtype=np.int32),
        available_bits=np.packbits(available.ravel()),
        support_bits=np.packbits(support.ravel()),
        support_counts=support.sum(2, dtype=np.int32),
        query_counts=query.sum(2, dtype=np.int32),
    )
    manifest = {
        "task": "causal_streaming_current_slice_completion",
        "qos_type": "tp", "qos_name": "TP", "qos_unit": "kbps",
        "data_file": str(args.data_file.resolve()), "data_sha256": sha256,
        "shape_time_user_service": list(available.shape), "density": args.density, "seed": args.seed,
        "validity_rule": f"0 < TP < {TP_UPPER_EXCLUSIVE:g}",
        "invalid_raw_records_excluded": invalid,
        **duplicate_stats,
        "available_entries": int(available.sum()), "support_entries": int(support.sum()), "query_entries": int(query.sum()),
        "preprocessing": {"name": "fixed_log1p", "upper_bound": TP_LOG_UPPER,
                           "fit_data": "none; fixed ex-ante domain bound, so query and future labels cannot affect it"},
        "verification": "support and query are disjoint and exactly partition valid entries per user/time slice",
    }
    (args.output_dir / f"{tag}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
