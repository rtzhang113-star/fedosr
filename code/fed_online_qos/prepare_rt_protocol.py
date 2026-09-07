"""Generate reproducible RT support/query masks for the causal protocol."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

NUM_USERS, NUM_SERVICES, NUM_TIMES = 142, 4500, 64
RT_UPPER_EXCLUSIVE = 19.9


def fixed_log_transform(values, upper_bound=20.0):
    values = np.asarray(values, dtype=float)
    assert np.all((values > 0) & (values < upper_bound))
    return np.log1p(values) / np.log1p(upper_bound)


def inverse_fixed_log_transform(values, upper_bound=20.0):
    return np.expm1(np.asarray(values) * np.log1p(upper_bound))


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


def load_available_entries(data_file):
    available, invalid, sha256, _ = _scan(data_file)
    return available, invalid, sha256


def _scan(data_file):
    """Scan RT data and return availability plus duplicate-policy audit.

    RT keeps an entry available when at least one duplicate raw record is
    valid, matching the historical RT loader.  Duplicate raw rows and
    conflicting values are nevertheless counted and written to the manifest.
    """
    available = np.zeros((NUM_TIMES, NUM_USERS, NUM_SERVICES), dtype=bool)
    total = NUM_TIMES * NUM_USERS * NUM_SERVICES
    seen = np.zeros(total, dtype=bool)
    first_values = np.full(total, np.nan, dtype=np.float32)
    digest, invalid = hashlib.sha256(), 0
    raw_records = duplicate_raw_records = duplicate_key_count = duplicate_conflicting_keys = 0
    duplicate_keys_seen: set[int] = set()
    conflicting_keys_seen: set[int] = set()
    with data_file.open("rb") as file:
        for line in file:
            digest.update(line)
            user, service, time_slice, value = line.split()
            raw_records += 1
            user, service, time_slice = int(user), int(service), int(time_slice)
            value = float(value)
            flat = (time_slice * NUM_USERS + user) * NUM_SERVICES + service
            if seen[flat]:
                duplicate_raw_records += 1
                if flat not in duplicate_keys_seen:
                    duplicate_keys_seen.add(flat)
                    duplicate_key_count += 1
                if not np.isclose(first_values[flat], value, rtol=0.0, atol=1e-6, equal_nan=True):
                    conflicting_keys_seen.add(flat)
            else:
                seen[flat] = True
                first_values[flat] = np.float32(value)
            if 0 < value < RT_UPPER_EXCLUSIVE:
                available[time_slice, user, service] = True
            else:
                invalid += 1
    return available, invalid, digest.hexdigest(), {
        "raw_records": raw_records,
        "unique_raw_keys": int(seen.sum()),
        "duplicate_raw_records": duplicate_raw_records,
        "duplicate_key_count": duplicate_key_count,
        "duplicate_conflicting_keys": len(conflicting_keys_seen),
        "duplicate_policy": "any_valid_record_wins; invalid duplicate records are excluded but do not clear a valid RT key",
    }


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


def audit_mask(mask_file, manifest_file):
    """Decode and independently verify a generated NPZ/manifest pair."""
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8-sig"))
    with np.load(mask_file) as data:
        shape = tuple(int(item) for item in data["shape"])
        total = int(np.prod(shape))
        available = np.unpackbits(data["available_bits"])[:total].reshape(shape).astype(bool)
        support = np.unpackbits(data["support_bits"])[:total].reshape(shape).astype(bool)
    if shape != (NUM_TIMES, NUM_USERS, NUM_SERVICES):
        raise AssertionError(f"RT mask shape mismatch: {shape}")
    query = available & ~support
    if np.any(support & ~available):
        raise AssertionError("support is not a subset of available")
    counts = {
        "available_entries": int(available.sum()),
        "support_entries": int(support.sum()),
        "query_entries": int(query.sum()),
    }
    expected = tuple(int(manifest.get(name, -1)) for name in counts)
    if tuple(counts.values()) != expected:
        raise AssertionError(f"RT NPZ counts mismatch: actual={counts} manifest={expected}")
    if int(manifest.get("seed", -1)) < 0 or abs(float(manifest.get("density", -1.0)) - 0.10) > 1e-12:
        raise AssertionError("RT mask seed/density metadata")
    if manifest.get("qos_type") != "rt" or manifest.get("validity_rule") != f"0 < RT < {RT_UPPER_EXCLUSIVE}":
        raise AssertionError("RT mask contract metadata")
    return {"shape": list(shape), **counts, "support_subset_of_available": True, "seed": int(manifest["seed"]), "density": float(manifest["density"]), "validity_rule": manifest["validity_rule"], "data_sha256": manifest.get("data_sha256")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        values = np.array([0.001, 1.0, 19.899])
        assert np.allclose(values, inverse_fixed_log_transform(fixed_log_transform(values)))
        print("self-test passed")
        return

    available, invalid, sha256, stats = _scan(args.data_file)
    support = make_support_mask(available, args.density, args.seed)
    query = verify(available, support, args.density)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"rt_density_{args.density:.2f}_seed_{args.seed}"
    np.savez_compressed(
        args.output_dir / f"{tag}_masks.npz",
        shape=np.array(available.shape, dtype=np.int32),
        available_bits=np.packbits(available.ravel()),
        support_bits=np.packbits(support.ravel()),
        support_counts=support.sum(2, dtype=np.int32),
        query_counts=query.sum(2, dtype=np.int32),
    )
    manifest = {
        "task": "causal_streaming_current_slice_completion", "qos_type": "rt",
        "data_file": str(args.data_file.resolve()), "data_sha256": sha256,
        "shape_time_user_service": list(available.shape), "density": args.density, "seed": args.seed,
        "validity_rule": f"0 < RT < {RT_UPPER_EXCLUSIVE}", "invalid_raw_records_excluded": invalid,
        "available_entries": int(available.sum()), "support_entries": int(support.sum()), "query_entries": int(query.sum()),
        "preprocessing": {"name": "fixed_log1p", "upper_bound": 20.0, "fit_data": "none; fixed published/domain bound, so query and future labels cannot affect it"},
        "verification": "support and query are disjoint and exactly partition valid entries per user/time slice",
        "duplicate_policy": stats["duplicate_policy"], "duplicate_audit": stats,
        "mask_schema": "shape+packed available_bits+packed support_bits+support_counts+query_counts",
    }
    (args.output_dir / f"{tag}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
