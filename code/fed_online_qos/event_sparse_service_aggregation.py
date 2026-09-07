"""Exact service-state ESSD support inside the FedSRA aggregator.

The existing AMF runner sends a fixed-layout row for every service to every
participating client.  ESSD keeps the numerical aggregation unchanged and
only changes the representation of that server-to-client payload: a fixed
bitmap identifies services whose server state changed in the current support
round, followed by the changed rows.  A client can reconstruct the complete
state by overwriting those rows in its cached state.

No query or future labels are accepted by this module.  The bitmap is derived
from the current support/update mask only.  The byte planner deliberately
chooses the full payload when the bitmap representation would not be smaller.
"""

from __future__ import annotations

import math

import numpy as np


DEFAULT_VALUE_BYTES = 8
BIT_ORDER = "little"


def bitmap_bytes(num_services: int) -> int:
    """Return the exact byte count of a fixed-length service bitmap."""

    if isinstance(num_services, bool) or int(num_services) != num_services or num_services < 0:
        raise ValueError("num_services must be a nonnegative integer")
    return int(math.ceil(int(num_services) / 8.0))


def _normalise_mask(service_indices_or_mask, num_services: int) -> np.ndarray:
    """Convert service indices or a boolean mask into a validated mask."""

    if isinstance(num_services, bool) or int(num_services) != num_services or num_services < 0:
        raise ValueError("num_services must be a nonnegative integer")
    num_services = int(num_services)
    values = np.asarray(service_indices_or_mask)
    if values.ndim != 1:
        raise ValueError("service selection must be one-dimensional")
    if values.dtype == np.bool_:
        if len(values) != num_services:
            raise ValueError("boolean service mask length must equal num_services")
        return values.astype(bool, copy=True)
    if len(values) and not np.issubdtype(values.dtype, np.integer):
        raise TypeError("service indices must be integers or a boolean mask")
    indices = values.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= num_services):
        raise ValueError("service index outside protocol range")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("service indices must be unique")
    mask = np.zeros(num_services, dtype=bool)
    mask[indices] = True
    return mask


def encode_service_bitmap(service_indices_or_mask, num_services: int) -> bytes:
    """Encode a service selection as a deterministic little-endian bitmap."""

    mask = _normalise_mask(service_indices_or_mask, num_services)
    return np.packbits(mask, bitorder=BIT_ORDER).tobytes()


def decode_service_bitmap(payload: bytes, num_services: int) -> np.ndarray:
    """Decode and validate a fixed-length little-endian service bitmap."""

    expected = bitmap_bytes(num_services)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("bitmap payload must be bytes-like")
    raw = bytes(payload)
    if len(raw) != expected:
        raise ValueError(f"bitmap payload must contain exactly {expected} bytes")
    if not expected:
        return np.zeros(0, dtype=bool)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder=BIT_ORDER)
    # Bits above the fixed service range are padding and must remain zero.  A
    # strict check prevents a malformed payload from silently selecting a
    # service outside the protocol universe.
    if np.any(bits[int(num_services):]):
        raise ValueError("bitmap padding bits must be zero")
    return bits[: int(num_services)].astype(bool, copy=True)


def encode_service_state_rows(service_vectors, service_biases, service_errors,
                              service_indices_or_mask):
    """Return ``(bitmap, rows)`` for the selected service state rows.

    ``rows`` uses the runner's fixed double-precision accounting layout:
    ``[dimension service-vector values, service bias, service error]``.  The
    function does not mutate any input array.
    """

    vectors = np.asarray(service_vectors)
    biases = np.asarray(service_biases)
    errors = np.asarray(service_errors)
    if vectors.ndim != 2 or biases.shape != (len(vectors),) or errors.shape != (len(vectors),):
        raise ValueError("service state arrays have incompatible shapes")
    if not (np.all(np.isfinite(vectors)) and np.all(np.isfinite(biases)) and
            np.all(np.isfinite(errors))):
        raise ValueError("service state arrays must be finite")
    mask = _normalise_mask(service_indices_or_mask, len(vectors))
    rows = np.column_stack((vectors[mask], biases[mask], errors[mask])).copy()
    return encode_service_bitmap(mask, len(vectors)), rows


def reconstruct_service_state(previous_vectors, previous_biases, previous_errors,
                              bitmap: bytes, rows):
    """Apply sparse rows to a cached state and return the reconstructed state.

    The returned arrays are copies.  Rows absent from the bitmap are retained
    exactly, which is the protocol invariant used by ESSD.
    """

    vectors = np.asarray(previous_vectors)
    biases = np.asarray(previous_biases)
    errors = np.asarray(previous_errors)
    if vectors.ndim != 2 or biases.shape != (len(vectors),) or errors.shape != (len(vectors),):
        raise ValueError("cached service state arrays have incompatible shapes")
    if not (np.all(np.isfinite(vectors)) and np.all(np.isfinite(biases)) and
            np.all(np.isfinite(errors))):
        raise ValueError("cached service state arrays must be finite")
    mask = decode_service_bitmap(bitmap, len(vectors))
    rows = np.asarray(rows)
    expected_shape = (int(mask.sum()), vectors.shape[1] + 2)
    if rows.shape != expected_shape:
        raise ValueError(f"sparse rows must have shape {expected_shape}")
    if not np.all(np.isfinite(rows)):
        raise ValueError("sparse service rows must be finite")
    result_vectors, result_biases, result_errors = vectors.copy(), biases.copy(), errors.copy()
    result_vectors[mask] = rows[:, :-2]
    result_biases[mask] = rows[:, -2]
    result_errors[mask] = rows[:, -1]
    return result_vectors, result_biases, result_errors


def encode_ephemeral_service_values(values, service_indices_or_mask):
    """Encode current-round per-service values under the shared bitmap.

    Unlike model rows, DCLCSRRC corrections are ephemeral: bitmap-absent
    entries mean exact zero for the current slice and must not be read from a
    client cache.
    """

    values = np.asarray(values)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("ephemeral service values must be one-dimensional and finite")
    mask = _normalise_mask(service_indices_or_mask, len(values))
    return encode_service_bitmap(mask, len(values)), values[mask].copy()


def reconstruct_ephemeral_service_values(bitmap: bytes, selected_values, num_services: int,
                                         *, dtype=np.float32) -> np.ndarray:
    """Reconstruct an ephemeral service vector with zero outside the bitmap."""

    mask = decode_service_bitmap(bitmap, num_services)
    selected_values = np.asarray(selected_values)
    if selected_values.shape != (int(mask.sum()),) or not np.all(np.isfinite(selected_values)):
        raise ValueError("selected ephemeral values do not match the bitmap")
    result = np.zeros(int(num_services), dtype=dtype)
    result[mask] = selected_values
    return result


def plan_service_downlink(participants: int, num_services: int, dimension: int,
                          transmitted_rows: int, *, value_bytes: int = DEFAULT_VALUE_BYTES,
                          extra_row_bytes: int = 0, extra_per_participant_bytes: int = 0,
                          enabled: bool = True) -> dict:
    """Choose full or bitmap-plus-row service downlink for one time slice.

    ``extra_row_bytes`` accounts for an optional per-service block sharing the
    same bitmap.  For DCLCSRRC this is one float32 correction (4 bytes).  The
    optional per-participant term supports calibration payloads with a shared
    scalar while keeping the base ESSD path simple.
    """

    integer_values = (participants, num_services, dimension, transmitted_rows,
                      value_bytes, extra_row_bytes, extra_per_participant_bytes)
    if any(isinstance(value, bool) or int(value) != value for value in integer_values):
        raise ValueError("communication dimensions must be integers")
    participants, num_services, dimension, transmitted_rows, value_bytes = map(
        int, (participants, num_services, dimension, transmitted_rows, value_bytes))
    extra_row_bytes, extra_per_participant_bytes = map(
        int, (extra_row_bytes, extra_per_participant_bytes))
    if min(participants, num_services, dimension, transmitted_rows,
           value_bytes, extra_row_bytes, extra_per_participant_bytes) < 0:
        raise ValueError("communication dimensions must be nonnegative")
    if value_bytes <= 0 or transmitted_rows > num_services:
        raise ValueError("invalid service downlink dimensions")

    model_row_bytes = (dimension + 2) * value_bytes
    bitmap = bitmap_bytes(num_services)
    full_per_participant = num_services * (model_row_bytes + extra_row_bytes) + extra_per_participant_bytes
    sparse_per_participant = bitmap + transmitted_rows * (model_row_bytes + extra_row_bytes) + extra_per_participant_bytes
    full_bytes = participants * full_per_participant
    sparse_bytes = participants * sparse_per_participant
    use_sparse = bool(enabled and sparse_bytes < full_bytes)
    mode = "sparse" if use_sparse else "full"
    chosen_bytes = sparse_bytes if use_sparse else full_bytes
    chosen_extra_bytes = participants * (
        transmitted_rows * extra_row_bytes + extra_per_participant_bytes
        if use_sparse else num_services * extra_row_bytes + extra_per_participant_bytes)
    return {
        "mode": mode,
        "bitmap_bytes": bitmap if use_sparse else 0,
        "transmitted_rows": (transmitted_rows if use_sparse else (num_services if participants else 0)),
        "full_bytes": int(full_bytes),
        "sparse_bytes": int(sparse_bytes),
        "chosen_bytes": int(chosen_bytes),
        "saved_bytes": int(full_bytes - chosen_bytes),
        "chosen_extra_bytes": int(chosen_extra_bytes),
        "model_row_bytes": int(model_row_bytes),
        "extra_row_bytes": int(extra_row_bytes),
        "extra_per_participant_bytes": int(extra_per_participant_bytes),
    }


__all__ = [
    "BIT_ORDER",
    "DEFAULT_VALUE_BYTES",
    "bitmap_bytes",
    "decode_service_bitmap",
    "encode_ephemeral_service_values",
    "encode_service_bitmap",
    "encode_service_state_rows",
    "plan_service_downlink",
    "reconstruct_ephemeral_service_values",
    "reconstruct_service_state",
]
