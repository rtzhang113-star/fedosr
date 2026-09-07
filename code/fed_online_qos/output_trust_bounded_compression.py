"""Deterministic output-trust-bounded delta compression.

The codec is deliberately independent of the predictor runner.  It supports
true symmetric signed INT16 quantization with a transmitted scale, private
error-feedback residuals, and a raw floating-point fallback when the
quantization error could exceed the supplied output-trust slack.  No random
state or label data is involved.
"""

from dataclasses import dataclass

import numpy as np


INT16_MIN = -32768
INT16_MAX = 32767
FLAG_FP = 0
FLAG_INT16 = 1
FLAG_BYTES = 1
DEFAULT_PARAMETER_WIDTH = 4
_BOUND_TOLERANCE_MULTIPLIER = 16.0


def _validate_parameter_width(parameter_width):
    if isinstance(parameter_width, bool) or int(parameter_width) not in (4, 8):
        raise ValueError("parameter_width must be exactly 4 or 8 bytes")
    return int(parameter_width)


def _wire_dtype(parameter_width):
    return np.dtype("<f4" if _validate_parameter_width(parameter_width) == 4 else "<f8")


def _finite_vector(values, name, *, copy=True):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values.copy() if copy else values


def _stored_scale(scale, parameter_width):
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative")
    stored = float(np.asarray([scale], dtype=_wire_dtype(parameter_width))[0])
    if not np.isfinite(stored) or stored < 0:
        raise FloatingPointError("scale is not representable at parameter_width")
    return stored


@dataclass(frozen=True)
class PayloadByteStats:
    """Exact byte accounting for one delta payload."""

    flag_bytes: int
    scale_bytes: int
    data_bytes: int
    payload_bytes: int
    parameter_width: int
    element_count: int
    compressed: bool

    def as_dict(self):
        return {
            "flag_bytes": self.flag_bytes,
            "scale_bytes": self.scale_bytes,
            "data_bytes": self.data_bytes,
            "payload_bytes": self.payload_bytes,
            "parameter_width": self.parameter_width,
            "element_count": self.element_count,
            "compressed": self.compressed,
        }


@dataclass(frozen=True)
class DeltaPacket:
    """Wire payload plus the metadata needed for deterministic decoding."""

    payload: bytes
    element_count: int
    parameter_width: int
    compressed: bool
    scale: float
    flag: int

    @property
    def byte_stats(self):
        return payload_byte_stats(
            self.element_count,
            parameter_width=self.parameter_width,
            compressed=self.compressed,
        )

    @property
    def payload_bytes(self):
        return len(self.payload)


@dataclass
class ErrorFeedbackState:
    """Private residual state for one service-vector and bias delta stream."""

    vector: np.ndarray
    bias: float = 0.0

    def __post_init__(self):
        self.vector = _finite_vector(self.vector, "residual vector")
        self.bias = float(self.bias)
        if not np.isfinite(self.bias):
            raise ValueError("residual bias must be finite")

    @classmethod
    def zeros(cls, dimension):
        if isinstance(dimension, bool) or int(dimension) < 0:
            raise ValueError("dimension must be a nonnegative integer")
        return cls(np.zeros(int(dimension), dtype=np.float64), 0.0)

    def copy(self):
        return ErrorFeedbackState(self.vector.copy(), self.bias)


@dataclass(frozen=True)
class CompressionResult:
    """Selected payload, reconstructed delta and trust-bound diagnostics."""

    packet: DeltaPacket
    reconstructed_vector: np.ndarray
    reconstructed_bias: float
    residual_vector: np.ndarray
    residual_bias: float
    quantization_bound: float
    selected_bound: float
    slack: float
    quantization_safe: bool
    used_fallback: bool

    @property
    def payload_bytes(self):
        return self.packet.payload_bytes

    @property
    def byte_stats(self):
        return self.packet.byte_stats


@dataclass(frozen=True)
class BatchCompressionResult:
    """Vectorized result for a batch of service-vector/bias deltas."""

    reconstructed_vectors: np.ndarray
    reconstructed_biases: np.ndarray
    residual_vectors: np.ndarray
    residual_biases: np.ndarray
    safe_mask: np.ndarray
    fallback_mask: np.ndarray
    bounds: np.ndarray
    selected_bounds: np.ndarray
    payload_bytes: int
    compressed_payload_bytes: int
    fallback_payload_bytes: int
    parameter_width: int
    element_count: int

    @property
    def quantization_bounds(self):
        return self.bounds

    @property
    def compressed_rows(self):
        return int(np.count_nonzero(self.safe_mask))

    @property
    def fallback_rows(self):
        return int(np.count_nonzero(self.fallback_mask))


@dataclass(frozen=True)
class ClientBlockCompressionResult:
    """Output-trust compression result for a client block.

    The leading dimension is the client dimension ``B`` and the second
    dimension is the selected service-row dimension ``R``.  This API has no
    private residual state: callers pass the target-minus-cache deltas as
    ``vector_deltas``/``bias_deltas`` and use the reconstructed arrays to
    update their caches.  The per-client byte arrays are provided so the
    caller can aggregate communication without rebuilding packet objects.
    """

    reconstructed_vectors: np.ndarray
    reconstructed_biases: np.ndarray
    safe_mask: np.ndarray
    fallback_mask: np.ndarray
    bounds: np.ndarray
    selected_bounds: np.ndarray
    payload_bytes: int
    compressed_payload_bytes: int
    fallback_payload_bytes: int
    payload_bytes_by_client: np.ndarray
    compressed_payload_bytes_by_client: np.ndarray
    fallback_payload_bytes_by_client: np.ndarray
    parameter_width: int
    element_count: int

    @property
    def compressed_rows(self):
        return int(np.count_nonzero(self.safe_mask))

    @property
    def fallback_rows(self):
        return int(np.count_nonzero(self.fallback_mask))

    @property
    def clients(self):
        return int(self.safe_mask.shape[0])

    @property
    def rows_per_client(self):
        return int(self.safe_mask.shape[1])


def payload_byte_stats(element_count, *, parameter_width=DEFAULT_PARAMETER_WIDTH,
                       compressed=True):
    """Return exact payload bytes, including one flag and one scale field.

    INT16 payloads use ``1 + parameter_width + 2*n`` bytes.  The fallback
    payload uses ``1 + parameter_width + parameter_width*n`` bytes; with the
    default width of 4 this is the FP32 fallback wire format.
    """
    width = _validate_parameter_width(parameter_width)
    if isinstance(element_count, bool) or int(element_count) < 0:
        raise ValueError("element_count must be a nonnegative integer")
    count = int(element_count)
    data_width = 2 if compressed else width
    data_bytes = count * data_width
    return PayloadByteStats(
        flag_bytes=FLAG_BYTES,
        scale_bytes=width,
        data_bytes=data_bytes,
        payload_bytes=FLAG_BYTES + width + data_bytes,
        parameter_width=width,
        element_count=count,
        compressed=bool(compressed),
    )


def quantize_signed_int16(values, *, parameter_width=DEFAULT_PARAMETER_WIDTH):
    """Quantize finite values to signed INT16 and return ``(q, scale)``.

    The scale is symmetric and transmitted at the requested floating-point
    width.  ``np.rint`` gives deterministic nearest-even rounding.
    """
    values = _finite_vector(values, "delta")
    width = _validate_parameter_width(parameter_width)
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    scale = _stored_scale(0.0 if max_abs == 0.0 else max_abs / INT16_MAX, width)
    if scale == 0.0:
        quantized = np.zeros(values.shape, dtype=np.dtype("<i2"))
    else:
        quantized = np.rint(values / scale)
        quantized = np.clip(quantized, INT16_MIN, INT16_MAX).astype(np.dtype("<i2"))
    return quantized, scale


def dequantize_signed_int16(quantized, scale):
    """Dequantize signed INT16 values using the transmitted scale."""
    quantized = np.asarray(quantized)
    if quantized.ndim != 1 or not np.issubdtype(quantized.dtype, np.integer):
        raise ValueError("quantized values must be a one-dimensional integer vector")
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative")
    result = quantized.astype(np.float64) * float(scale)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("dequantized values are non-finite")
    return result


def encode_int16_delta(values, *, parameter_width=DEFAULT_PARAMETER_WIDTH):
    """Encode a finite delta vector as a deterministic signed INT16 payload."""
    values = _finite_vector(values, "delta")
    width = _validate_parameter_width(parameter_width)
    quantized, scale = quantize_signed_int16(values, parameter_width=width)
    header = bytes((FLAG_INT16,)) + np.asarray([scale], dtype=_wire_dtype(width)).tobytes()
    payload = header + quantized.tobytes()
    expected = payload_byte_stats(len(values), parameter_width=width, compressed=True).payload_bytes
    if len(payload) != expected:
        raise AssertionError("INT16 payload byte accounting mismatch")
    return DeltaPacket(payload, len(values), width, True, scale, FLAG_INT16)


def encode_fp32_fallback(values, *, parameter_width=DEFAULT_PARAMETER_WIDTH):
    """Encode raw floating-point values; width 4 is the FP32 fallback."""
    values = _finite_vector(values, "delta")
    width = _validate_parameter_width(parameter_width)
    wire_values = np.asarray(values, dtype=_wire_dtype(width))
    if not np.all(np.isfinite(wire_values)):
        raise FloatingPointError("fallback conversion produced non-finite values")
    header = bytes((FLAG_FP,)) + np.zeros(1, dtype=_wire_dtype(width)).tobytes()
    payload = header + wire_values.tobytes()
    expected = payload_byte_stats(len(values), parameter_width=width, compressed=False).payload_bytes
    if len(payload) != expected:
        raise AssertionError("fallback payload byte accounting mismatch")
    return DeltaPacket(payload, len(values), width, False, 0.0, FLAG_FP)


def _packet_parts(payload, element_count, parameter_width):
    width = _validate_parameter_width(parameter_width)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    if isinstance(element_count, bool) or int(element_count) < 0:
        raise ValueError("element_count must be a nonnegative integer")
    raw = bytes(payload)
    count = int(element_count)
    header_bytes = FLAG_BYTES + width
    if len(raw) < header_bytes:
        raise ValueError("payload is shorter than its flag and scale header")
    flag = raw[0]
    if flag not in (FLAG_FP, FLAG_INT16):
        raise ValueError("payload flag is not recognized")
    scale = float(np.frombuffer(raw[FLAG_BYTES:header_bytes], dtype=_wire_dtype(width), count=1)[0])
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("payload scale must be finite and nonnegative")
    compressed = flag == FLAG_INT16
    expected = payload_byte_stats(count, parameter_width=width, compressed=compressed).payload_bytes
    if len(raw) != expected:
        raise ValueError(f"payload has {len(raw)} bytes; expected {expected}")
    return raw, count, width, flag, scale, header_bytes


def decode_delta(packet_or_payload, element_count=None, *, parameter_width=None):
    """Decode either a :class:`DeltaPacket` or a raw payload to float64."""
    if isinstance(packet_or_payload, DeltaPacket):
        packet = packet_or_payload
        count = packet.element_count if element_count is None else element_count
        width = packet.parameter_width if parameter_width is None else parameter_width
        if int(count) != packet.element_count or _validate_parameter_width(width) != packet.parameter_width:
            raise ValueError("decode metadata does not match DeltaPacket")
        raw, count, width, flag, scale, header_bytes = _packet_parts(
            packet.payload, count, width)
    else:
        if element_count is None or parameter_width is None:
            raise ValueError("raw payload decoding requires element_count and parameter_width")
        raw, count, width, flag, scale, header_bytes = _packet_parts(
            packet_or_payload, element_count, parameter_width)
    data = raw[header_bytes:]
    if flag == FLAG_INT16:
        quantized = np.frombuffer(data, dtype=np.dtype("<i2"), count=count)
        return dequantize_signed_int16(quantized, scale)
    decoded = np.frombuffer(data, dtype=_wire_dtype(width), count=count).astype(np.float64)
    if not np.all(np.isfinite(decoded)):
        raise FloatingPointError("decoded fallback values are non-finite")
    return decoded


def output_trust_error_bound(prediction_vector, epsilon_vector, epsilon_bias):
    """Return ``||p_u||_2 ||epsilon_v||_2 + |epsilon_b|``."""
    prediction_vector = _finite_vector(prediction_vector, "prediction vector")
    epsilon_vector = _finite_vector(epsilon_vector, "vector error")
    epsilon_bias = float(epsilon_bias)
    if not np.isfinite(epsilon_bias):
        raise ValueError("bias error must be finite")
    if prediction_vector.shape != epsilon_vector.shape:
        raise ValueError("prediction and vector-error dimensions must match")
    bound = float(np.linalg.norm(prediction_vector) * np.linalg.norm(epsilon_vector) +
                  abs(epsilon_bias))
    if not np.isfinite(bound):
        raise FloatingPointError("output-trust error bound is non-finite")
    return bound


def output_trust_safe(prediction_vector, epsilon_vector, epsilon_bias, slack):
    """Return whether the output-trust error bound fits the supplied slack."""
    slack = float(slack)
    if not np.isfinite(slack) or slack < 0:
        raise ValueError("slack must be finite and nonnegative")
    bound = output_trust_error_bound(prediction_vector, epsilon_vector, epsilon_bias)
    tolerance = _BOUND_TOLERANCE_MULTIPLIER * np.finfo(float).eps * max(1.0, slack, bound)
    return bool(bound <= slack + tolerance)


def bounded_output_trust_compress_batch(
        vector_deltas, bias_deltas, prediction_vector, slack,
        residual_vector, residual_bias, *,
        parameter_width=DEFAULT_PARAMETER_WIDTH):
    """Vectorized output-trust compression for ``R`` delta rows.

    ``vector_deltas`` has shape ``(R, D)`` and ``bias_deltas`` has shape
    ``(R,)``.  The dense residual arrays are updated in place.  All rows use
    the same deterministic per-row INT16 protocol as
    :func:`bounded_output_trust_compress`; this implementation performs the
    scale, quantization, dequantization, trust test and fallback selection as
    array operations and never constructs one packet object per row.
    """
    vector_deltas = np.asarray(vector_deltas, dtype=np.float64)
    bias_deltas = np.asarray(bias_deltas, dtype=np.float64)
    prediction_vector = _finite_vector(prediction_vector, "prediction vector")
    residual_vector_array = np.asarray(residual_vector)
    residual_bias_array = np.asarray(residual_bias)
    if vector_deltas.ndim != 2 or not np.all(np.isfinite(vector_deltas)):
        raise ValueError("vector_deltas must be a finite two-dimensional array")
    if bias_deltas.ndim != 1 or not np.all(np.isfinite(bias_deltas)):
        raise ValueError("bias_deltas must be a finite vector")
    rows, dimension = vector_deltas.shape
    if bias_deltas.shape != (rows,):
        raise ValueError("bias_deltas must have one entry per vector delta row")
    if prediction_vector.shape != (dimension,):
        raise ValueError("prediction_vector dimension must match vector deltas")
    if residual_vector_array.shape != (rows, dimension):
        raise ValueError("residual_vector must match vector_deltas shape")
    if residual_bias_array.shape != (rows,):
        raise ValueError("residual_bias must have one entry per delta row")
    if (not np.issubdtype(residual_vector_array.dtype, np.floating) or
            not np.issubdtype(residual_bias_array.dtype, np.floating)):
        raise TypeError("residual arrays must use floating-point dtypes")
    if not residual_vector_array.flags.writeable or not residual_bias_array.flags.writeable:
        raise ValueError("residual arrays must be writable for in-place feedback")
    if not np.all(np.isfinite(residual_vector_array)) or not np.all(np.isfinite(residual_bias_array)):
        raise ValueError("residual arrays must contain only finite values")
    slack = float(slack)
    if not np.isfinite(slack) or slack < 0:
        raise ValueError("slack must be finite and nonnegative")
    width = _validate_parameter_width(parameter_width)

    working_vectors = vector_deltas + residual_vector_array
    working_biases = bias_deltas + residual_bias_array
    if not np.all(np.isfinite(working_vectors)) or not np.all(np.isfinite(working_biases)):
        raise FloatingPointError("working deltas are non-finite")
    combined = np.concatenate((working_vectors, working_biases[:, None]), axis=1)
    max_abs = np.max(np.abs(combined), axis=1) if rows else np.empty(0, dtype=np.float64)
    scales = max_abs / INT16_MAX
    scales = np.asarray(scales, dtype=_wire_dtype(width)).astype(np.float64)
    if not np.all(np.isfinite(scales)):
        raise FloatingPointError("batch scale is not representable at parameter_width")
    quantized = np.zeros(combined.shape, dtype=np.dtype("<i2"))
    nonzero_scale = scales > 0.0
    if np.any(nonzero_scale):
        quantized[nonzero_scale] = np.clip(
            np.rint(combined[nonzero_scale] / scales[nonzero_scale, None]),
            INT16_MIN, INT16_MAX).astype(np.dtype("<i2"))
    int16_reconstructed = quantized.astype(np.float64) * scales[:, None]
    int16_epsilon = int16_reconstructed - combined
    prediction_norm = float(np.linalg.norm(prediction_vector))
    quantization_bounds = (prediction_norm * np.linalg.norm(int16_epsilon[:, :-1], axis=1) +
                           np.abs(int16_epsilon[:, -1]))
    if not np.all(np.isfinite(quantization_bounds)):
        raise FloatingPointError("batch output-trust bounds are non-finite")
    tolerance = _BOUND_TOLERANCE_MULTIPLIER * np.finfo(float).eps * np.maximum(
        1.0, np.maximum(slack, quantization_bounds))
    safe_mask = quantization_bounds <= slack + tolerance
    fallback_mask = ~safe_mask

    wire_dtype = _wire_dtype(width)
    # Keep the INT16 reconstruction as the default and materialize only the
    # rows that actually take the raw-wire fallback.  The previous dense
    # fallback conversion and dense selection both allocated a second full
    # ``(rows, dimension + 1)`` matrix even when fallback was rare.
    reconstructed = int16_reconstructed.copy()
    fallback_indices = np.flatnonzero(fallback_mask)
    if fallback_indices.size:
        reconstructed[fallback_indices] = (
            combined[fallback_indices].astype(wire_dtype).astype(np.float64))
    reconstructed_vectors = reconstructed[:, :-1].copy()
    reconstructed_biases = reconstructed[:, -1].copy()
    residual_vectors_after = working_vectors - reconstructed_vectors
    residual_biases_after = working_biases - reconstructed_biases
    if (not np.all(np.isfinite(residual_vectors_after)) or
            not np.all(np.isfinite(residual_biases_after))):
        raise FloatingPointError("batch error-feedback residual became non-finite")
    residual_vector_array[...] = residual_vectors_after
    residual_bias_array[...] = residual_biases_after

    # Safe rows retain their already-computed quantization bound exactly.
    # Only fallback rows need a second trust-bound calculation, and they are
    # evaluated from the fallback subset rather than a dense epsilon matrix.
    selected_bounds = quantization_bounds.copy()
    if fallback_indices.size:
        fallback_selected = reconstructed[fallback_indices]
        fallback_vector_error = (
            fallback_selected[:, :-1] - working_vectors[fallback_indices])
        fallback_bias_error = (
            fallback_selected[:, -1] - working_biases[fallback_indices])
        selected_bounds[fallback_indices] = (
            prediction_norm * np.linalg.norm(fallback_vector_error, axis=1) +
            np.abs(fallback_bias_error))
    if not np.all(np.isfinite(selected_bounds)):
        raise FloatingPointError("selected batch output-trust bounds are non-finite")
    element_count = dimension + 1
    compressed_per_row = payload_byte_stats(
        element_count, parameter_width=width, compressed=True).payload_bytes
    fallback_per_row = payload_byte_stats(
        element_count, parameter_width=width, compressed=False).payload_bytes
    compressed_payload_bytes = int(np.count_nonzero(safe_mask)) * compressed_per_row
    fallback_payload_bytes = int(np.count_nonzero(fallback_mask)) * fallback_per_row
    return BatchCompressionResult(
        reconstructed_vectors=reconstructed_vectors,
        reconstructed_biases=reconstructed_biases,
        residual_vectors=residual_vectors_after.copy(),
        residual_biases=residual_biases_after.copy(),
        safe_mask=safe_mask.copy(),
        fallback_mask=fallback_mask.copy(),
        bounds=quantization_bounds.copy(),
        selected_bounds=selected_bounds.copy(),
        payload_bytes=compressed_payload_bytes + fallback_payload_bytes,
        compressed_payload_bytes=compressed_payload_bytes,
        fallback_payload_bytes=fallback_payload_bytes,
        parameter_width=width,
        element_count=element_count,
    )


def bounded_output_trust_compress_client_block(
        vector_deltas, bias_deltas, prediction_vectors, slacks, *,
        parameter_width=DEFAULT_PARAMETER_WIDTH):
    """Compress OTBC deltas for a block of clients in one vectorized call.

    ``vector_deltas`` has shape ``(B, R, D)`` and ``bias_deltas`` has shape
    ``(B, R)``.  ``prediction_vectors`` has shape ``(B, D)`` and ``slacks``
    has shape ``(B,)``.  Each client uses its own prediction vector and trust
    slack.  The deltas are assumed to already include the caller's
    target-minus-cache error feedback; this function therefore owns no
    residual arrays and does not mutate its inputs.

    The per-row codec, tolerance and fallback rule are exactly the same as
    :func:`bounded_output_trust_compress_batch`.  The implementation keeps
    vector and bias channels separate instead of materializing a dense
    ``(B, R, D + 1)`` matrix.  The caller chooses the block size, allowing it
    to bound the working ``B x R x D`` memory at the integration site.
    """
    vector_deltas = np.asarray(vector_deltas, dtype=np.float64)
    bias_deltas = np.asarray(bias_deltas, dtype=np.float64)
    prediction_vectors = np.asarray(prediction_vectors, dtype=np.float64)
    slacks = np.asarray(slacks, dtype=np.float64)
    if vector_deltas.ndim != 3 or not np.all(np.isfinite(vector_deltas)):
        raise ValueError("vector_deltas must be a finite three-dimensional array")
    if bias_deltas.ndim != 2 or not np.all(np.isfinite(bias_deltas)):
        raise ValueError("bias_deltas must be a finite two-dimensional array")
    if prediction_vectors.ndim != 2 or not np.all(np.isfinite(prediction_vectors)):
        raise ValueError("prediction_vectors must be a finite two-dimensional array")
    if slacks.ndim != 1 or not np.all(np.isfinite(slacks)):
        raise ValueError("slacks must be a finite vector")

    clients, rows, dimension = vector_deltas.shape
    if bias_deltas.shape != (clients, rows):
        raise ValueError("bias_deltas must match the client and row dimensions")
    if prediction_vectors.shape != (clients, dimension):
        raise ValueError("prediction_vectors must match client and vector dimensions")
    if slacks.shape != (clients,) or np.any(slacks < 0.0):
        raise ValueError("slacks must have one finite nonnegative entry per client")
    width = _validate_parameter_width(parameter_width)

    # Flatten only the row axes.  The flattened views preserve C-order and
    # therefore have the same per-row arithmetic order as calling the
    # existing R-row batch API once for each client.
    flat_rows = clients * rows
    flat_vectors = vector_deltas.reshape(flat_rows, dimension)
    flat_biases = bias_deltas.reshape(flat_rows)
    if dimension:
        vector_max_abs = np.max(np.abs(flat_vectors), axis=1)
    else:
        vector_max_abs = np.zeros(flat_rows, dtype=np.float64)
    max_abs = np.maximum(vector_max_abs, np.abs(flat_biases))
    scales = max_abs / INT16_MAX
    scales = np.asarray(scales, dtype=_wire_dtype(width)).astype(np.float64)
    if not np.all(np.isfinite(scales)):
        raise FloatingPointError("client-block scale is not representable at parameter_width")

    quantized_vectors = np.zeros((flat_rows, dimension), dtype=np.dtype("<i2"))
    quantized_biases = np.zeros(flat_rows, dtype=np.dtype("<i2"))
    nonzero_scale = scales > 0.0
    nonzero_indices = np.flatnonzero(nonzero_scale)
    if nonzero_indices.size:
        nonzero_scales = scales[nonzero_indices]
        quantized_vectors[nonzero_indices] = np.clip(
            np.rint(flat_vectors[nonzero_indices] / nonzero_scales[:, None]),
            INT16_MIN, INT16_MAX).astype(np.dtype("<i2"))
        quantized_biases[nonzero_indices] = np.clip(
            np.rint(flat_biases[nonzero_indices] / nonzero_scales),
            INT16_MIN, INT16_MAX).astype(np.dtype("<i2"))

    reconstructed_vectors = (
        quantized_vectors.astype(np.float64) * scales[:, None])
    reconstructed_biases = quantized_biases.astype(np.float64) * scales
    epsilon_vectors = reconstructed_vectors - flat_vectors
    epsilon_biases = reconstructed_biases - flat_biases

    # Calling np.linalg.norm once per prediction row matches the scalar
    # prediction-norm operation in the established per-client API.  Reusing
    # those norms across R selected rows avoids a BxRxD prediction matrix.
    prediction_norms = np.asarray(
        [float(np.linalg.norm(prediction_vectors[index]))
         for index in range(clients)],
        dtype=np.float64)
    prediction_norms_flat = np.repeat(prediction_norms, rows)
    slacks_flat = np.repeat(slacks, rows)
    quantization_bounds = (
        prediction_norms_flat * np.linalg.norm(epsilon_vectors, axis=1)
        + np.abs(epsilon_biases))
    if not np.all(np.isfinite(quantization_bounds)):
        raise FloatingPointError("client-block output-trust bounds are non-finite")
    tolerance = _BOUND_TOLERANCE_MULTIPLIER * np.finfo(float).eps * np.maximum(
        1.0, np.maximum(slacks_flat, quantization_bounds))
    safe_mask = quantization_bounds <= slacks_flat + tolerance
    fallback_mask = ~safe_mask

    fallback_indices = np.flatnonzero(fallback_mask)
    wire_dtype = _wire_dtype(width)
    if fallback_indices.size:
        # Convert only rows that actually use raw-wire fallback.  This avoids
        # constructing a second full float64 fallback matrix for safe rows.
        reconstructed_vectors[fallback_indices] = (
            flat_vectors[fallback_indices].astype(wire_dtype).astype(np.float64))
        reconstructed_biases[fallback_indices] = (
            flat_biases[fallback_indices].astype(wire_dtype).astype(np.float64))

    if (not np.all(np.isfinite(reconstructed_vectors)) or
            not np.all(np.isfinite(reconstructed_biases))):
        raise FloatingPointError("client-block reconstruction is non-finite")

    # Safe rows retain their quantization bound exactly.  Recompute the
    # selected bound only for fallback rows, matching the established batch
    # implementation while avoiding a selected-error matrix for all rows.
    selected_bounds = quantization_bounds.copy()
    if fallback_indices.size:
        fallback_vector_error = (
            reconstructed_vectors[fallback_indices]
            - flat_vectors[fallback_indices])
        fallback_bias_error = (
            reconstructed_biases[fallback_indices]
            - flat_biases[fallback_indices])
        fallback_vector_norms = np.linalg.norm(fallback_vector_error, axis=1)
        selected_bounds[fallback_indices] = (
            prediction_norms_flat[fallback_indices] * fallback_vector_norms
            + np.abs(fallback_bias_error))
    if not np.all(np.isfinite(selected_bounds)):
        raise FloatingPointError("client-block selected output-trust bounds are non-finite")

    safe_mask = safe_mask.reshape(clients, rows)
    fallback_mask = fallback_mask.reshape(clients, rows)
    reconstructed_vectors = reconstructed_vectors.reshape(clients, rows, dimension)
    reconstructed_biases = reconstructed_biases.reshape(clients, rows)
    bounds = quantization_bounds.reshape(clients, rows)
    selected_bounds = selected_bounds.reshape(clients, rows)
    element_count = dimension + 1
    compressed_per_row = payload_byte_stats(
        element_count, parameter_width=width, compressed=True).payload_bytes
    fallback_per_row = payload_byte_stats(
        element_count, parameter_width=width, compressed=False).payload_bytes
    compressed_rows_by_client = np.count_nonzero(safe_mask, axis=1).astype(np.int64)
    fallback_rows_by_client = np.count_nonzero(fallback_mask, axis=1).astype(np.int64)
    compressed_bytes_by_client = compressed_rows_by_client * compressed_per_row
    fallback_bytes_by_client = fallback_rows_by_client * fallback_per_row
    payload_bytes_by_client = compressed_bytes_by_client + fallback_bytes_by_client
    return ClientBlockCompressionResult(
        reconstructed_vectors=reconstructed_vectors,
        reconstructed_biases=reconstructed_biases,
        safe_mask=safe_mask,
        fallback_mask=fallback_mask,
        bounds=bounds,
        selected_bounds=selected_bounds,
        payload_bytes=int(np.sum(payload_bytes_by_client, dtype=np.int64)),
        compressed_payload_bytes=int(np.sum(compressed_bytes_by_client, dtype=np.int64)),
        fallback_payload_bytes=int(np.sum(fallback_bytes_by_client, dtype=np.int64)),
        payload_bytes_by_client=payload_bytes_by_client,
        compressed_payload_bytes_by_client=compressed_bytes_by_client,
        fallback_payload_bytes_by_client=fallback_bytes_by_client,
        parameter_width=width,
        element_count=element_count,
    )


def bounded_output_trust_compress(vector_delta, bias_delta, prediction_vector,
                                  slack, state, *,
                                  parameter_width=DEFAULT_PARAMETER_WIDTH):
    """Compress one service-vector/bias delta under an output-trust slack.

    ``state`` is updated in place with the reconstruction residual.  The
    INT16 candidate is accepted exactly when
    ``||prediction_vector||_2 * ||epsilon_vector||_2 + |epsilon_bias| <= slack``.
    If that condition fails, the same error-feedback-corrected delta is sent
    in the raw floating-point fallback format (FP32 at the default width).
    """
    vector_delta = _finite_vector(vector_delta, "vector delta")
    prediction_vector = _finite_vector(prediction_vector, "prediction vector")
    bias_delta = float(bias_delta)
    slack = float(slack)
    if not np.isfinite(bias_delta) or not np.isfinite(slack) or slack < 0:
        raise ValueError("bias delta must be finite and slack must be nonnegative")
    if vector_delta.shape != prediction_vector.shape:
        raise ValueError("vector delta and prediction vector dimensions must match")
    if not isinstance(state, ErrorFeedbackState):
        raise TypeError("state must be an ErrorFeedbackState")
    if state.vector.shape != vector_delta.shape:
        raise ValueError("error-feedback state dimension must match vector delta")
    width = _validate_parameter_width(parameter_width)

    working_vector = vector_delta + state.vector
    working_bias = bias_delta + state.bias
    working = np.concatenate((working_vector, [working_bias]))
    int16_packet = encode_int16_delta(working, parameter_width=width)
    int16_decoded = decode_delta(int16_packet)
    int16_epsilon_vector = int16_decoded[:-1] - working_vector
    int16_epsilon_bias = float(int16_decoded[-1] - working_bias)
    quantization_bound = output_trust_error_bound(
        prediction_vector, int16_epsilon_vector, int16_epsilon_bias)
    quantization_safe = output_trust_safe(
        prediction_vector, int16_epsilon_vector, int16_epsilon_bias, slack)

    if quantization_safe:
        packet = int16_packet
        decoded = int16_decoded
        used_fallback = False
    else:
        packet = encode_fp32_fallback(working, parameter_width=width)
        decoded = decode_delta(packet)
        used_fallback = True

    reconstructed_vector = decoded[:-1].copy()
    reconstructed_bias = float(decoded[-1])
    residual_vector = working_vector - reconstructed_vector
    residual_bias = float(working_bias - reconstructed_bias)
    if not np.all(np.isfinite(residual_vector)) or not np.isfinite(residual_bias):
        raise FloatingPointError("error-feedback residual became non-finite")
    state.vector[...] = residual_vector
    state.bias = residual_bias
    selected_bound = output_trust_error_bound(
        prediction_vector, reconstructed_vector - working_vector,
        reconstructed_bias - working_bias)
    return CompressionResult(
        packet=packet,
        reconstructed_vector=reconstructed_vector,
        reconstructed_bias=reconstructed_bias,
        residual_vector=residual_vector.copy(),
        residual_bias=residual_bias,
        quantization_bound=quantization_bound,
        selected_bound=selected_bound,
        slack=slack,
        quantization_safe=quantization_safe,
        used_fallback=used_fallback,
    )


# Descriptive aliases for callers that prefer the protocol terminology.
compress_output_trust_delta = bounded_output_trust_compress
compress_output_trust_delta_batch = bounded_output_trust_compress_batch
compress_output_trust_delta_client_block = bounded_output_trust_compress_client_block
decode_compressed_delta = decode_delta


__all__ = [
    "BatchCompressionResult", "ClientBlockCompressionResult", "CompressionResult",
    "DEFAULT_PARAMETER_WIDTH", "DeltaPacket",
    "ErrorFeedbackState", "FLAG_FP", "FLAG_INT16", "INT16_MAX", "INT16_MIN",
    "PayloadByteStats", "bounded_output_trust_compress",
    "compress_output_trust_delta", "compress_output_trust_delta_batch",
    "compress_output_trust_delta_client_block",
    "decode_compressed_delta", "decode_delta",
    "dequantize_signed_int16", "encode_fp32_fallback", "encode_int16_delta",
    "output_trust_error_bound", "output_trust_safe", "payload_byte_stats",
    "quantize_signed_int16", "bounded_output_trust_compress_batch",
    "bounded_output_trust_compress_client_block",
]
