"""Shared SAMT binary tensor IO for parity, benchmark and verification tooling.

Format contract (mirrors ``cpp_ggml/src/common.hpp``):

    magic "SAMT" | int32 ndims | int64 ne[ndims] | int32 ggml_type | data

``ne`` uses ggml order (``ne[0]`` is the fastest-varying dimension), so the
logical NumPy shape is ``tuple(reversed(ne))``.  Two historical GGML_TYPE_I32
encodings (26 in this build's ggml, 30 in upstream ggml) are accepted on read
and normalized to 26 on write, matching ``load_raw_tensor`` in
``cpp_ggml/src/common.cpp``.

This module is deliberately torch-free: parity scripts import it from their
own directory without triggering any heavy environment setup.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

SAMT_MAGIC = b"SAMT"
GGML_F32 = 0                            # GGML_TYPE_F32
GGML_I32 = 26                           # GGML_TYPE_I32 in this build's ggml
GGML_I32_COMPAT = frozenset({GGML_I32, 30})  # 30 = upstream's newer encoding

_F32 = np.dtype("<f4")
_I32 = np.dtype("<i4")


def _dtype_for_ggml_type(ggml_type: int, path) -> np.dtype:
    if ggml_type == GGML_F32:
        return _F32
    if ggml_type in GGML_I32_COMPAT:
        return _I32
    raise ValueError(f"{path}: unsupported SAMT ggml type {ggml_type}")


def _open_header(path):
    stream = Path(path).open("rb")
    try:
        if stream.read(4) != SAMT_MAGIC:
            raise ValueError(f"{path}: invalid SAMT magic")
        (rank,) = struct.unpack("<i", stream.read(4))
        if rank <= 0 or rank > 8:
            raise ValueError(f"{path}: invalid SAMT rank {rank}")
        ne = struct.unpack(f"<{rank}q", stream.read(rank * 8))
        (ggml_type,) = struct.unpack("<i", stream.read(4))
    except BaseException:
        stream.close()
        raise
    return stream, ne, ggml_type


def _check_size(path, values: np.ndarray, ne) -> None:
    expected = int(np.prod(ne, dtype=np.int64))
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} values, got {values.size}")


def read_samt_header(path) -> tuple[tuple[int, ...], int]:
    """Return ``(ne, ggml_type)`` without loading the payload."""
    stream, ne, ggml_type = _open_header(path)
    stream.close()
    return ne, ggml_type


def read_samt(path) -> tuple[tuple[int, ...], np.ndarray]:
    """Read ``(ne, flat writable values)``; F32/I32 payloads are auto-mapped."""
    stream, ne, ggml_type = _open_header(path)
    try:
        values = np.frombuffer(stream.read(), dtype=_dtype_for_ggml_type(ggml_type, path)).copy()
    finally:
        stream.close()
    _check_size(path, values, ne)
    return ne, values


def read_samt_f32(path) -> tuple[tuple[int, ...], np.ndarray]:
    """Read ``(ne, flat writable values)`` from a strictly-F32 payload."""
    stream, ne, ggml_type = _open_header(path)
    try:
        if ggml_type != GGML_F32:
            raise ValueError(f"{path}: expected F32 payload, type={ggml_type}")
        values = np.frombuffer(stream.read(), dtype=_F32).copy()
    finally:
        stream.close()
    _check_size(path, values, ne)
    return ne, values


def read_samt_bytes(path) -> tuple[tuple[int, ...], bytes]:
    """Read ``(ne, raw F32 payload bytes)`` without any dtype conversion."""
    stream, ne, ggml_type = _open_header(path)
    try:
        if ggml_type != GGML_F32:
            raise ValueError(f"{path}: expected F32 SAMT")
        payload = stream.read()
    finally:
        stream.close()
    expected_bytes = 4
    for dimension in ne:
        expected_bytes *= int(dimension)
    if len(payload) != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes} bytes, got {len(payload)}")
    return ne, payload


def read_samt_array(path, *, dtype: np.dtype | None = None,
                    reverse_dims: bool = True) -> np.ndarray:
    """Read a reshaped logical NumPy array.

    ``dtype`` acts as a strict payload gate: F32 request accepts only F32
    payloads, an integer dtype accepts only I32 payloads (both historical
    encodings).  Without it, F32/I32 payloads are auto-mapped.
    """
    stream, ne, ggml_type = _open_header(path)
    try:
        if dtype is not None:
            accepted = (GGML_F32,) if dtype == _F32 else tuple(GGML_I32_COMPAT)
            if ggml_type not in accepted:
                expected = "/".join(str(value) for value in sorted(accepted))
                raise ValueError(f"{path}: expected GGML type {expected}, got {ggml_type}")
            values = np.frombuffer(stream.read(), dtype=dtype)
        else:
            values = np.frombuffer(stream.read(), dtype=_dtype_for_ggml_type(ggml_type, path))
    finally:
        stream.close()
    _check_size(path, values, ne)
    dimensions = tuple(reversed(ne)) if reverse_dims else tuple(ne)
    return values.reshape(dimensions).copy()


def load_samt(path) -> np.ndarray:
    """Load a SAMT tensor with its logical NumPy shape (reversed ``ne``)."""
    return read_samt_array(path)


def read_samt_shape(path) -> tuple[int, ...]:
    """Return the logical (reversed) shape without loading the payload."""
    ne, _ = read_samt_header(path)
    return tuple(reversed(ne))


def _write(path, ne, ggml_type: int, data) -> None:
    payload = np.ascontiguousarray(data)
    expected = int(np.prod(ne, dtype=np.int64))
    if payload.size != expected:
        raise ValueError(f"{path}: expected {expected} values, got {payload.size}")
    with Path(path).open("wb") as stream:
        stream.write(SAMT_MAGIC)
        stream.write(struct.pack("<i", len(ne)))
        stream.write(struct.pack(f"<{len(ne)}q", *ne))
        stream.write(struct.pack("<i", ggml_type))
        stream.write(payload.tobytes())


def write_samt_f32(path, ne, data) -> None:
    """Write an F32 payload; ``ne`` is written verbatim in ggml order."""
    _write(path, ne, GGML_F32, np.asarray(data, dtype=_F32))


def write_samt_i32(path, ne, data) -> None:
    """Write an I32 payload (this build's GGML_TYPE_I32); ``ne`` in ggml order."""
    _write(path, ne, GGML_I32, np.asarray(data, dtype=_I32))


def write_samt_array(path, values: np.ndarray, ggml_type: int) -> None:
    """Write a logical-shaped NumPy array; dimensions are reversed to ggml order."""
    values = np.ascontiguousarray(values)
    _write(path, tuple(reversed(values.shape)), int(ggml_type), values)


def save_samt(path, values: np.ndarray) -> None:
    """Write a logical-shaped NumPy array, inferring F32/I32 from its dtype."""
    values = np.ascontiguousarray(values)
    if values.dtype == np.float32:
        ggml_type = GGML_F32
    elif values.dtype == np.int32:
        ggml_type = GGML_I32
    else:
        raise ValueError(f"{path}: SAMT only accepts F32/I32 here, got {values.dtype}")
    write_samt_array(path, values, ggml_type)
