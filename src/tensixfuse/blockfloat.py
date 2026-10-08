"""Tenstorrent block-float formats (bfloat8_b, bfloat4_b) in NumPy.

Mirrors tt-metal's host-side packer (tt_metal/impl/data_format/blockfloat_common.cpp
at v0.79.0), so the values here should match what ttnn.from_torch(...,
dtype=ttnn.bfloat8_b) stores, bit for bit:

- Each group of 16 consecutive values along a row (one row of a 16x16 face, so
  columns 16c..16c+15) shares the largest of their 8-bit fp32 exponents.
- Each value keeps a sign and a mantissa of 7 (bfp8_b) or 3 (bfp4_b) bits,
  hidden bit included, after shifting right by its exponent's distance from the
  shared one. Rounding is to nearest, ties to even; a round-up that overflows
  saturates instead of renormalizing.
- Zeros and fp32 denormals become 0.
"""

from __future__ import annotations

import numpy as np

GROUP = 16
MANTISSA_BITS = {"bfloat8_b": 7, "bfloat4_b": 3}


def quantize(x: np.ndarray, fmt: str = "bfloat8_b", truncate: bool = False) -> np.ndarray:
    """Round-trip x through a block-float format; returns float32 of x's shape.

    Groups run along the last axis, which must be a multiple of 16 (pad first).
    """
    bits = MANTISSA_BITS[fmt]
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.shape[-1] % GROUP:
        raise ValueError(f"last dim {x.shape[-1]} is not a multiple of {GROUP}")
    shape = x.shape
    u = x.reshape(-1, GROUP).view(np.uint32)

    exp = ((u >> 23) & 0xFF).astype(np.int64)
    sign = (u >> 31).astype(np.int64)
    mant = ((u & 0x7FFFFF) | (1 << 23)).astype(np.int64)  # hidden bit back in

    shared = exp.max(axis=1, keepdims=True)
    shift = shared - exp
    mant = np.where(shift >= 32, 0, mant >> np.minimum(shift, 31))

    drop = 24 - bits
    if truncate:
        q = mant >> drop
    else:
        rest = mant & ((1 << drop) - 1)
        q = mant >> drop
        tie = 1 << (drop - 1)
        q = q + ((rest > tie) | ((rest == tie) & (q & 1 == 1)))
        q = np.minimum(q, (1 << bits) - 1)
    q = np.where(exp == 0, 0, q)  # zeros and denormals

    value = np.ldexp(q.astype(np.float64), (shared - 127 - (bits - 1)).astype(np.int32))
    value = np.where(sign == 1, -value, value)
    return value.astype(np.float32).reshape(shape)


def quantize_weight(w_kn: np.ndarray, fmt: str) -> np.ndarray:
    """Quantize a [K, N] matmul operand the way it's laid out on device.

    Pads N up to a multiple of 32 tiles' width with zeros, which is what tile
    layout does, then strips the padding again.
    """
    if fmt in ("bfloat16", "float32"):
        return as_bf16(w_kn) if fmt == "bfloat16" else w_kn.astype(np.float32)
    n = w_kn.shape[1]
    pad = (-n) % 32
    padded = np.pad(np.asarray(w_kn, dtype=np.float32), ((0, 0), (0, pad)))
    return quantize(padded, fmt)[:, :n]


def as_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (nearest, ties to even) and back to float32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16
    return rounded.astype(np.uint32).view(np.float32).reshape(np.shape(x))


def zeroed_fraction(x: np.ndarray, fmt: str) -> float:
    """Share of nonzero values that the format stores as exactly zero."""
    q = quantize(x, fmt)
    nonzero = x != 0
    return float(((q == 0) & nonzero).sum() / max(1, nonzero.sum()))
