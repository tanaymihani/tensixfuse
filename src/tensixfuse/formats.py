"""Other low-bit weight formats, to compare with Tenstorrent's block floats.

Round-trip emulation in NumPy (quantize, then back to float32):

- MXFP8 / MXFP4 (OCP Microscaling spec v1.0): 32 values share a power-of-two
  scale (E8M0) set from their largest magnitude; elements are FP8 E4M3 or
  FP4 E2M1, rounded to nearest even and clamped to the largest normal.
- NVFP4: 16 values share an FP8 E4M3 scale, under one FP32 scale per tensor;
  elements are FP4 E2M1.
- INT4 g128: asymmetric round-to-nearest, one FP16 scale and a 4-bit zero
  point per 128 values. That's how GPTQ and AWQ store weights, without the
  calibration they do to pick them.

Groups run along the last axis. round_trip() takes a [K, N] matmul weight and
groups the way each format is laid out: block floats along N, as tt-metal's
tile layout packs them (tensixfuse.blockfloat), the rest along K.
"""

from __future__ import annotations

import numpy as np

from tensixfuse.blockfloat import as_bf16, quantize_weight

# mantissa bits, exponent bias, largest normal
E4M3 = (3, 7, 448.0)
E2M1 = (1, 1, 6.0)

# Storage per weight, scales included.
BITS_PER_WEIGHT = {
    "float32": 32.0,
    "bfloat16": 16.0,
    "bfloat8_b": 8.5,  # 1088 B per 32x32 tile: 8-bit sign+mantissa, 1 exponent byte per 16
    "bfloat4_b": 4.5,  # 576 B per tile
    "bfloat4_b_k": 4.5,
    "mxfp8": 8 + 8 / 32,
    "mxfp4": 4 + 8 / 32,
    "nvfp4": 4 + 8 / 16,  # plus one FP32 per tensor
    "int4_g128": 4 + (16 + 4) / 128,
}
FORMATS = tuple(BITS_PER_WEIGHT)


def round_minifloat(x: np.ndarray, mbits: int, bias: int, max_normal: float) -> np.ndarray:
    """Round to a small float format with subnormals: nearest, ties to even, saturating."""
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    _, e = np.frexp(a)  # a = m * 2**e with 0.5 <= m < 1, so floor(log2(a)) = e - 1
    e = np.maximum(e - 1, 1 - bias)  # subnormals keep the smallest normal's step
    step = np.ldexp(1.0, e - mbits)
    q = np.minimum(np.round(a / step) * step, max_normal)  # np.round ties to even
    return np.copysign(q, x)


def _groups(x: np.ndarray, size: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] % size:
        raise ValueError(f"last dim {x.shape[-1]} is not a multiple of {size}")
    return x.reshape(-1, size)


def mx(x: np.ndarray, element: str = "fp4", block: int = 32) -> np.ndarray:
    """OCP MX round trip: E8M0 scale per block, then E2M1 ("fp4") or E4M3 ("fp8")."""
    mbits, bias, max_normal = E2M1 if element == "fp4" else E4M3
    emax = int(np.log2(max_normal))  # exponent of the largest normal: 2 or 8
    g = _groups(x, block)
    _, e = np.frexp(np.abs(g).max(axis=1, keepdims=True))
    scale = np.ldexp(1.0, np.clip(e - 1 - emax, -127, 127))
    q = round_minifloat(g / scale, mbits, bias, max_normal) * scale
    return q.astype(np.float32).reshape(np.shape(x))


def nvfp4(x: np.ndarray, block: int = 16) -> np.ndarray:
    """NVFP4 round trip: FP32 tensor scale, E4M3 scale per 16, E2M1 elements."""
    g = _groups(x, block)
    tensor_amax = np.abs(g).max()
    if tensor_amax == 0:
        return np.zeros(np.shape(x), dtype=np.float32)
    tensor_scale = tensor_amax / (E2M1[2] * E4M3[2])  # puts block scales in E4M3's range
    block_scale = round_minifloat(np.abs(g).max(axis=1, keepdims=True) / 6.0 / tensor_scale, *E4M3)
    scale = block_scale * tensor_scale
    safe = np.where(scale > 0, scale, 1.0)
    q = np.where(scale > 0, round_minifloat(g / safe, *E2M1) * scale, 0.0)
    return q.astype(np.float32).reshape(np.shape(x))


def int4(x: np.ndarray, group: int = 128) -> np.ndarray:
    """Asymmetric INT4 round trip: FP16 scale and 4-bit zero point per group."""
    g = _groups(x, group)
    lo = np.minimum(g.min(axis=1, keepdims=True), 0.0)
    hi = np.maximum(g.max(axis=1, keepdims=True), 0.0)
    scale = ((hi - lo) / 15).astype(np.float16).astype(np.float64)
    scale = np.where(scale > 0, scale, 1.0)
    zero = np.clip(np.round(-lo / scale), 0, 15)
    q = np.clip(np.round(g / scale) + zero, 0, 15)
    return ((q - zero) * scale).astype(np.float32).reshape(np.shape(x))


def round_trip(w_kn: np.ndarray, fmt: str) -> np.ndarray:
    """Round-trip a [K, N] matmul weight (K = input features) through fmt."""
    w = np.asarray(w_kn, dtype=np.float32)
    if fmt == "float32":
        return w.copy()
    if fmt == "bfloat16":
        return as_bf16(w)
    if fmt in ("bfloat8_b", "bfloat4_b"):
        return quantize_weight(w, fmt)
    if fmt == "bfloat4_b_k":  # what-if: weight stored [N, K], so groups run along K
        return quantize_weight(w.T, "bfloat4_b").T.copy()
    along_k = {
        "mxfp8": lambda t: mx(t, "fp8"),
        "mxfp4": lambda t: mx(t, "fp4"),
        "nvfp4": nvfp4,
        "int4_g128": int4,
    }
    if fmt not in along_k:
        raise ValueError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")
    return along_k[fmt](w.T).T.copy()
