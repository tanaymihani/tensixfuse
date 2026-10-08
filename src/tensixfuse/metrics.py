"""Comparison metrics: PCC, bf16 ULP distance, and multi-label macro-F1."""

from __future__ import annotations

import numpy as np


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two arrays, flattened. Scale-invariant, like
    Tenstorrent's model tests use."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def bf16_ulp_distance(a_bits: np.ndarray, b_bits: np.ndarray) -> np.ndarray:
    """How many bf16 steps apart two arrays of raw bf16 bits are.

    Maps sign-magnitude bits onto a monotonic integer line, so +0 and -0 are
    the same point and adjacent representable values differ by 1.
    """

    def ordered(bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, np.uint16).astype(np.int64)
        mag = bits & 0x7FFF
        return np.where(bits & 0x8000, -mag, mag)

    return np.abs(ordered(a_bits) - ordered(b_bits))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multi-label macro-F1, same as sklearn's f1_score(average="macro",
    zero_division=0), which GPU-CorruptNet reports."""
    t = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    tp = (t & p).sum(0)
    fp = (~t & p).sum(0)
    fn = (t & ~p).sum(0)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())
