import numpy as np
import pytest

from tensixfuse.blockfloat import as_bf16
from tensixfuse.metrics import bf16_ulp_distance, macro_f1, pcc


def bits(x):
    return (as_bf16(np.asarray(x, np.float32)).view(np.uint32) >> 16).astype(np.uint16)


def test_pcc_is_scale_invariant():
    a = np.arange(10.0)
    assert pcc(a, 3 * a + 1) == pytest.approx(1.0)
    assert pcc(a, -a) == pytest.approx(-1.0)


def test_ulp_distance():
    one = bits([1.0])
    assert bf16_ulp_distance(one, one)[0] == 0
    assert bf16_ulp_distance(one, bits([1.0 + 2**-7]))[0] == 1
    assert bf16_ulp_distance(bits([0.0]), bits([-0.0]))[0] == 0
    # Crossing zero counts every step on both sides.
    assert bf16_ulp_distance(bits([2**-133]), bits([-(2**-133)]))[0] == 2


def test_macro_f1_matches_hand_count():
    y = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    p = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    # class 0: tp 1, fp 1, fn 1 -> 0.5. class 1: tp 2, fp 0, fn 0 -> 1.0.
    assert macro_f1(y, p) == pytest.approx(0.75)


def test_macro_f1_absent_class_counts_as_zero():
    y = np.array([[1, 0], [1, 0]])
    p = np.array([[1, 0], [1, 0]])
    assert macro_f1(y, p) == pytest.approx(0.5)  # sklearn zero_division=0
