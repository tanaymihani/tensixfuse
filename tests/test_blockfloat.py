"""Block-float emulation: hand-checked cases. The bit-for-bit comparison with
tt-metal's own packer runs on Linux in CI (scripts/check_bfp_packing.py)."""

import numpy as np
import pytest

from tensixfuse.blockfloat import as_bf16, quantize, zeroed_fraction


def test_exact_values_survive():
    # Powers of two and small integers in a group of matching scale are exact.
    x = np.array([[1, 2, 3, 4, 0.5, 0.25, -1, -2, 1.5, 3.5, 0, 0, 1, 1, 2, 2]], np.float32)
    assert np.array_equal(quantize(x, "bfloat8_b"), x)


def test_shared_exponent_flattens_small_neighbors():
    # One value of 16 sets the exponent; 3 mantissa bits can't hold 1/16 of it.
    x = np.full((1, 16), 1.0, np.float32)
    x[0, 0] = 16.0
    q4 = quantize(x, "bfloat4_b")
    assert q4[0, 0] == 16.0
    assert np.all(q4[0, 1:] == 0.0)
    # bfloat8_b has room: 1.0 is 4 bits below 16 and keeps its value.
    assert np.all(quantize(x, "bfloat8_b")[0, 1:] == 1.0)


def test_round_to_nearest_even_and_saturation():
    # 1.9921875 = 0b1.1111111 needs 8 bits; bfp8_b keeps 7 (hidden bit included),
    # so it rounds up to 2.0, which would need renormalizing: tt-metal saturates.
    x = np.zeros((1, 16), np.float32)
    x[0, 0] = 1.9921875
    assert quantize(x, "bfloat8_b")[0, 0] == pytest.approx(1.984375)  # 127 / 64


def test_groups_are_independent():
    x = np.ones((2, 32), np.float32)
    x[0, 0] = 1000.0  # only the first group of the first row is affected
    q = quantize(x, "bfloat4_b")
    assert np.all(q[0, 1:16] == 0) and np.all(q[0, 16:] == 1) and np.all(q[1] == 1)


def test_relative_error_bounds_on_gaussian():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((256, 256)).astype(np.float32)
    err8 = np.abs(quantize(x, "bfloat8_b") - x).max()
    err4 = np.abs(quantize(x, "bfloat4_b") - x).max()
    group_max = np.abs(x.reshape(-1, 16)).max(axis=1)
    # Worst case is half a step at the group's exponent: 2^-6 and 2^-2 of 2^e.
    assert err8 <= group_max.max() * 2**-6
    assert err4 <= group_max.max() * 2**-2
    assert zeroed_fraction(x, "bfloat4_b") > zeroed_fraction(x, "bfloat8_b")


def test_bf16_rounding():
    x = np.array([1.0 + 2**-8, 1.0 + 3 * 2**-8, 1.0 + 2**-7], np.float32)
    assert list(as_bf16(x)) == [1.0, 1.0 + 2**-6, 1.0 + 2**-7]
