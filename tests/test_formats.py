"""MX, NVFP4 and INT4 emulation: hand-checked values, spec edge cases, and
which axis each format groups along."""

import numpy as np
import pytest

from tensixfuse.blockfloat import quantize
from tensixfuse.formats import (
    BITS_PER_WEIGHT,
    E2M1,
    E4M3,
    FORMATS,
    int4,
    mx,
    nvfp4,
    round_minifloat,
    round_trip,
)
from tensixfuse.traffic import TILE_BYTES


def test_e2m1_rounds_to_nearest_even_and_saturates():
    x = np.array([0, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7, 100, -2.5])
    want = np.array([0, 0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, -2.0])
    assert np.array_equal(round_minifloat(x, *E2M1), want)


def test_every_e4m3_value_survives():
    subnormals = [k * 2.0**-9 for k in range(8)]
    normals = [(1 + m / 8) * 2.0**e for e in range(-6, 9) for m in range(8)]
    grid = np.array(subnormals + [v for v in normals if v <= 448])
    assert grid.max() == 448
    assert np.array_equal(round_minifloat(grid, *E4M3), grid)
    assert np.array_equal(round_minifloat(-grid, *E4M3), -grid)
    # Halfway between 0 and the smallest subnormal goes to the even one (0);
    # past the largest normal saturates.
    assert round_minifloat(np.array([2.0**-10]), *E4M3)[0] == 0
    assert round_minifloat(np.array([3 * 2.0**-10]), *E4M3)[0] == 2.0**-8
    assert round_minifloat(np.array([1e6]), *E4M3)[0] == 448


def test_mx_scale_is_a_power_of_two_from_the_block_max():
    # Largest magnitude 0.75: floor(log2) = -1, minus E2M1's emax of 2, so the
    # scale is 2**-3 and these values land on the FP4 grid exactly.
    vals = np.array([6, -4, 3, 0.5, -1.5, 2, 0, 1] * 4) * 2.0**-3
    assert np.array_equal(mx(vals[None, :], "fp4")[0], vals.astype(np.float32))
    # 7 * 2**-3 has the same scale but 7 isn't on the grid: MX clamps it to 6.
    vals[0] = 7 * 2.0**-3
    assert mx(vals[None, :], "fp4")[0, 0] == np.float32(6 * 2.0**-3)


def test_mxfp8_keeps_three_mantissa_bits_and_clamps_past_448():
    x = np.array([[1.9] + [0.3, -0.7, 1.1, 0.05] * 7 + [0.2, 0.4, 0.6]])
    q = mx(x, "fp8")
    # Scale 2**(floor(log2 1.9) - 8) = 2**-8 puts 1.9 at 486, past E4M3's 448:
    # the spec rounds the scale down, so a block's largest values can clamp.
    assert q[0, 0] == np.float32(448 * 2.0**-8)
    # The rest are within half a step, 2**-4 of their own size.
    assert np.all(np.abs(q[0, 1:] - x[0, 1:]) <= 2.0**-4 * np.abs(x[0, 1:]))
    assert np.array_equal(mx(np.zeros((1, 32)), "fp8"), np.zeros((1, 32), np.float32))


def test_nvfp4_keeps_each_block_max_within_e4m3_rounding():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((16, 64)) * rng.uniform(0.01, 10, (16, 1))
    q = nvfp4(x)
    amax = np.abs(x.reshape(-1, 16)).max(axis=1)
    qmax = np.abs(q.reshape(-1, 16)).max(axis=1)
    assert np.all(np.abs(qmax - amax) <= 0.07 * amax)
    assert np.array_equal(nvfp4(np.zeros((2, 16))), np.zeros((2, 16), np.float32))


def test_int4_error_is_at_most_half_a_step():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 256))
    q = int4(x)
    for g, qg in zip(x.reshape(-1, 128), q.reshape(-1, 128), strict=True):
        step = (max(g.max(), 0) - min(g.min(), 0)) / 15
        assert np.abs(qg - g).max() <= 0.5 * step * 1.01
        assert len(np.unique(qg)) <= 16


def test_round_trip_groups_along_the_right_axis():
    # One large weight in column 5 (an output feature). Formats grouped along K
    # flatten only column 5's own block; block floats grouped along N flatten
    # row 0, which mixes in every output feature.
    w = np.full((32, 32), 0.01, np.float32)
    w[0, 5] = 100.0
    for fmt in ("mxfp4", "nvfp4", "int4_g128", "bfloat4_b_k"):
        if fmt == "int4_g128":
            w_big = np.tile(w, (4, 1))  # K = 128
            q = round_trip(w_big, fmt)
        else:
            q = round_trip(w, fmt)
        assert q[1, 5] == 0, fmt
        assert q[1, 4] != 0, fmt
    q = round_trip(w, "bfloat4_b")
    assert q[0, 4] == 0 and q[1, 5] != 0
    assert np.array_equal(round_trip(w, "bfloat4_b_k"), quantize(w.T, "bfloat4_b").T)


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip_keeps_shape_and_dtype(fmt):
    w = np.random.default_rng(3).standard_normal((256, 96)).astype(np.float32)
    q = round_trip(w, fmt)
    assert q.shape == w.shape and q.dtype == np.float32


def test_unknown_format_is_an_error():
    with pytest.raises(ValueError, match="unknown format"):
        round_trip(np.zeros((32, 32)), "fp3")


def test_block_float_bits_match_tile_sizes():
    for fmt in ("bfloat16", "bfloat8_b", "bfloat4_b", "float32"):
        assert BITS_PER_WEIGHT[fmt] == TILE_BYTES[fmt] * 8 / 1024
