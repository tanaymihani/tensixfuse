"""Analytic traffic model, checked against hand counts. No simulator needed."""

import pytest

from tensixfuse.traffic import (
    TILE_BYTES,
    MatmulConfig,
    compulsory_tiles,
    fused_l1_bytes,
    fused_traffic,
    fusion_saving_tiles,
    tiles_to_mib,
    unfused_traffic,
)


def test_small_case_by_hand():
    # 256^3 = 8x8x8 tiles, 4x4 blocks -> 2x2 output blocks.
    # A row panel (4x8 tiles) is read once per block column: 8*8*2 = 128.
    cfg = MatmulConfig(256, 256, 256, 4, 4, 4)
    fused = fused_traffic(cfg)
    assert (fused["a"], fused["b"], fused["c"], fused["y"]) == (128, 128, 64, 64)
    assert fused["total"] == 384
    assert unfused_traffic(cfg)["total"] == 640


@pytest.mark.parametrize("size", [256, 512, 1024, 2048])
@pytest.mark.parametrize("block", [(1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 4)])
def test_fusion_always_saves_two_intermediates(size, block):
    cfg = MatmulConfig(size, size, size, *block)
    Mt = size // 32
    # y1 and y2, each written once and read back once.
    assert fusion_saving_tiles(cfg) == 4 * Mt * Mt


def test_readme_table_1024():
    rows = {
        ((1, 1, 1), (1, 1), False): (140, 132),
        ((4, 4, 4), (1, 1), False): (44, 36),
        ((8, 8, 4), (1, 1), False): (28, 20),
        ((4, 4, 4), (8, 8), False): (44, 36),
        ((4, 4, 4), (8, 8), True): (16, 8),
    }
    for (block, grid, mcast), (unfused_mib, fused_mib) in rows.items():
        cfg = MatmulConfig(1024, 1024, 1024, *block, grid=grid, mcast=mcast)
        assert tiles_to_mib(unfused_traffic(cfg)["total"]) == unfused_mib
        assert tiles_to_mib(fused_traffic(cfg)["total"]) == fused_mib


def test_multicast_with_one_block_per_core_hits_the_floor():
    cfg = MatmulConfig(1024, 1024, 1024, 4, 4, 4, grid=(8, 8), mcast=True)
    assert fused_traffic(cfg)["total"] == compulsory_tiles(cfg)


def test_grid_without_multicast_moves_as_much_as_one_core():
    one = MatmulConfig(1024, 1024, 1024, 4, 4, 4)
    grid = MatmulConfig(1024, 1024, 1024, 4, 4, 4, grid=(8, 8))
    assert fused_traffic(one)["total"] == fused_traffic(grid)["total"]


def test_l1_footprint():
    assert fused_l1_bytes(MatmulConfig(1024, 1024, 1024, 8, 8, 4)) == 1024 * 1024
    assert fused_l1_bytes(MatmulConfig(1024, 1024, 1024, 16, 16, 4)) == 3584 * 1024


def test_block_float_tile_sizes():
    # 1,024 values plus one shared 8-bit exponent per 16 values.
    assert TILE_BYTES["bfloat8_b"] == 1088
    assert TILE_BYTES["bfloat4_b"] == 576


def test_rejects_misaligned_shapes():
    with pytest.raises(ValueError):
        MatmulConfig(100, 256, 256, 1, 1, 1)
    with pytest.raises(ValueError):
        MatmulConfig(256, 256, 256, 3, 3, 3)
