"""The simulator's DRAM tile counts must equal the analytic model, tile for tile.

Runs the real TT-Lang kernels on tt-lang-sim, so these are marked `sim`.
"""

import shutil
from pathlib import Path

import pytest

from tensixfuse.simrun import REPO, matmul_args, run_kernel_script, sim_executable
from tensixfuse.traffic import MatmulConfig, fused_traffic, unfused_traffic

pytestmark = pytest.mark.sim

if not (Path(sim_executable()).exists() or shutil.which("tt-lang-sim")):
    pytest.skip("tt-lang-sim not installed", allow_module_level=True)

SCRIPT = REPO / "kernels" / "ttlang" / "run_matmul.py"

CASES = [
    # size, block, grid, mcast
    (256, (1, 1, 1), (1, 1), False),
    (256, (4, 4, 4), (1, 1), False),
    (512, (8, 8, 4), (1, 1), False),
    (512, (4, 4, 4), (4, 4), False),
    (512, (4, 4, 4), (4, 4), True),
    (512, (2, 2, 2), (2, 2), True),  # two blocks per core: the row source re-sends A
]


def _run(cfg: MatmulConfig, variant: str, act: str = "relu", dry_run: bool = False) -> dict:
    return run_kernel_script(SCRIPT, matmul_args(cfg, variant, act), dry_run=dry_run)


@pytest.mark.parametrize("variant", ["fused", "unfused"])
@pytest.mark.parametrize("size,block,grid,mcast", CASES)
def test_measured_traffic_equals_model(size, block, grid, mcast, variant):
    cfg = MatmulConfig(size, size, size, *block, grid=grid, mcast=mcast)
    res = _run(cfg, variant)
    model = fused_traffic(cfg) if variant == "fused" else unfused_traffic(cfg)
    assert res["dram"]["total"] == model["total"]
    assert res["pcc"] > 0.999


@pytest.mark.parametrize("act", ["relu", "gelu", "sigmoid", "none"])
def test_fused_activations_match_pytorch(act):
    cfg = MatmulConfig(256, 256, 256, 4, 4, 4)
    assert _run(cfg, "fused", act=act)["pcc"] > 0.999


def test_odd_grid_leaves_trailing_cores_idle():
    # 3x3 grid over 4x4 output blocks: the last row and column get fewer blocks.
    cfg = MatmulConfig(512, 512, 512, 4, 4, 4, grid=(3, 3))
    res = _run(cfg, "fused")
    assert res["dram"]["total"] == fused_traffic(cfg)["total"]
    assert res["pcc"] > 0.999


def test_simulator_warns_when_16x16_blocks_overflow_l1():
    cfg = MatmulConfig(1024, 1024, 1024, 16, 16, 4)
    res = _run(cfg, "fused", dry_run=True)
    assert any("exceeds the L1 memory limit" in w for w in res["sim_warnings"])
