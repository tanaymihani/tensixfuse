"""DRAM traffic must not go up. Baselines live in bench/baselines.json.

The simulator-vs-model tests catch a kernel that drifts from the model; this
catches a change that makes the kernel and the model worse together.
"""

import json
import shutil
from pathlib import Path

import pytest

from tensixfuse.simrun import REPO, matmul_args, run_kernel_script, sim_executable
from tensixfuse.traffic import MatmulConfig

pytestmark = pytest.mark.sim

if not (Path(sim_executable()).exists() or shutil.which("tt-lang-sim")):
    pytest.skip("tt-lang-sim not installed", allow_module_level=True)

BASELINES = json.loads((REPO / "bench" / "baselines.json").read_text())["dram_tiles"]
SCRIPT = REPO / "kernels" / "ttlang" / "run_matmul.py"


def parse(key: str) -> tuple[str, MatmulConfig]:
    variant, size, _, block, _, grid, *rest = key.split()
    bm, bn, bk = (int(v) for v in block.split("x"))
    cols, rows = (int(v) for v in grid.split("x"))
    s = int(size)
    return variant, MatmulConfig(s, s, s, bm, bn, bk, grid=(cols, rows), mcast="mcast" in rest)


@pytest.mark.parametrize("key", sorted(BASELINES))
def test_dram_tiles_do_not_exceed_baseline(key):
    variant, cfg = parse(key)
    res = run_kernel_script(SCRIPT, matmul_args(cfg, variant), dry_run=True)
    assert res["dram"]["total"] <= BASELINES[key]
