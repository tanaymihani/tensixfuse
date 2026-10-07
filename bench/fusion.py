"""Fusion study: fused vs unfused y = relu(a @ b + c) on TT-Lang's simulator.

For every size and data-reuse level, runs both variants, measures DRAM tiles
from the trace, and checks them against src/tensixfuse/traffic.py.

    python bench/fusion.py                      # all sizes
    python bench/fusion.py --sizes 256 512      # quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tensixfuse.simrun import REPO, matmul_args, run_kernel_script
from tensixfuse.traffic import (
    MatmulConfig,
    compulsory_tiles,
    fused_l1_bytes,
    fused_traffic,
    tiles_to_mib,
    unfused_traffic,
)

SCRIPT = REPO / "kernels" / "ttlang" / "run_matmul.py"


def reuse_levels(size: int) -> list[tuple[str, tuple[int, int, int], tuple[int, int], bool]]:
    """(label, block, grid, mcast) from least to most data reuse."""
    g = min(8, size // 32 // 4)  # one 4x4 block per core, up to an 8x8 grid
    return [
        ("1x1 tile blocks", (1, 1, 1), (1, 1), False),
        ("2x2 tile blocks", (2, 2, 2), (1, 1), False),
        ("4x4 tile blocks", (4, 4, 4), (1, 1), False),
        ("8x8 tile blocks", (8, 8, 4), (1, 1), False),
        (f"4x4 blocks per core, {g}x{g} grid", (4, 4, 4), (g, g), False),
        (f"4x4 blocks per core, {g}x{g} grid, multicast", (4, 4, 4), (g, g), True),
    ]


def block_ops(cfg: MatmulConfig) -> int:
    Mb, Nb, Kb = cfg.blocks
    return Mb * Nb * Kb


def run_one(cfg: MatmulConfig, variant: str, label: str, full: bool) -> dict:
    t0 = time.perf_counter()
    res = run_kernel_script(SCRIPT, matmul_args(cfg, variant), dry_run=not full)
    res["seconds"] = round(time.perf_counter() - t0, 1)
    model = fused_traffic(cfg) if variant == "fused" else unfused_traffic(cfg)
    res.update(
        label=label,
        cores=cfg.cores,
        model_tiles=model["total"],
        measured_tiles=res["dram"]["total"],
        matches_model=res["dram"]["total"] == model["total"],
        compulsory_tiles=compulsory_tiles(cfg),
        l1_bytes_per_core=fused_l1_bytes(cfg) if variant == "fused" else None,
    )
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    p.add_argument(
        "--max-full-ops",
        type=int,
        default=40_000,
        help="run with numerics (and a PyTorch check) when block matmuls <= this; else dry-run",
    )
    p.add_argument("--out", default=str(REPO / "results" / "fusion.json"))
    args = p.parse_args()

    records = []
    for size in args.sizes:
        for label, block, grid, mcast in reuse_levels(size):
            cfg = MatmulConfig(size, size, size, *block, grid=grid, mcast=mcast)
            full = block_ops(cfg) <= args.max_full_ops
            for variant in ("unfused", "fused"):
                rec = run_one(cfg, variant, label, full)
                records.append(rec)
                pcc = f"pcc={rec['pcc']:.6f}" if "pcc" in rec else "dry-run"
                flag = "ok" if rec["matches_model"] else "MISMATCH"
                print(
                    f"{size:5d} {label:42s} {variant:8s} "
                    f"{tiles_to_mib(rec['measured_tiles']):8.1f} MiB  model {flag:8s} "
                    f"{pcc}  {rec['seconds']}s",
                    flush=True,
                )
                if rec["sim_warnings"]:
                    print("      sim:", "; ".join(rec["sim_warnings"])[:300], flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1))
    bad = [r for r in records if not r["matches_model"]]
    print(f"\n{len(records)} runs, {len(bad)} mismatches -> {out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
