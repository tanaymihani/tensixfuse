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


def estimated_seconds(cfg: MatmulConfig, variant: str, max_full_ops: int) -> float:
    """Rough cost of one run, from the number of block matmuls the simulator steps
    through. Full runs also do the math, dry runs only move data."""
    # The unfused chain's extra add and ReLU passes are cheap next to the matmul.
    per_op = 1.8e-4 if block_ops(cfg) <= max_full_ops else 1.0e-4
    extra = 1.15 if variant == "unfused" else 1.0
    return 1.0 + block_ops(cfg) * per_op * extra


def balanced_shard(runs: list, shard: int, n_shards: int, max_full_ops: int) -> list:
    """Longest-first greedy split: each run goes to the shard with the least
    work so far, so the big runs land on separate shards. It can't split a run,
    though: the slowest shard is never faster than the largest single run (the
    unblocked 2048^3 dry run, about a minute on a hosted runner)."""
    load = [0.0] * n_shards
    owner = {}
    order = sorted(
        range(len(runs)),
        key=lambda i: -estimated_seconds(runs[i][2], runs[i][3], max_full_ops),
    )
    for i in order:
        s = min(range(n_shards), key=load.__getitem__)
        owner[i] = s
        load[s] += estimated_seconds(runs[i][2], runs[i][3], max_full_ops)
    return [run for i, run in enumerate(runs) if owner[i] == shard]


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
    p.add_argument("--shard", default="0/1", help="i/N: run every N-th configuration from i")
    p.add_argument(
        "--emit-json", action="store_true", help="also print the records as one tagged line"
    )
    args = p.parse_args()
    shard, n_shards = (int(v) for v in args.shard.split("/"))

    runs = [
        (size, label, MatmulConfig(size, size, size, *block, grid=grid, mcast=mcast), variant)
        for size in args.sizes
        for label, block, grid, mcast in reuse_levels(size)
        for variant in ("unfused", "fused")
    ]
    mine = balanced_shard(runs, shard, n_shards, args.max_full_ops)
    print(f"shard {shard}/{n_shards}: {len(mine)} of {len(runs)} runs", flush=True)

    t_start = time.perf_counter()
    records = []
    for size, label, cfg, variant in mine:
        rec = run_one(cfg, variant, label, block_ops(cfg) <= args.max_full_ops)
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
    wall = round(time.perf_counter() - t_start, 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1))
    bad = [r for r in records if not r["matches_model"]]
    print(f"\n{len(records)} runs in {wall}s, {len(bad)} mismatches -> {out}")
    if args.emit_json:
        payload = {"shard": shard, "shards": n_shards, "seconds": wall, "records": records}
        print("TENSIXFUSE_RESULTS " + json.dumps(payload), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
