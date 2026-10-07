"""Run TT-Lang kernel scripts through tt-lang-sim and read back what moved.

The simulator writes a JSON Lines trace. Every copy_end event between a tensor
and an L1 block carries tile counts split by where the tensor lives (dram,
local_l1, remote_l1), which is the same data tt-lang-sim-stats summarizes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def dram_tiles(trace_path: str | os.PathLike) -> dict:
    """Sum DRAM tile reads and writes in a trace, overall and per tensor."""
    reads = writes = 0
    by_tensor: dict[str, dict[str, int]] = defaultdict(lambda: {"read": 0, "write": 0})
    pipe_tiles_sent = 0
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ev = json.loads(line)
            name = ev.get("event")
            if name == "copy_end":
                dram = ev.get("dram", 0)
                tensor = ev.get("tensor", "?")
                if ev.get("direction") == "read":
                    reads += dram
                    by_tensor[tensor]["read"] += dram
                elif ev.get("direction") == "write":
                    writes += dram
                    by_tensor[tensor]["write"] += dram
            elif name == "pipe_send":
                pipe_tiles_sent += ev.get("tiles", 0)
    return {
        "read": reads,
        "write": writes,
        "total": reads + writes,
        "by_tensor": dict(by_tensor),
        "pipe_tiles_sent": pipe_tiles_sent,
    }


def matmul_args(cfg, variant: str, act: str = "relu") -> list[str]:
    """Command-line arguments for kernels/ttlang/run_matmul.py."""
    args = [
        *("--M", str(cfg.M), "--K", str(cfg.K), "--N", str(cfg.N)),
        *("--block", f"{cfg.bm},{cfg.bn},{cfg.bk}", "--grid", f"{cfg.grid[0]},{cfg.grid[1]}"),
        *("--variant", variant, "--act", act),
    ]
    if cfg.mcast:
        args.append("--mcast")
    return args


def sim_executable() -> str:
    exe = Path(sys.executable).with_name("tt-lang-sim")
    return str(exe) if exe.exists() else "tt-lang-sim"


def run_kernel_script(
    script: str | os.PathLike,
    script_args: list[str],
    *,
    trace: bool = True,
    dry_run: bool = False,
    sim_args: list[str] | None = None,
    timeout: float | None = None,
) -> dict:
    """Run one kernel script under the simulator and return its JSON result.

    The script must accept --out PATH and write a JSON object there. When
    trace is on, the result gains a "dram" entry from the trace.
    """
    with tempfile.TemporaryDirectory(prefix="tensixfuse-") as tmp:
        out = Path(tmp) / "result.json"
        trace_path = Path(tmp) / "trace.jsonl"
        cmd = [sim_executable(), str(script)]
        if trace:
            cmd += ["--trace", str(trace_path), "--trace-events", "copy,pipe"]
        if dry_run:
            cmd += ["--dry-run"]
        cmd += list(sim_args or [])
        cmd += ["--", *script_args, "--out", str(out)]
        if dry_run:
            cmd += ["--no-check"]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=tmp, check=False
        )
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"simulator run failed ({proc.returncode}): {' '.join(cmd)}\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
            )
        result = json.loads(out.read_text())
        result["sim_warnings"] = [
            line for line in proc.stderr.splitlines() if "warn" in line.lower()
        ]
        if trace:
            result["dram"] = dram_tiles(trace_path)
        return result
