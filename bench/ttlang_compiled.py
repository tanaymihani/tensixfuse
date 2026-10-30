"""The fusion study's TT-Lang kernels, compiled by the TT-Lang compiler and run
on ttsim instead of TT-Lang's functional simulator.

Runs kernels/ttlang/run_matmul.py with plain Python, so `ttl` is the full
tt-lang package (compiler included) and the device is whatever ttnn opens:
under scripts/setup_ttsim.sh, a simulated chip. Each run checks the output
against PyTorch and reports PCC and the largest error.

    TENSIXFUSE_PINS=pins/tt-lang-1.1.6 bash scripts/setup_ttsim.sh blackhole
    python bench/ttlang_compiled.py --out results/ttlang_compiled.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "kernels" / "ttlang" / "run_matmul.py"

# (label, M=K=N, block bm,bn,bk, grid cols,rows, multicast)
RUNS = {
    "quick": [("1x1 tile blocks", 128, "1,1,1", "1,1", False)],
    "fusion": [
        ("1x1 tile blocks", 256, "1,1,1", "1,1", False),
        ("2x2 tile blocks", 256, "2,2,2", "1,1", False),
        ("4x4 tile blocks", 256, "4,4,4", "1,1", False),
        ("8x8 tile blocks", 256, "8,8,4", "1,1", False),
        ("4x4 blocks per core, 2x2 grid", 256, "4,4,4", "2,2", False),
        ("4x4 blocks per core, 2x2 grid, multicast", 256, "4,4,4", "2,2", True),
    ],
}


def run_one(label, size, block, grid, mcast, variant, act, timeout):
    cmd = [sys.executable, str(SCRIPT), "--M", str(size), "--K", str(size), "--N", str(size)]
    cmd += ["--block", block, "--grid", grid, "--variant", variant, "--act", act]
    if mcast:
        cmd.append("--mcast")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO, check=False
        )
        out, err, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        out, err, code = e.stdout or "", (e.stderr or "") + f"\ntimed out after {timeout}s", -1
    record = {"label": label, "size": size, "variant": variant, "act": act}
    record["seconds"] = round(time.perf_counter() - t0, 1)
    lines = [ln for ln in (out or "").splitlines() if ln.startswith("{")]
    if code == 0 and lines:
        record.update(json.loads(lines[-1]))
        record["ok"] = record.get("pcc", 0) > 0.999
    else:
        record["ok"] = False
        # ttnn's bindings print a long list of nanobind "leaked" warnings at
        # exit after an error; drop them so the traceback survives the cut.
        lines = (err or out or "").strip().splitlines()
        lines = [ln for ln in lines if "leaked" not in ln and "nanobind" not in ln]
        record["error"] = "\n".join(lines[-40:])
    return record


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", choices=sorted(RUNS), default="fusion")
    p.add_argument("--variants", default="fused,unfused")
    p.add_argument("--act", default="relu")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out", default="results/ttlang_compiled.json")
    args = p.parse_args()

    import ttl
    import ttnn

    records = []
    for label, size, block, grid, mcast in RUNS[args.runs]:
        for variant in args.variants.split(","):
            r = run_one(label, size, block, grid, mcast, variant, args.act, args.timeout)
            records.append(r)
            status = f"pcc {r['pcc']:.6f}" if "pcc" in r else "FAILED"
            print(f"{label:42s} {variant:8s} {status}  {r['seconds']}s", flush=True)
            if "error" in r:
                print(r["error"], flush=True)

    out = {
        "tt_lang": getattr(ttl, "__version__", "unknown"),
        "ttnn": getattr(ttnn, "__version__", "unknown"),
        "arch": os.environ.get("TENSIXFUSE_ARCH", "unknown"),
        "simulator": os.path.basename(os.environ.get("TT_METAL_SIMULATOR", "")) or "none",
        "runs": records,
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print("->", dest)
    if not all(r["ok"] for r in records):
        sys.exit(1)


if __name__ == "__main__":
    main()
