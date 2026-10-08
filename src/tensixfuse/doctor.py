"""Known-answer health check for a simulator node.

Opens the simulated chip, runs a matmul and a bfloat8_b round trip against
answers computed on the host, and prints one JSON report. Exits 1 if anything
is off, so CI, Docker and the Ansible health_check role can all gate on it.

    tensixfuse-doctor                # chip from TT_METAL_SIMULATOR
    tensixfuse-doctor --out health.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from tensixfuse.blockfloat import quantize
from tensixfuse.metrics import pcc

REPO = Path(__file__).resolve().parents[2]


def _pinned(name: str) -> str | None:
    path = REPO / name
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split("=", 1)[-1].strip("'\"")
    return None


def run_checks() -> dict:
    report: dict = {
        "host": platform.node(),
        "python": platform.python_version(),
        "ttsim_version": _pinned("ttsim-version"),
        "simulator": os.environ.get("TT_METAL_SIMULATOR"),
        "slow_dispatch": os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") == "1",
        "checks": {},
    }
    if not report["simulator"] or not Path(report["simulator"]).exists():
        report["checks"]["simulator_present"] = False
        report["healthy"] = False
        return report
    report["checks"]["simulator_present"] = True

    import torch
    import ttnn

    report["ttnn"] = getattr(ttnn, "__version__", None)
    t0 = time.perf_counter()
    device = ttnn.open_device(device_id=0)
    try:
        report["arch"] = str(device.arch())
        rng = np.random.default_rng(1234)
        a = rng.standard_normal((64, 128)).astype(np.float32)
        b = rng.standard_normal((128, 64)).astype(np.float32)

        def dev(x, dtype):
            return ttnn.from_torch(
                torch.from_numpy(x), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
            )

        out = ttnn.to_torch(ttnn.matmul(dev(a, ttnn.bfloat16), dev(b, ttnn.bfloat16)))
        matmul_pcc = pcc(out.float().numpy(), a @ b)
        report["checks"]["matmul_pcc"] = round(matmul_pcc, 6)

        back = ttnn.to_torch(dev(a, ttnn.bfloat8_b)).float().numpy()
        want = quantize(a, "bfloat8_b")
        same = (back.view(np.uint32) == want.view(np.uint32)) | ((back == 0) & (want == 0))
        report["checks"]["bfloat8_b_bit_exact"] = bool(same.all())
    finally:
        ttnn.close_device(device)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    report["healthy"] = matmul_pcc > 0.999 and report["checks"]["bfloat8_b_bit_exact"]
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", help="also write the report here")
    args = p.parse_args(argv)
    report = run_checks()
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
