"""Run the TT-Metalium BatchNorm + ReLU program on ttsim and check every element.

Takes the real input to GPU-CorruptNet's layer4.2.bn3 (32 frames, channels
last) and that layer's folded scale and shift, writes them as raw bf16, runs
the C++ binary twice, and compares the output with NumPy:

- how many bf16 steps (ULPs) each element is from the fp32 reference rounded
  to bf16, and how many match it exactly
- whether two runs give the same bits

    python bench/bn_relu.py --binary build/bn_relu/bn_relu --out results/bn_relu.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tensixfuse.blockfloat import as_bf16
from tensixfuse.metrics import bf16_ulp_distance, pcc


def bf16_bits(x: np.ndarray) -> np.ndarray:
    return (as_bf16(x).view(np.uint32) >> 16).astype(np.uint16)


def bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", required=True)
    p.add_argument("--data", default="data/corruptnet.npz")
    p.add_argument("--cores", type=int, default=8)
    p.add_argument("--out", default="results/bn_relu.json")
    args = p.parse_args()

    d = np.load(args.data)
    x = bf16_bits(d["bn_input_nhwc"])
    scale = bf16_bits(d["bn_scale"])
    shift = bf16_bits(d["bn_shift"])
    rows, cols = x.shape

    # Reference: the same bf16 inputs, fp32 math, rounded to bf16 once at the end.
    ref = bits_to_f32(x) * bits_to_f32(scale) + bits_to_f32(shift)
    ref_bits = bf16_bits(np.maximum(ref, 0))

    outputs, timings = [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, arr in (("x", x), ("scale", scale), ("shift", shift)):
            arr.tofile(tmp / f"{name}.bin")
        for run in range(2):
            out_path = tmp / f"y{run}.bin"
            proc = subprocess.run(
                [
                    args.binary,
                    str(tmp / "x.bin"),
                    str(tmp / "scale.bin"),
                    str(tmp / "shift.bin"),
                    str(out_path),
                    str(rows),
                    str(cols),
                    str(args.cores),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )
            if proc.returncode != 0:
                raise SystemExit(f"bn_relu failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}")
            timings.append(json.loads(proc.stdout.strip().splitlines()[-1]))
            outputs.append(np.fromfile(out_path, dtype=np.uint16).reshape(rows, cols))

    y = outputs[0]
    ulp = bf16_ulp_distance(y, ref_bits)
    result = {
        "arch": os.environ.get("TENSIXFUSE_ARCH", "?"),
        "rows": rows,
        "channels": cols,
        "elements": int(y.size),
        "tiles": timings[0]["tiles"],
        "cores": timings[0]["cores"],
        "max_ulp": int(ulp.max()),
        "exact_fraction": float((ulp == 0).mean()),
        "within_1_ulp_fraction": float((ulp <= 1).mean()),
        "pcc": pcc(bits_to_f32(y), bits_to_f32(ref_bits)),
        "relu_zeros": float((y == 0).mean()),
        "runs_identical": bool(np.array_equal(outputs[0], outputs[1])),
        "seconds": [t["seconds"] for t in timings],
    }
    print(json.dumps(result))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
