"""My TT-Metalium linear layer against ttnn.linear, on the same simulated chip.

y = act(x @ w + b) with x in bf16, w in bfloat8_b and b in bf16: once with
kernels/metalium/linear_bias_act (my program) and once with ttnn.linear at the
same settings (HiFi4, fp32 accumulation), then compares the two outputs bit for
bit. Both are also checked against fp32 math on the same rounded inputs (w
through tensixfuse.blockfloat, the emulation of tt-metal's packer).

The default shape is GPT-2's c_fc on 128 tokens (768 -> 3072) with GELU. The
data is seeded random: x is unit variance like a LayerNorm output, w and b
scaled like GPT-2's.

    python bench/linear_bias_act.py --binary build/linear_bias_act/linear_bias_act \
        --out results/linear_bias_act_blackhole.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tensixfuse.blockfloat import as_bf16, quantize_weight
from tensixfuse.metrics import bf16_ulp_distance, pcc

TT_FORMAT = {"bfp8": "bfloat8_b", "bf16": "bfloat16"}


def core_grid(ttnn):
    """4 x 8 cores: one row of output tiles per core row, 12 columns per core. Fits both chips."""
    return ttnn.CoreGrid(y=4, x=8)


def bf16_bits(t) -> np.ndarray:
    """Raw bits of a torch bf16 tensor."""
    import torch

    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def reference(x, w, b, fmt: str, act: str) -> np.ndarray:
    """fp32 math on the rounded inputs, for checking both outputs."""
    wq = quantize_weight(w, TT_FORMAT[fmt])
    y = x.astype(np.float64) @ wq.astype(np.float64) + b.astype(np.float64)
    if act == "relu":
        y = np.maximum(y, 0)
    elif act.startswith("gelu"):  # the exact form; ttnn's accurate GELU approximates it
        y = 0.5 * y * (1 + np.vectorize(math.erf)(y / math.sqrt(2)))
    return y.astype(np.float32)


def run_ttnn(x_t, w_t, b_t, fmt: str, act: str, acc: str) -> np.ndarray:
    import torch
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        kwargs = {
            "math_fidelity": ttnn.MathFidelity.HiFi4,
            "math_approx_mode": False,
            "fp32_dest_acc_en": acc == "fp32",
            "packer_l1_acc": False,
        }
        try:
            cfg = ttnn.init_device_compute_kernel_config(device.arch(), **kwargs)
        except (AttributeError, TypeError):
            cfg = ttnn.WormholeComputeKernelConfig(**kwargs)
        dtype = ttnn.bfloat8_b if fmt == "bfp8" else ttnn.bfloat16
        x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        w = ttnn.from_torch(w_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        b = ttnn.from_torch(
            b_t.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        # Both of my GELU variants are compared with ttnn.linear's one GELU.
        ttnn_act = {"none": None, "relu": "relu"}.get(act, "gelu")
        y = ttnn.linear(x, w, bias=b, activation=ttnn_act, compute_kernel_config=cfg)
        fused = bf16_bits(ttnn.to_torch(y).to(torch.bfloat16))
        post = None
        if ttnn_act == "gelu":
            # The same with GELU as its own op on the bf16 output of the linear.
            y0 = ttnn.linear(x, w, bias=b, compute_kernel_config=cfg)
            post = bf16_bits(ttnn.to_torch(ttnn.gelu(y0)).to(torch.bfloat16))
        # Without a core grid, ttnn's matmul.cpp runs the activation as a separate op after the
        # matmul; with one, it goes into the kernel.
        try:
            y_grid = ttnn.linear(
                x,
                w,
                bias=b,
                activation=ttnn_act,
                compute_kernel_config=cfg,
                core_grid=core_grid(ttnn),
            )
            grid = bf16_bits(ttnn.to_torch(y_grid).to(torch.bfloat16))
        except RuntimeError as e:  # recorded, so one failing config doesn't stop the others
            grid = str(e).splitlines()[0][:300]
        return fused, post, grid
    finally:
        ttnn.close_device(device)


def run_mine(binary, x_t, w_t, b_t, shape, act, fmt, cores, acc):
    m, k, n = shape
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bf16_bits(x_t).tofile(tmp / "x.bin")
        w_t.float().numpy().astype(np.float32).tofile(tmp / "w.bin")  # bf16 values, exactly
        bf16_bits(b_t).tofile(tmp / "b.bin")
        cmd = [binary, *(str(tmp / f) for f in ("x.bin", "w.bin", "b.bin", "y.bin"))]
        cmd += [str(m), str(k), str(n), act, fmt, str(cores), acc]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, env=os.environ.copy()
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"linear_bias_act failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
            )
        # ttsim prints its own stats line at exit, after ours.
        line = next(s for s in reversed(proc.stdout.splitlines()) if s.startswith('{"rows"'))
        return np.fromfile(tmp / "y.bin", dtype=np.uint16).reshape(m, n), json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", required=True)
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--K", type=int, default=768)
    p.add_argument("--N", type=int, default=3072)
    p.add_argument(
        "--acts", default="none,gelu,gelu_after_round", help="none, relu, gelu, gelu_after_round"
    )
    p.add_argument("--fmt", choices=("bfp8", "bf16"), default="bfp8")
    p.add_argument("--cores", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--accs", default="fp32,bf16", help="accumulation: fp32 and/or bf16")
    p.add_argument("--skip-ttnn-accs", default="", help="accumulations to skip ttnn.linear for")
    p.add_argument("--skip-ttnn", default="", help="why ttnn.linear is skipped for those")
    p.add_argument("--out", default="results/linear_bias_act.json")
    args = p.parse_args()

    import torch

    torch.manual_seed(args.seed)
    shape = (args.M, args.K, args.N)
    x_t = torch.randn(args.M, args.K).to(torch.bfloat16)
    w_t = (torch.randn(args.K, args.N) * 0.02).to(torch.bfloat16)
    b_t = (torch.randn(args.N) * 0.02).to(torch.bfloat16)
    x, w, b = x_t.float().numpy(), w_t.float().numpy(), b_t.float().numpy()

    out = {
        "arch": os.environ.get("TENSIXFUSE_ARCH", "?"),
        "shape": list(shape),
        "weights": TT_FORMAT[args.fmt],
        "ttnn_skipped": {acc: args.skip_ttnn for acc in args.skip_ttnn_accs.split(",") if acc}
        or None,
        "runs": {},
    }
    skip = set(filter(None, args.skip_ttnn_accs.split(",")))
    for acc, act in ((acc, act) for acc in args.accs.split(",") for act in args.acts.split(",")):
        mine, timing = run_mine(args.binary, x_t, w_t, b_t, shape, act, args.fmt, args.cores, acc)
        ref = reference(x, w, b, args.fmt, act)
        once = bf16_bits(torch.from_numpy(as_bf16(ref)).to(torch.bfloat16)).reshape(mine.shape)
        r = {
            "cores": timing["cores"],
            "seconds": timing["seconds"],
            "dram_read_bytes": timing.get("dram_read_bytes"),
            "pcc_mine_vs_fp32": pcc(bits_to_f32(mine), ref),
            "max_abs_err_mine_vs_fp32": float(np.abs(bits_to_f32(mine) - ref).max()),
            "exact_mine_vs_fp32_rounded_once": float((mine == once).mean()),
        }
        if acc not in skip:
            theirs, post, grid = run_ttnn(x_t, w_t, b_t, args.fmt, act, acc)
            theirs = theirs.reshape(mine.shape)
            ulp = bf16_ulp_distance(mine, theirs)
            r.update(
                {
                    "pcc_ttnn_vs_fp32": pcc(bits_to_f32(theirs), ref),
                    "exact_ttnn_vs_fp32_rounded_once": float((theirs == once).mean()),
                    "bit_exact_vs_ttnn": float((mine == theirs).mean()),
                    "within_1_ulp_vs_ttnn": float((ulp <= 1).mean()),
                    "max_ulp_vs_ttnn": int(ulp.max()),
                }
            )
            if post is not None:
                r["ttnn_gelu_same_as_separate_gelu_op"] = float(
                    (theirs == post.reshape(mine.shape)).mean()
                )
            if isinstance(grid, str):
                r["ttnn_core_grid_error"] = grid
            else:
                grid = grid.reshape(mine.shape)
                r.update(
                    {
                        "exact_ttnn_core_grid_vs_fp32_rounded_once": float((grid == once).mean()),
                        "bit_exact_vs_ttnn_core_grid": float((mine == grid).mean()),
                        "max_ulp_vs_ttnn_core_grid": int(bf16_ulp_distance(mine, grid).max()),
                    }
                )
        r["elements"] = int(mine.size)
        out["runs"][f"{acc}:{act}"] = r
        print(acc, act, json.dumps(r), flush=True)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print("->", dest)


if __name__ == "__main__":
    main()
