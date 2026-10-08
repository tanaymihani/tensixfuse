"""Block-float weights on GPT-2's 12 MLP blocks, on ttsim.

For every block and weight format, runs c_fc (+ fused GELU) and c_proj with
TTNN on the simulated chip and compares the block's output with fp32 PyTorch.
Next to each device number is the same computation done on CPU with the
weights rounded exactly the way tt-metal packs them (tensixfuse.blockfloat):
that isolates what the format itself costs from the chip's arithmetic.
Then sweeps math fidelity on the block that bfloat4_b hurts most.

    python bench/quant_gpt2.py --data data/gpt2_mlp.npz --out results/quant_gpt2.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

from tensixfuse.blockfloat import as_bf16, quantize_weight, zeroed_fraction
from tensixfuse.metrics import pcc
from tensixfuse.traffic import TILE_BYTES

CONFIGS = {
    "bf16": ("bfloat16", "bfloat16"),
    "bfloat8_b": ("bfloat8_b", "bfloat8_b"),
    "bfloat4_b": ("bfloat4_b", "bfloat4_b"),
    "bfloat4_b c_fc, bfloat8_b c_proj": ("bfloat4_b", "bfloat8_b"),
    "bfloat8_b c_fc, bfloat4_b c_proj": ("bfloat8_b", "bfloat4_b"),
}
TT_DTYPE = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "bfloat4_b": ttnn.bfloat4_b}
FIDELITIES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")


def kernel_config(device, fidelity: str):
    fid = getattr(ttnn.MathFidelity, fidelity)
    kwargs = {
        "math_fidelity": fid,
        "math_approx_mode": False,
        "fp32_dest_acc_en": True,
        "packer_l1_acc": False,
    }
    try:
        return ttnn.init_device_compute_kernel_config(device.arch(), **kwargs)
    except (AttributeError, TypeError):
        return ttnn.WormholeComputeKernelConfig(**kwargs)


def to_dev(a: np.ndarray, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(a, np.float32)),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def run_on_device(device, d, i: int, fmt_fc: str, fmt_proj: str, fidelity: str) -> np.ndarray:
    cfg = kernel_config(device, fidelity)
    x = to_dev(d[f"x{i}"], device)
    w_fc = to_dev(d[f"w_fc{i}"], device, TT_DTYPE[fmt_fc])
    b_fc = to_dev(d[f"b_fc{i}"].reshape(1, -1), device)
    w_proj = to_dev(d[f"w_proj{i}"], device, TT_DTYPE[fmt_proj])
    b_proj = to_dev(d[f"b_proj{i}"].reshape(1, -1), device)
    h = ttnn.linear(x, w_fc, bias=b_fc, activation="gelu", compute_kernel_config=cfg)
    y = ttnn.linear(h, w_proj, bias=b_proj, compute_kernel_config=cfg)
    return ttnn.to_torch(y).float().numpy()


def emulate(d, i: int, fmt_fc: str, fmt_proj: str) -> np.ndarray:
    """Same block on CPU: bf16 activations, weights rounded like tt-metal does."""
    x = as_bf16(d[f"x{i}"])
    h = gelu_tanh(x @ quantize_weight(d[f"w_fc{i}"], fmt_fc) + as_bf16(d[f"b_fc{i}"]))
    return as_bf16(h) @ quantize_weight(d[f"w_proj{i}"], fmt_proj) + as_bf16(d[f"b_proj{i}"])


def weight_mib(d, n_blocks: int, fmt_fc: str, fmt_proj: str) -> float:
    total = 0
    for i in range(n_blocks):
        for key, fmt in ((f"w_fc{i}", fmt_fc), (f"w_proj{i}", fmt_proj)):
            k, n = d[key].shape
            total += (k // 32) * (-(-n // 32)) * TILE_BYTES[fmt]
    return total / 2**20


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gpt2_mlp.npz")
    p.add_argument("--out", default="results/quant_gpt2.json")
    p.add_argument("--blocks", type=int, default=12)
    args = p.parse_args()
    d = dict(np.load(args.data))
    n = args.blocks

    results: dict = {"blocks": n, "tokens": int(d["x0"].shape[0]), "configs": {}, "weights": []}
    for i in range(n):
        results["weights"].append(
            {
                "block": i,
                "c_fc_zeroed_by_bfloat4_b": zeroed_fraction(d[f"w_fc{i}"], "bfloat4_b"),
                "c_proj_zeroed_by_bfloat4_b": zeroed_fraction(d[f"w_proj{i}"], "bfloat4_b"),
                "c_fc_abs_max": float(np.abs(d[f"w_fc{i}"]).max()),
                "c_proj_abs_max": float(np.abs(d[f"w_proj{i}"]).max()),
            }
        )

    device = ttnn.open_device(device_id=0)
    try:
        results["arch"] = str(device.arch())
        for name, (fmt_fc, fmt_proj) in CONFIGS.items():
            rows = []
            t0 = time.perf_counter()
            for i in range(n):
                y_dev = run_on_device(device, d, i, fmt_fc, fmt_proj, "HiFi4")
                rows.append(
                    {
                        "block": i,
                        "pcc_device": pcc(y_dev, d[f"y{i}"]),
                        "pcc_emulated": pcc(emulate(d, i, fmt_fc, fmt_proj), d[f"y{i}"]),
                    }
                )
            dev = np.array([r["pcc_device"] for r in rows])
            results["configs"][name] = {
                "c_fc": fmt_fc,
                "c_proj": fmt_proj,
                "weight_mib": weight_mib(d, n, fmt_fc, fmt_proj),
                "median_pcc": float(np.median(dev)),
                "worst_pcc": float(dev.min()),
                "worst_block": int(dev.argmin()),
                "blocks": rows,
                "seconds": round(time.perf_counter() - t0, 1),
            }
            c = results["configs"][name]
            print(
                f"{name:34s} {c['weight_mib']:6.1f} MiB  median {c['median_pcc']:.5f}  "
                f"worst {c['worst_pcc']:.5f} (block {c['worst_block']})  {c['seconds']}s",
                flush=True,
            )

        worst = results["configs"]["bfloat4_b"]["worst_block"]
        sweep = {}
        for fmt in ("bfloat16", "bfloat8_b", "bfloat4_b"):
            sweep[fmt] = {
                fid: pcc(run_on_device(device, d, worst, fmt, fmt, fid), d[f"y{worst}"])
                for fid in FIDELITIES
            }
            print("fidelity", fmt, {k: round(v, 5) for k, v in sweep[fmt].items()}, flush=True)
        results["fidelity"] = {"block": worst, "pcc": sweep}
    finally:
        ttnn.close_device(device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
