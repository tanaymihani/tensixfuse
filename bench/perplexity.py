"""GPT-2 small's WikiText-2 perplexity with its weights in low-bit formats.

Weight-only emulation on CPU. Each format rounds the chosen weights the way it
stores them, and everything else (activations, biases, embeddings, LayerNorm,
the LM head) stays fp32:

- bfloat8_b / bfloat4_b: tensixfuse.blockfloat, which matches tt-metal's
  packer bit for bit; groups of 16 along a tile row, as ttnn.linear's weight
  is laid out. bfloat4_b_k is a what-if: the same format with the weight
  stored transposed, so groups run along K instead.
- MXFP8, MXFP4, NVFP4, INT4 g128: tensixfuse.formats, grouped along K.

Scopes: "mlp" is c_fc and c_proj in all 12 blocks (the weights
bench/quant_gpt2.py runs on ttsim); "all" adds attention's c_attn and c_proj.
Perplexity is over the whole WikiText-2 test set (raw, joined with blank lines)
in non-overlapping 1024-token windows, as quantization papers usually report it.

    python bench/perplexity.py --configs fp32,mlp:bfloat8_b --out shards/a.json
    python bench/perplexity.py --merge shards/*.json --out results/perplexity.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

from tensixfuse.formats import BITS_PER_WEIGHT, round_trip

LAYERS = {
    "mlp": ("mlp.c_fc", "mlp.c_proj"),
    "all": ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"),
}
WINDOW = 1024


def layer_formats(config: str) -> dict[str, str]:
    """'mlp:bfloat8_b' -> every MLP weight in bfloat8_b; 'mlp:bfloat4_b/bfloat8_b'
    -> c_fc in bfloat4_b, c_proj in bfloat8_b; 'fp32' -> nothing changes."""
    if config == "fp32":
        return {}
    scope, fmts = config.split(":")
    parts = fmts.split("/")
    for fmt in parts:
        if fmt not in BITS_PER_WEIGHT:
            raise ValueError(f"unknown format {fmt!r} in {config!r}")
    if len(parts) == 1:
        return dict.fromkeys(LAYERS[scope], parts[0])
    if scope != "mlp" or len(parts) != 2:
        raise ValueError(f"a split needs mlp scope and two formats: {config!r}")
    return dict(zip(LAYERS["mlp"], parts, strict=True))


def evaluate(model, windows, batch: int) -> tuple[float, int]:
    import torch
    import torch.nn.functional as F

    nll, count = 0.0, 0
    with torch.inference_mode():
        for s in range(0, len(windows), batch):
            x = windows[s : s + batch]
            logits = model(x).logits[:, :-1]
            target = x[:, 1:]
            nll += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum"
            ).item()
            count += target.numel()
    return nll, count


def run(args) -> None:
    import datasets
    import torch
    import transformers
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    torch.set_num_threads(os.cpu_count() or 1)
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    test = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[0]
    n = ids.numel() // WINDOW
    if args.max_windows:
        n = min(n, args.max_windows)
    windows = ids[: n * WINDOW].view(n, WINDOW)

    modules = {
        f"h.{i}.{name}": block.get_submodule(name)
        for i, block in enumerate(model.transformer.h)
        for name in LAYERS["all"]
    }
    original = {key: m.weight.detach().clone() for key, m in modules.items()}

    out = {
        "model": "openai-community/gpt2",
        "dataset": "wikitext-2-raw-v1 test",
        "test_tokens": int(ids.numel()),
        "window": WINDOW,
        "windows": n,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cpu": platform.processor() or platform.machine(),
        "configs": {},
    }
    for config in args.configs.split(","):
        t0 = time.perf_counter()
        fmts = layer_formats(config)
        params = bits = 0
        err2 = ref2 = 0.0
        for key, m in modules.items():
            fmt = fmts.get(key.split(".", 2)[2])
            if fmt is None:
                continue
            w = original[key].numpy()
            q = round_trip(w, fmt)
            m.weight.data.copy_(torch.from_numpy(q))
            params += w.size
            bits += w.size * BITS_PER_WEIGHT[fmt]
            err2 += float(np.square(q.astype(np.float64) - w).sum())
            ref2 += float(np.square(w.astype(np.float64)).sum())
        nll, count = evaluate(model, windows, args.batch)
        for key, m in modules.items():
            m.weight.data.copy_(original[key])
        out["configs"][config] = {
            "ppl": math.exp(nll / count),
            "nll": nll,
            "predicted_tokens": count,
            "params_quantized": params,
            "bits_per_weight": bits / params if params else 32.0,
            "quantized_mib": bits / 8 / 2**20,
            "weight_rel_rmse": math.sqrt(err2 / ref2) if ref2 else 0.0,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        c = out["configs"][config]
        print(
            f"{config:28s} ppl {c['ppl']:8.3f}  {c['bits_per_weight']:5.2f} b/w  "
            f"weight rmse {c['weight_rel_rmse']:.4f}  {c['seconds']}s",
            flush=True,
        )

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print("->", dest)


def merge(args) -> None:
    shards = [json.loads(Path(p).read_text()) for p in args.merge]
    first = shards[0]
    for s in shards[1:]:
        for key in ("test_tokens", "window", "windows"):
            if s[key] != first[key]:
                raise SystemExit(f"shards disagree on {key}: {s[key]} vs {first[key]}")
    merged = {k: v for k, v in first.items() if k != "configs"}
    merged["configs"] = {}
    for s in shards:
        merged["configs"].update(s["configs"])
    if "fp32" not in merged["configs"]:
        raise SystemExit("no fp32 baseline among the shards")
    base = merged["configs"]["fp32"]["ppl"]
    for c in merged["configs"].values():
        c["ppl_increase_pct"] = 100 * (c["ppl"] / base - 1)
    Path(args.out).write_text(json.dumps(merged, indent=1))

    lines = [
        (
            f"GPT-2 small, WikiText-2 test ({merged['test_tokens']:,} tokens, "
            f"{merged['windows']} windows of {merged['window']})"
        ),
        "",
        "| weights | bits/weight | perplexity | vs fp32 |",
        "|---|---:|---:|---:|",
    ]
    for name, c in sorted(merged["configs"].items(), key=lambda kv: kv[1]["ppl"]):
        lines.append(
            f"| {name} | {c['bits_per_weight']:.2f} | {c['ppl']:.3f} | "
            f"{c['ppl_increase_pct']:+.2f}% |"
        )
    print("\n".join(lines))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="fp32", help="comma-separated, e.g. fp32,all:mxfp4")
    p.add_argument("--out", default="results/perplexity.json")
    p.add_argument("--batch", type=int, default=2, help="windows per forward pass")
    p.add_argument("--max-windows", type=int, default=0, help="for a quick check; 0 = all")
    p.add_argument("--merge", nargs="+", help="shard JSONs to merge into --out")
    args = p.parse_args()
    if args.merge:
        merge(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
