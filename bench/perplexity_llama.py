"""tt_transformers' weight-precision recipes on a Llama-style model: WikiText-2
perplexity of Qwen2.5-0.5B with its weights rounded the way each recipe stores them.

tt_transformers (tt-metal's models/tt_transformers/tt/model_config.py) keeps
every linear weight in bfloat8_b by default, and its "performance" setting
moves the MLP's FF1 and FF3 (gate and up projections) to bfloat4_b. It makes
an exception for Qwen2.5-7B, which it says degrades under that setting, and
keeps its MLP in bfloat8_b and attention in bf16. This checks the recipes on a
small model of the same family, the same way bench/perplexity.py does GPT-2:
weight-only, everything else fp32, whole test set in 1024-token windows.

A config is a list of group=format pairs joined by "+"; groups are attn (q, k,
v, o), ff13 (gate and up) and ff2 (down). "fp32" changes nothing.

    python bench/perplexity_llama.py --configs fp32,attn=bfloat8_b+ff13=bfloat4_b+ff2=bfloat8_b
    python bench/perplexity_llama.py --merge shards/*.json --out results/perplexity_qwen.json
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
from perplexity import WINDOW, evaluate

from tensixfuse.formats import BITS_PER_WEIGHT, round_trip

MODEL = "Qwen/Qwen2.5-0.5B"
GROUPS = {
    "attn": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"),
    "ff13": ("mlp.gate_proj", "mlp.up_proj"),
    "ff2": ("mlp.down_proj",),
}


def layer_formats(config: str) -> dict[str, str]:
    if config == "fp32":
        return {}
    out = {}
    for part in config.split("+"):
        group, fmt = part.split("=")
        if group not in GROUPS or fmt not in BITS_PER_WEIGHT:
            raise ValueError(f"bad part {part!r} in {config!r}")
        out.update(dict.fromkeys(GROUPS[group], fmt))
    return out


def merge(paths: list[str], dest: str) -> None:
    shards = [json.loads(Path(s).read_text()) for s in paths]
    merged = {k: v for k, v in shards[0].items() if k != "configs"}
    merged["configs"] = {}
    for s in shards:
        if s["test_tokens"] != merged["test_tokens"] or s["windows"] != merged["windows"]:
            raise SystemExit("shards disagree on the test set")
        merged["configs"].update(s["configs"])
    base = merged["configs"]["fp32"]["ppl"]
    for c in merged["configs"].values():
        c["ppl_increase_pct"] = 100 * (c["ppl"] / base - 1)
    Path(dest).write_text(json.dumps(merged, indent=1))
    lines = [
        (
            f"{merged['model']}, WikiText-2 test ({merged['test_tokens']:,} tokens, "
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
    p.add_argument("--configs", default="fp32")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default="results/perplexity_qwen.json")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--max-windows", type=int, default=0, help="for a quick check; 0 = all")
    p.add_argument("--merge", nargs="+", help="shard JSONs to merge into --out")
    args = p.parse_args()
    if args.merge:
        merge(args.merge, args.out)
        return

    import datasets
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 1)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    test = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[0]
    n = ids.numel() // WINDOW
    if args.max_windows:
        n = min(n, args.max_windows)
    windows = ids[: n * WINDOW].view(n, WINDOW)

    names = [name for group in GROUPS.values() for name in group]
    modules = {
        f"layers.{i}.{name}": layer.get_submodule(name)
        for i, layer in enumerate(model.model.layers)
        for name in names
    }
    original = {key: m.weight.detach().clone() for key, m in modules.items()}

    out = {
        "model": args.model,
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
            # nn.Linear stores [out, in]; round_trip wants the matmul's [K, N].
            w = original[key].numpy()
            q = round_trip(w.T, fmt).T
            m.weight.data.copy_(torch.from_numpy(np.ascontiguousarray(q)))
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
        print(f"{config:48s} ppl {c['ppl']:8.3f}  {c['seconds']}s", flush=True)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print("->", dest)


if __name__ == "__main__":
    main()
