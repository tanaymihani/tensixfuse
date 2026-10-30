"""Rebuild docs/RESULTS.md from the JSON files in results/.

    python bench/report.py

Each section says which simulator its numbers came from and which command made
the JSON it reads. Sections whose JSON is missing are skipped.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from tensixfuse.traffic import tiles_to_mib

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def load(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def fusion_section(recs: list[dict]) -> list[str]:
    out = [
        "## Fusion and data movement",
        "",
        (
            "TT-Lang functional simulator (tt-lang-sim 1.1.6). `y = relu(a @ b + c)`, square, bf16. "
            "DRAM traffic is tile reads plus writes from the simulator's trace, at 2 KiB per tile. "
            "Made by `python bench/fusion.py`."
        ),
        "",
    ]
    matched = sum(r["matches_model"] for r in recs)
    pccs = [r["pcc"] for r in recs if "pcc" in r]
    out.append(
        f"{matched} of {len(recs)} runs match the analytic model tile for tile. "
        f"{len(pccs)} runs computed real values; lowest PCC against PyTorch fp32: {min(pccs):.6f}."
    )
    for size in sorted({r["M"] for r in recs}):
        out += [
            "",
            f"### {size}³",
            "",
            "| Data reuse | Cores | Unfused | Fused | Fusion saves |",
            "|---|---|---|---|---|",
        ]
        rows: dict[str, dict] = {}
        for r in recs:
            if r["M"] == size:
                rows.setdefault(r["label"], {})[r["variant"]] = r
        for label, pair in rows.items():
            u, f = pair["unfused"], pair["fused"]
            um, fm = tiles_to_mib(u["measured_tiles"]), tiles_to_mib(f["measured_tiles"])
            out.append(
                f"| {label} | {f['cores']} | {um:.1f} MiB | {fm:.1f} MiB | {1 - fm / um:.0%} |"
            )
    return out


def quant_section(q: dict) -> list[str]:
    out = [
        "## Block-float weights on GPT-2's MLP blocks",
        "",
        (
            f"ttsim, simulated {q['arch'].split('.')[-1].title()}, ttnn 0.79.0, HiFi4. "
            f"All {q['blocks']} MLP blocks of GPT-2 small on {q['tokens']} tokens of WikiText-2; "
            'PCC of each block\'s output against fp32 PyTorch. "Emulated" is the same block on CPU '
            "with weights rounded the way tt-metal packs them. Made by `python bench/quant_gpt2.py`."
        ),
        "",
        "| Weights | MLP weights | Median PCC | Worst PCC (block) | Emulated median | Emulated worst |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in q["configs"].items():
        em = [b["pcc_emulated"] for b in c["blocks"]]
        out.append(
            f"| {name} | {c['weight_mib']:.1f} MiB | {c['median_pcc']:.5f} | "
            f"{c['worst_pcc']:.5f} ({c['worst_block']}) | {statistics.median(em):.5f} | {min(em):.5f} |"
        )
    fid = q["fidelity"]
    out += [
        "",
        f"Math fidelity on block {fid['block']}, the one bfloat4_b hurts most:",
        "",
        "| Weights | LoFi | HiFi2 | HiFi3 | HiFi4 |",
        "|---|---|---|---|---|",
    ]
    for fmt, row in fid["pcc"].items():
        out.append(
            f"| {fmt} | "
            + " | ".join(f"{row[k]:.5f}" for k in ("LoFi", "HiFi2", "HiFi3", "HiFi4"))
            + " |"
        )
    return out


def head_section(h: dict) -> list[str]:
    out = [
        "## GPU-CorruptNet's head",
        "",
        (
            f"ttsim, simulated {h['arch'].split('.')[-1].title()}. Fused `sigmoid(x @ W + b)` over "
            f"{h['frames']} test frames ({h['tile_rows']:.0f} tile rows), calibration temperature "
            f"{h['temperature']:.4f} folded into W and b. Made by `python bench/corruptnet_head.py`."
        ),
        "",
        "| Weights | Label sets match PyTorch | Frames that differ | Logit PCC | Macro-F1 seen | Macro-F1 unseen |",
        "|---|---|---|---|---|---|",
        f"| PyTorch fp32 | | | | {h['pytorch']['macro_f1']['seen']:.4f} | {h['pytorch']['macro_f1']['unseen']:.4f} |",
    ]
    for name, r in h["formats"].items():
        out.append(
            f"| {name} | {r['label_set_agreement']:.2%} | {r['frames_that_differ']} | {r['logit_pcc']:.6f} | "
            f"{r['macro_f1']['seen']:.4f} | {r['macro_f1']['unseen']:.4f} |"
        )
    r = next(iter(h["formats"].values()))
    out += [
        "",
        (
            f"Padded output columns came out as {r['padded_column_values']} on the device. Thresholding before "
            f"slicing them off would give {r['labels_per_frame_if_padding_kept']:.2f} labels per frame "
            f"instead of {r['labels_per_frame']:.2f}."
        ),
    ]
    return out


def bn_section(runs: list[dict]) -> list[str]:
    out = [
        "## TT-Metalium C++: fused BatchNorm + ReLU",
        "",
        (
            "ttsim, built against tt-metal 0.79.0's SDK packages. Input: what goes into one of "
            "GPU-CorruptNet's BatchNorm layers for 32 frames, channels last. In ResNet-50, layer4.2.bn2 "
            "is followed by ReLU; layer4.2.bn3 is followed by the residual add, so BN + ReLU on it is a "
            "stand-in. `fpu` multiplies and adds on the matrix engine with a bf16 intermediate in L1; "
            "`sfpu` does all three steps in fp32 on the vector engine. Compared with NumPy fp32 rounded "
            "once to bf16. ReLU's zeros match trivially, so the nonzero columns count only the outputs "
            "it doesn't zero. Made by `python bench/bn_relu.py --binary build/bn_relu/bn_relu "
            "--layer bn2|bn3 --mode fpu|sfpu`."
        ),
        "",
        (
            "| Layer | Chip | Kernel | Elements | Nonzero outputs | PCC | Bit-exact, all | "
            "Bit-exact, nonzero | Within 1 bf16 step, nonzero | Max error / max output | "
            "Two runs identical |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        nz = r.get("nonzero_outputs")
        nz_cells = (
            f"{nz / r['elements']:.1%} | {r['exact_on_nonzero_vs_single_rounding']:.2%} | "
            f"{r['within_1_ulp_on_nonzero_vs_single_rounding']:.2%}"
            if nz is not None
            else "n/a | n/a | n/a"
        )
        out.append(
            f"| {r.get('layer', 'layer4.2.bn3')} | {r['arch']} | {r['mode']} | {r['elements']:,} | "
            f"{nz_cells.split(' | ')[0]} | {r['pcc']:.6f} | {r['exact_vs_single_rounding']:.2%} | "
            f"{' | '.join(nz_cells.split(' | ')[1:])} | "
            f"{r['max_abs_err_relative_to_max_output']:.2%} | {r['runs_identical']} |"
        )
    return out


RECIPES = {
    "attn=bfloat16+ff13=bfloat16+ff2=bfloat16": "bf16 everywhere",
    "attn=bfloat8_b+ff13=bfloat8_b+ff2=bfloat8_b": "tt_transformers default: all bfloat8_b",
    "attn=bfloat8_b+ff13=bfloat4_b+ff2=bfloat8_b": "tt_transformers performance: FF1/FF3 bfloat4_b",
    "attn=bfloat8_b+ff13=bfloat4_b+ff2=bfloat4_b": "all MLP weights bfloat4_b",
}


def recipe_section(p: dict) -> list[str]:
    out = [
        f"## {p['model']}: tt_transformers' weight recipes, WikiText-2 perplexity",
        "",
        (
            f"CPU emulation, weight-only, like the GPT-2 study. Whole test set, {p['test_tokens']:,} "
            f"tokens in {p['windows']} windows of {p['window']}. FF1/FF3 are the gate and up "
            "projections, FF2 the down projection; attention is q, k, v and o. Made by "
            "`python bench/perplexity_llama.py`."
        ),
        "",
        "| Recipe | Bits per weight | Perplexity | vs fp32 |",
        "|---|---|---|---|",
    ]
    for name, c in sorted(p["configs"].items(), key=lambda kv: kv[1]["ppl"]):
        out.append(
            f"| {RECIPES.get(name, name)} | {c['bits_per_weight']:.2f} | {c['ppl']:.3f} | "
            f"{c['ppl_increase_pct']:+.2f}% |"
        )
    return out


def perplexity_section(p: dict) -> list[str]:
    out = [
        "## GPT-2 small: WikiText-2 perplexity with low-bit weights",
        "",
        (
            f"CPU emulation, weight-only: each format rounds the chosen weights the way it stores them "
            f"and everything else stays fp32. Whole test set, {p['test_tokens']:,} tokens in "
            f"{p['windows']} non-overlapping windows of {p['window']}. `mlp` rounds c_fc and c_proj in "
            "all 12 blocks; `all` adds attention's c_attn and c_proj. Bits per weight include the "
            "scales. Made by `python bench/perplexity.py` (torch "
            f"{p['torch']}, transformers {p['transformers']})."
        ),
        "",
        "| Weights | Rounded | Bits per weight | Perplexity | vs fp32 | Weight error (rel. RMS) |",
        "|---|---|---|---|---|---|",
    ]
    configs = p["configs"]
    order = sorted(
        configs,
        key=lambda n: (n != "fp32", n.split(":")[0] if ":" in n else "", configs[n]["ppl"]),
    )
    for name in order:
        c = configs[name]
        scope, fmt = name.split(":") if ":" in name else ("", name)
        scope = {"mlp": "MLP", "all": "all linear", "": "nothing"}[scope]
        out.append(
            f"| {fmt.replace('/', ' c_fc, ') + (' c_proj' if '/' in fmt else '')} | {scope} | "
            f"{c['bits_per_weight']:.2f} | {c['ppl']:.3f} | {c['ppl_increase_pct']:+.2f}% | "
            f"{c['weight_rel_rmse']:.4f} |"
        )
    return out


def compiled_section(runs: list[dict]) -> list[str]:
    out = [
        "## The fusion kernels through the TT-Lang compiler",
        "",
        (
            "ttsim. The same kernels as the fusion study, compiled by the TT-Lang compiler instead of "
            "run in the functional simulator, at 256³, `y = relu(a @ b + c)`, bf16, checked against "
            "PyTorch. Made by `python bench/ttlang_compiled.py`."
        ),
        "",
        "| tt-lang | Chip | Data reuse | Variant | Compiled and ran | PCC | Max abs error |",
        "|---|---|---|---|---|---|---|",
    ]
    for res in runs:
        for r in res["runs"]:
            ok = "yes" if r["ok"] else "no"
            pcc = f"{r['pcc']:.6f}" if "pcc" in r else "n/a"
            err = f"{r['max_abs_err']:.3g}" if "max_abs_err" in r else "n/a"
            out.append(
                f"| {res['tt_lang']} | {res['arch']} | {r['label']} | {r['variant']} | {ok} | "
                f"{pcc} | {err} |"
            )
    return out


def main() -> None:
    parts = [
        "# Results",
        "",
        (
            "Generated by `python bench/report.py` from the JSON in `results/`. "
            "Nothing here ran on silicon, and no device timings are reported: ttsim is bit-exact "
            "but doesn't model time."
        ),
        "",
    ]
    if fusion := load("fusion.json"):
        parts += fusion_section(fusion) + [""]
    if quant := load("quant_gpt2.json"):
        parts += quant_section(quant) + [""]
    if head := load("corruptnet_head.json"):
        parts += head_section(head) + [""]
    if ppl := load("perplexity.json"):
        parts += perplexity_section(ppl) + [""]
    if qwen := load("perplexity_qwen.json"):
        parts += recipe_section(qwen) + [""]
    bn_runs = [
        load(f"bn_relu{layer}_{a}_{m}.json")
        for layer in ("_bn2", "")
        for a in ("wormhole", "blackhole")
        for m in ("fpu", "sfpu")
    ]
    if any(bn_runs):
        parts += bn_section([r for r in bn_runs if r]) + [""]
    compiled = [load(f"ttlang_compiled_{a}.json") for a in ("wormhole", "blackhole")]
    if any(compiled):
        parts += compiled_section([r for r in compiled if r]) + [""]
    dest = REPO / "docs" / "RESULTS.md"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text("\n".join(parts))
    print("->", dest)


if __name__ == "__main__":
    main()
