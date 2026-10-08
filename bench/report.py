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
            "ttsim, built against tt-metal 0.79.0's SDK packages. Input: what goes into GPU-CorruptNet's "
            "layer4.2.bn3 for 32 frames, channels last. `fpu` multiplies and adds on the matrix engine with "
            "a bf16 intermediate in L1; `sfpu` does all three steps in fp32 on the vector engine. "
            "Compared with NumPy fp32 rounded once to bf16. "
            "Made by `python bench/bn_relu.py --binary build/bn_relu/bn_relu --mode fpu|sfpu`."
        ),
        "",
        (
            "| Chip | Kernel | Elements | Cores | PCC | Bit-exact | Within 1 bf16 step | "
            "Max error / max output | Two runs identical |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        out.append(
            f"| {r['arch']} | {r['mode']} | {r['elements']:,} | {r['cores']} | {r['pcc']:.6f} | "
            f"{r['exact_vs_single_rounding']:.2%} | {r['within_1_ulp_vs_single_rounding']:.2%} | "
            f"{r['max_abs_err_relative_to_max_output']:.2%} | {r['runs_identical']} |"
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
    bn_runs = [
        load(f"bn_relu_{a}_{m}.json") for a in ("wormhole", "blackhole") for m in ("fpu", "sfpu")
    ]
    if any(bn_runs):
        parts += bn_section([r for r in bn_runs if r]) + [""]
    dest = REPO / "docs" / "RESULTS.md"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text("\n".join(parts))
    print("->", dest)


if __name__ == "__main__":
    main()
