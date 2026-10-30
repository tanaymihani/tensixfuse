"""Draw the README's figures from results/*.json, in light and dark versions.

    uv run --no-project --with matplotlib --with numpy python docs/make_figures.py

Writes docs/img/:
- staircase-{light,dark}.png: DRAM traffic per reuse level (results/fusion.json)
- perplexity-{light,dark}.png: WikiText-2 perplexity vs bits per weight
  (results/perplexity.json, skipped until it exists)
- tile-trip-{light,dark}.svg: how a tile moves through a Tensix core
- blockfloat-{light,dark}.svg: what a 32x32 tile costs in each format
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tensixfuse.traffic import TILE_BYTES

OUT = REPO / "docs" / "img"
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "plane": "#f3f2ee",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "grid": "#e4e3df",
        "s1": "#2a78d6",
        "s2": "#eb6834",
        "s3": "#1baf7a",
        "wash": "#e3eefb",
    },
    "dark": {
        "surface": "#1a1a19",
        "plane": "#262624",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "grid": "#383835",
        "s1": "#3987e5",
        "s2": "#d95926",
        "s3": "#199e70",
        "wash": "#1d2f45",
    },
}
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    ax.yaxis.grid(True, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)


def staircase() -> None:
    runs = json.loads((REPO / "results" / "fusion.json").read_text())
    at = [r for r in runs if r["M"] == r["K"] == r["N"] == 1024]
    pairs: dict = {}
    for r in at:
        pairs.setdefault((r["label"], r["cores"], r["mcast"]), {})[r["variant"]] = r
    rows = sorted(
        pairs.items(),
        key=lambda kv: (kv[0][1], kv[0][2], -kv[1]["fused"]["dram"]["total"]),
    )
    mib = 2048 / 2**20
    fused = [v["fused"]["dram"]["total"] * mib for _, v in rows]
    unfused = [v["unfused"]["dram"]["total"] * mib for _, v in rows]
    floor = rows[0][1]["fused"]["compulsory_tiles"] * mib
    labels = []
    for (label, cores, mcast), _ in rows:
        b = label.split()[0].replace("x", "×")
        if cores == 1:
            labels.append(f"{b} blocks\n1 core")
        else:
            labels.append(f"{b} per core\n64 cores" + ("\n+ multicast" if mcast else ""))

    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
        fig.patch.set_facecolor(t["surface"])
        style_axes(ax, t)
        x = range(len(rows))
        ax.bar(
            x,
            fused,
            width=0.42,
            color=t["s1"],
            edgecolor=t["surface"],
            linewidth=2,
            label="fused: y = relu(a @ b + c) in one kernel",
        )
        ax.bar(
            x,
            [u - f for u, f in zip(unfused, fused, strict=True)],
            bottom=fused,
            width=0.42,
            color=t["s2"],
            edgecolor=t["surface"],
            linewidth=2,
            label="extra if unfused (add and ReLU as separate ops)",
        )
        ax.axhline(
            floor,
            color=t["ink2"],
            linewidth=1,
            label=f"compulsory floor: {floor:.0f} MiB, each tensor moved once",
        )
        for i, (f, u) in enumerate(zip(fused, unfused, strict=True)):
            ax.text(
                i,
                u + 2.5,
                f"{f:.0f}",
                ha="center",
                va="bottom",
                fontsize=11,
                color=t["ink"],
                fontweight="bold",
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8.5, color=t["ink2"])
        ax.set_ylabel("DRAM traffic, MiB", fontsize=9.5, color=t["ink2"])
        ax.set_ylim(0, max(unfused) * 1.2)
        fig.suptitle(
            f"Reuse, not fusion, does most of the work: {fused[0]:.0f} → {fused[-1]:.0f} MiB",
            x=0.06,
            ha="left",
            fontsize=14,
            fontweight="bold",
            color=t["ink"],
        )
        ax.set_title(
            "Fused matmul + bias + ReLU at 1024³, bf16, on TT-Lang's functional simulator. "
            "Bold numbers: fused traffic",
            loc="left",
            fontsize=8.5,
            color=t["ink2"],
        )
        leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        for text in leg.get_texts():
            text.set_color(t["ink"])
        fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.22)
        fig.savefig(OUT / f"staircase-{mode}.png", facecolor=t["surface"])
        plt.close(fig)


FAMILY = {  # format -> (family, label)
    "bfloat8_b": ("Tenstorrent block float", "bfloat8_b"),
    "bfloat4_b": ("Tenstorrent block float", "bfloat4_b"),
    "bfloat4_b_k": ("Tenstorrent block float", "bfloat4_b,\ngroups along K"),
    "mxfp8": ("OCP MX / NVFP4", "MXFP8"),
    "mxfp4": ("OCP MX / NVFP4", "MXFP4"),
    "nvfp4": ("OCP MX / NVFP4", "NVFP4"),
    "int4_g128": ("INT4, group of 128", "INT4 g128"),
    "bfloat16": ("bfloat16", "bfloat16"),
}


def perplexity() -> None:
    src = REPO / "results" / "perplexity.json"
    if not src.exists():
        print("no results/perplexity.json yet; skipping that chart")
        return
    res = json.loads(src.read_text())
    base = res["configs"]["fp32"]["ppl"]
    points = [
        (c["bits_per_weight"], c["ppl"], name.split(":")[1])
        for name, c in res["configs"].items()
        if name.startswith("all:")
    ]
    families = list(dict.fromkeys(FAMILY[f][0] for _, _, f in points if f != "bfloat16"))
    for mode, t in THEMES.items():
        colors = dict(zip(families, (t["s1"], t["s2"], t["s3"]), strict=False))
        colors["bfloat16"] = t["ink2"]
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
        fig.patch.set_facecolor(t["surface"])
        style_axes(ax, t)
        ax.xaxis.grid(False)
        ax.axhline(base, color=t["ink2"], linewidth=1)
        ax.text(
            16.6, base, f"fp32 {base:.2f}", va="center", ha="left", fontsize=8.5, color=t["ink2"]
        )
        seen = set()
        for bits, ppl, fmt in sorted(points):
            fam, label = FAMILY[fmt]
            ax.scatter(
                bits,
                ppl,
                s=70,
                color=colors[fam],
                edgecolor=t["surface"],
                linewidth=2,
                zorder=3,
                label=None if fam in seen else fam,
            )
            seen.add(fam)
            ax.annotate(
                f"{label}  {ppl:.2f}",
                (bits, ppl),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=t["ink"],
            )
        ax.set_yscale("log")
        ax.set_xlim(3.5, 18.5)
        ax.set_xlabel("bits per weight, scales included", fontsize=9.5, color=t["ink2"])
        ax.set_ylabel("perplexity (log scale)", fontsize=9.5, color=t["ink2"])
        fig.suptitle(
            "GPT-2 small on WikiText-2: every linear weight in the 12 blocks rounded",
            x=0.06,
            ha="left",
            fontsize=13,
            fontweight="bold",
            color=t["ink"],
        )
        ax.set_title(
            f"Full test set, {res['windows']} windows of {res['window']} tokens; "
            "weight-only, everything else fp32. Lower is better",
            loc="left",
            fontsize=8.5,
            color=t["ink2"],
        )
        leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        for text in leg.get_texts():
            text.set_color(t["ink"])
        fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.13)
        fig.savefig(OUT / f"perplexity-{mode}.png", facecolor=t["surface"])
        plt.close(fig)


def svg(width: int, height: int, t: dict, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{title}">\n'
        f"<title>{title}</title>\n"
        f'<rect width="{width}" height="{height}" fill="{t["surface"]}"/>\n'
        f'<g font-family="{FONT}">\n{body}</g>\n</svg>\n'
    )


def box(x, y, w, h, t, title, sub, highlight=False):
    fill = t["wash"] if highlight else t["plane"]
    stroke = t["s1"] if highlight else t["grid"]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{2 if highlight else 1}"/>\n'
        f'<text x="{x + w / 2}" y="{y + h / 2 - 3}" text-anchor="middle" font-size="15" '
        f'font-weight="600" fill="{t["ink"]}">{title}</text>\n'
        f'<text x="{x + w / 2}" y="{y + h / 2 + 15}" text-anchor="middle" font-size="11" '
        f'fill="{t["ink2"]}">{sub}</text>\n'
    )


def arrow(x1, y, x2, t, label=""):
    out = (
        f'<line x1="{x1}" y1="{y}" x2="{x2 - 7}" y2="{y}" stroke="{t["ink2"]}" '
        f'stroke-width="1.5"/>\n'
        f'<path d="M{x2 - 8},{y - 4.5} L{x2},{y} L{x2 - 8},{y + 4.5} Z" fill="{t["ink2"]}"/>\n'
    )
    if label:
        out += (
            f'<text x="{(x1 + x2) / 2}" y="{y - 9}" text-anchor="middle" font-size="11" '
            f'fill="{t["ink2"]}">{label}</text>\n'
        )
    return out


def dram(x, y, t, sub):
    return (
        f'<path d="M{x},{y + 10} v54 a50,10 0 0 0 100,0 v-54" fill="{t["plane"]}" '
        f'stroke="{t["grid"]}"/>\n'
        f'<ellipse cx="{x + 50}" cy="{y + 10}" rx="50" ry="10" fill="{t["plane"]}" '
        f'stroke="{t["grid"]}"/>\n'
        f'<text x="{x + 50}" y="{y + 44}" text-anchor="middle" font-size="15" font-weight="600" '
        f'fill="{t["ink"]}">DRAM</text>\n'
        f'<text x="{x + 50}" y="{y + 61}" text-anchor="middle" font-size="11" '
        f'fill="{t["ink2"]}">{sub}</text>\n'
    )


def tile_trip() -> None:
    stages = [
        ("reader", "data movement core"),
        ("buffer", "L1 SRAM"),
        ("unpack", "TRISC 0"),
        ("math", "FPU · SFPU"),
        ("pack", "TRISC 2"),
        ("buffer", "L1 SRAM"),
        ("writer", "data movement core"),
    ]
    for mode, t in THEMES.items():
        y, h, w, gap, x0 = 88, 60, 104, 18, 216
        body = (
            f'<rect x="{x0 - 20}" y="30" width="{7 * w + 6 * gap + 40}" height="160" rx="14" '
            f'fill="none" stroke="{t["ink2"]}" stroke-width="1"/>\n'
            f'<text x="{x0 - 4}" y="56" font-size="13" font-weight="600" fill="{t["ink2"]}">'
            f"one Tensix core (a chip has a grid of them; the study uses up to 64)</text>\n"
        )
        body += dram(24, y - 12, t, "a, b, c")
        body += arrow(124, y + h / 2, x0, t, "NoC read")
        for i, (title, sub) in enumerate(stages):
            x = x0 + i * (w + gap)
            body += box(x, y, w, h, t, title, sub, highlight=(title == "math"))
            if i < len(stages) - 1:
                body += arrow(x + w, y + h / 2, x + w + gap, t)
        end = x0 + 7 * w + 6 * gap
        body += arrow(end, y + h / 2, end + 92, t, "NoC write")
        body += dram(end + 92, y - 12, t, "y")
        mx = x0 + 3 * (w + gap) + w / 2
        body += (
            f'<line x1="{mx}" y1="{y + h}" x2="{mx}" y2="{y + h + 52}" stroke="{t["s1"]}" '
            f'stroke-width="1.5"/>\n'
            f'<text x="{mx}" y="{y + h + 70}" text-anchor="middle" font-size="12.5" '
            f'fill="{t["ink"]}">fused: + bias and activation happen here, before pack</text>\n'
        )
        width = end + 92 + 100 + 24
        (OUT / f"tile-trip-{mode}.svg").write_text(
            svg(width, 262, t, body, "How a tile moves through a Tensix core")
        )


def swatch(x, y, color) -> str:
    return f'<rect x="{x}" y="{y}" width="11" height="11" rx="2" fill="{color}"/>\n'


def blockfloat() -> None:
    fmts = [
        ("bfloat16", TILE_BYTES["bfloat16"], 0, "16 bits per value"),
        ("bfloat8_b", TILE_BYTES["bfloat8_b"], 64, "sign + 7-bit mantissa, 8.5 bits per value"),
        ("bfloat4_b", TILE_BYTES["bfloat4_b"], 64, "sign + 3-bit mantissa, 4.5 bits per value"),
    ]
    for mode, t in THEMES.items():
        cell, gx, gy = 7, 40, 70
        body = ""
        # The tile: 32x32 values as four 16x16 faces; one face row highlighted.
        for fy in range(2):
            for fx in range(2):
                x = gx + fx * (16 * cell + 6)
                yy = gy + fy * (16 * cell + 6)
                body += (
                    f'<rect x="{x}" y="{yy}" width="{16 * cell}" height="{16 * cell}" '
                    f'fill="{t["plane"]}" stroke="{t["grid"]}"/>\n'
                )
                for r in range(1, 16):
                    body += (
                        f'<line x1="{x}" y1="{yy + r * cell}" x2="{x + 16 * cell}" '
                        f'y2="{yy + r * cell}" stroke="{t["grid"]}" stroke-width="0.5"/>\n'
                    )
        body += (
            f'<rect x="{gx}" y="{gy + 3 * cell}" width="{16 * cell}" height="{cell}" '
            f'fill="{t["s1"]}"/>\n'
            f'<text x="{gx}" y="{gy - 14}" font-size="13" font-weight="600" fill="{t["ink"]}">'
            f"one 32×32 tile = four 16×16 faces</text>\n"
            + swatch(gx, gy + 2 * (16 * cell) + 22, t["s1"])
            + f'<text x="{gx + 16}" y="{gy + 2 * (16 * cell) + 32}" font-size="12" '
            f'fill="{t["ink"]}">16 values in a face row share one exponent</text>\n'
        )
        bx, scale = 360, 0.27
        for i, (name, total, exp_bytes, sub) in enumerate(fmts):
            y = gy + 8 + i * 74
            w_total = total * scale
            body += (
                f'<text x="{bx}" y="{y - 6}" font-size="13" font-weight="600" fill="{t["ink"]}">'
                f'{name} <tspan font-weight="400" fill="{t["ink2"]}">{sub}</tspan></text>\n'
            )
            if exp_bytes:
                w_exp = exp_bytes * scale
                body += (
                    f'<rect x="{bx}" y="{y}" width="{w_exp - 1}" height="26" rx="3" '
                    f'fill="{t["s2"]}"/>\n'
                    f'<rect x="{bx + w_exp + 1}" y="{y}" width="{w_total - w_exp - 1}" '
                    f'height="26" rx="3" fill="{t["s1"]}"/>\n'
                )
            else:
                body += (
                    f'<rect x="{bx}" y="{y}" width="{w_total}" height="26" rx="3" '
                    f'fill="{t["ink2"]}" fill-opacity="0.45"/>\n'
                )
            body += (
                f'<text x="{bx + w_total + 10}" y="{y + 18}" font-size="13" fill="{t["ink"]}">'
                f"{total:,} B</text>\n"
            )
        ly = gy + 8 + 3 * 74 - 6
        body += (
            swatch(bx, ly, t["s2"])
            + f'<text x="{bx + 16}" y="{ly + 10}" font-size="12" fill="{t["ink2"]}">'
            f"64 exponent bytes, one per 16 values</text>\n"
            + swatch(bx + 260, ly, t["s1"])
            + f'<text x="{bx + 276}" y="{ly + 10}" font-size="12" fill="{t["ink2"]}">'
            f"signs and mantissas</text>\n"
        )
        (OUT / f"blockfloat-{mode}.svg").write_text(
            svg(1000, 360, t, body, "What one 32x32 tile costs in bf16 and the block floats")
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    staircase()
    perplexity()
    tile_trip()
    blockfloat()
    print("figures ->", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
