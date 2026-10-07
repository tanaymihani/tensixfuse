"""Check tensixfuse.blockfloat against tt-metal's own bfloat8_b / bfloat4_b packer.

Converts a few matrices to each format through ttnn (to the simulated device
and back) and compares with the NumPy emulation, bit for bit.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import ttnn

from tensixfuse.blockfloat import quantize


def cases(rng: np.random.Generator) -> dict[str, np.ndarray]:
    normal = rng.standard_normal((256, 512)).astype(np.float32)
    outliers = normal.copy()
    idx = rng.integers(0, outliers.size, 40)
    outliers.flat[idx] *= 50.0  # a few large values, like LLM weights
    tiny = (normal * 1e-30).astype(np.float32)  # near fp32 denormals
    ties = np.round(normal * 64) / 64 + 1 / 128  # lands exactly on rounding ties
    return {"normal": normal, "outliers": outliers, "tiny": tiny, "ties": ties.astype(np.float32)}


def main() -> None:
    rng = np.random.default_rng(0)
    report = {}
    device = ttnn.open_device(device_id=0)
    try:
        for fmt, tt_dtype in (("bfloat8_b", ttnn.bfloat8_b), ("bfloat4_b", ttnn.bfloat4_b)):
            for name, x in cases(rng).items():
                t = ttnn.from_torch(
                    torch.from_numpy(x), dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )
                got = ttnn.to_torch(t).float().numpy()
                want = quantize(x, fmt)
                same = got.view(np.uint32) == want.view(np.uint32)
                # +0.0 and -0.0 are the same value; count them as equal.
                same |= (got == 0) & (want == 0)
                report[f"{fmt}/{name}"] = {
                    "elements": int(x.size),
                    "mismatches": int((~same).sum()),
                    "max_abs_diff": float(np.abs(got - want).max()),
                }
    finally:
        ttnn.close_device(device)
    print(json.dumps(report, indent=1))
    assert all(r["mismatches"] == 0 for r in report.values()), "emulation differs from tt-metal"


if __name__ == "__main__":
    main()
