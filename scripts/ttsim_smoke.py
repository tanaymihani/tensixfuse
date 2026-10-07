"""Smallest end-to-end check on ttsim: one TTNN matmul against PyTorch.

Run after scripts/setup_ttsim.sh. Prints a JSON line with the PCC.
"""

from __future__ import annotations

import json
import os
import time

import torch
import ttnn


def main() -> None:
    torch.manual_seed(0)
    a = torch.randn(64, 128)
    b = torch.randn(128, 96)
    t0 = time.perf_counter()
    device = ttnn.open_device(device_id=0)
    try:
        ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.to_torch(ttnn.matmul(ta, tb)).float()
        arch = str(device.arch())
    finally:
        ttnn.close_device(device)
    expected = a.bfloat16().float() @ b.bfloat16().float()
    pcc = torch.corrcoef(torch.stack([out.flatten(), expected.flatten()]))[0, 1].item()
    print(
        json.dumps(
            {
                "arch": arch,
                "simulator": os.environ.get("TT_METAL_SIMULATOR"),
                "pcc": pcc,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
    )
    assert pcc > 0.99, pcc


if __name__ == "__main__":
    main()
