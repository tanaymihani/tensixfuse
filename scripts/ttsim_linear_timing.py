"""How long one GPT-2-sized fused linear + GELU takes on ttsim, per weight format.

128 tokens x 768 -> 3072 (GPT-2 small's c_fc). Prints PCC against fp32 PyTorch
and wall-clock seconds, to size the GPT-2 study.
"""

from __future__ import annotations

import json
import sys
import time

import torch
import ttnn


def main() -> None:
    m, k, n = (int(v) for v in (sys.argv[1:4] if len(sys.argv) > 3 else (128, 768, 3072)))
    torch.manual_seed(0)
    x = torch.randn(m, k)
    w = torch.randn(k, n) / k**0.5
    b = torch.randn(n) * 0.1
    expected = torch.nn.functional.gelu(x @ w + b)
    device = ttnn.open_device(device_id=0)
    out = {}
    try:
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tb = ttnn.from_torch(
            b.reshape(1, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        for name, dt in (
            ("bf16", ttnn.bfloat16),
            ("bfp8_b", ttnn.bfloat8_b),
            ("bfp4_b", ttnn.bfloat4_b),
        ):
            tw = ttnn.from_torch(w, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            t0 = time.perf_counter()
            y = ttnn.to_torch(ttnn.linear(tx, tw, bias=tb, activation="gelu")).float()
            secs = time.perf_counter() - t0
            pcc = torch.corrcoef(torch.stack([y.flatten(), expected.flatten()]))[0, 1].item()
            out[name] = {"pcc": round(pcc, 6), "seconds": round(secs, 1)}
            print(name, out[name], flush=True)
    finally:
        ttnn.close_device(device)
    print(json.dumps({"shape": [m, k, n], **out}))


if __name__ == "__main__":
    main()
