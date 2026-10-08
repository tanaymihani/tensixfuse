"""GPU-CorruptNet's classifier head on ttsim, as one fused sigmoid(x @ W + b).

The head maps 2,048 pooled ResNet-50 features to 10 artifact logits. Features
come from data/corruptnet.npz (1,560 seen + 2,600 unseen test frames, which is
exactly 130 tile rows together). The calibration temperature is folded into W
and b, and the 10 outputs are padded to one 32-wide tile.

    python bench/corruptnet_head.py --data data/corruptnet.npz --out results/corruptnet_head.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import ttnn

from tensixfuse.metrics import macro_f1, pcc

FORMATS = {"bf16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "bfloat4_b": ttnn.bfloat4_b}


def to_dev(a: np.ndarray, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(a, np.float32)),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/corruptnet.npz")
    p.add_argument("--out", default="results/corruptnet_head.json")
    args = p.parse_args()
    d = np.load(args.data)

    n_seen = len(d["features_seen"])
    x = np.concatenate([d["features_seen"], d["features_unseen"]])
    labels = np.concatenate([d["labels_seen"], d["labels_unseen"]]).astype(bool)
    t = float(d["temperature"])
    ref_logits = np.concatenate([d["logits_seen"], d["logits_unseen"]]) / t
    ref_pred = ref_logits >= 0  # sigmoid(z) >= 0.5

    n_cls = d["head_weight"].shape[0]
    w = np.zeros((x.shape[1], 32), np.float32)
    w[:, :n_cls] = d["head_weight"].T / t
    b = np.zeros((1, 32), np.float32)
    b[0, :n_cls] = d["head_bias"] / t

    def f1s(pred: np.ndarray) -> dict:
        return {
            "seen": macro_f1(labels[:n_seen], pred[:n_seen]),
            "unseen": macro_f1(labels[n_seen:], pred[n_seen:]),
        }

    results = {
        "frames": len(x),
        "tile_rows": len(x) / 32,
        "temperature": t,
        "pytorch": {"macro_f1": f1s(ref_pred)},
        "formats": {},
    }
    device = ttnn.open_device(device_id=0)
    try:
        results["arch"] = str(device.arch())
        tx = to_dev(x, device)
        tb = to_dev(b, device)
        for name, dtype in FORMATS.items():
            tw = to_dev(w, device, dtype)
            try:
                probs_t = ttnn.linear(tx, tw, bias=tb, activation="sigmoid")
                fused = True
            except (RuntimeError, TypeError, ValueError):  # no fused sigmoid: run it on its own
                probs_t = ttnn.sigmoid(ttnn.linear(tx, tw, bias=tb))
                fused = False
            probs = ttnn.to_torch(probs_t).float().numpy()
            logits = ttnn.to_torch(ttnn.linear(tx, tw, bias=tb)).float().numpy()

            pred = probs[:, :n_cls] >= 0.5
            padded = probs[:, n_cls:]
            results["formats"][name] = {
                "fused_sigmoid": fused,
                "label_set_agreement": float((pred == ref_pred).all(axis=1).mean()),
                "frames_that_differ": int((pred != ref_pred).any(axis=1).sum()),
                "logit_pcc": pcc(logits[:, :n_cls], ref_logits),
                "macro_f1": f1s(pred),
                # The padding gotcha: what thresholding before slicing would do.
                "padded_column_values": sorted({float(v) for v in np.unique(padded)})[:5],
                "labels_per_frame_if_padding_kept": float((probs >= 0.5).sum(axis=1).mean()),
                "labels_per_frame": float(pred.sum(axis=1).mean()),
            }
            print(name, json.dumps(results["formats"][name]), flush=True)
    finally:
        ttnn.close_device(device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
