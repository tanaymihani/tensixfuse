"""Export layer4.2.bn2's input for the BatchNorm + ReLU kernel.

corruptnet_export.py exported the input to layer4.2.bn3, but in ResNet-50
bn3 is followed by the residual add, not by ReLU, so BatchNorm + ReLU on it
isn't an op the network runs (and 93% of its outputs come out zero).
bn2 is followed by ReLU. This runs the same 32 seen-content test frames
through the model, saves bn2's input (channels last) and its folded scale and
shift, and checks that bn3's input matches what data-v1 has, so both layers
come from the same frames.

    ~/PycharmProjects/GPU_corruptNet/.venv/bin/python model/corruptnet_bn2_export.py \
        --corruptnet ~/PycharmProjects/GPU_corruptNet --v1 data/corruptnet.npz \
        --out data/corruptnet_bn2.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def folded(bn) -> tuple[np.ndarray, np.ndarray]:
    """BatchNorm (eval) as y = x * s + b."""
    s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    b = (bn.bias - bn.running_mean * s).detach()
    return s.numpy(), b.numpy()


def nhwc(x: torch.Tensor) -> np.ndarray:
    return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]).numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corruptnet", required=True, help="path to the GPU-CorruptNet repo")
    p.add_argument("--checkpoint", default="model_resnet50_20260826-012704.pt")
    p.add_argument("--v1", default="data/corruptnet.npz", help="data-v1, to check the frames")
    p.add_argument("--out", default="data/corruptnet_bn2.npz")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    repo = Path(args.corruptnet).expanduser()
    sys.path.insert(0, str(repo / "src"))
    from gpu_corruptnet.data.clean_sources import load_clean_splits
    from gpu_corruptnet.data.corrupt_dataset import CorruptionDataset
    from gpu_corruptnet.models import load_classifier

    torch.manual_seed(0)
    net, ckpt = load_classifier(str(repo / args.checkpoint))
    net.eval()
    # Same split and settings as corruptnet_export.py's seen-content test set.
    splits = load_clean_splits(str(repo / "data"), seed=1337)
    ds = CorruptionDataset(splits.seen_test, base_seed=3, img_size=ckpt["img_size"])

    block = net.layer4[2]
    captured: dict[str, list[torch.Tensor]] = {"bn2": [], "bn3": []}
    for name, store in captured.items():
        getattr(block, name).register_forward_hook(
            lambda m, inp, out, store=store: store.append(inp[0].detach())
        )
    with torch.no_grad():
        for start in range(0, args.frames, args.batch_size):
            idx = range(start, min(start + args.batch_size, args.frames))
            net(torch.stack([ds[i][0] for i in idx]))

    bn2_in = nhwc(torch.cat(captured["bn2"]))
    bn3_in = nhwc(torch.cat(captured["bn3"]))
    v1 = np.load(args.v1)["bn_input_nhwc"]
    diff = float(np.abs(bn3_in - v1).max())
    print(f"bn3 input vs data-v1: shape {bn3_in.shape} vs {v1.shape}, max abs diff {diff:.3g}")
    if bn3_in.shape != v1.shape or diff > 1e-4 * float(np.abs(v1).max()):
        raise SystemExit("not the same frames as data-v1")

    s, b = folded(block.bn2)
    relu_zero = float((np.maximum(bn2_in * s + b, 0) == 0).mean())
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, bn2_input_nhwc=bn2_in, bn2_scale=s, bn2_shift=b)
    print(
        f"bn2 input {bn2_in.shape}, ReLU zeros {relu_zero:.1%} -> {dest} "
        f"({dest.stat().st_size / 2**20:.1f} MiB)"
    )


if __name__ == "__main__":
    main()
