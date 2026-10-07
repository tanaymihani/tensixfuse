"""Export what the Tenstorrent runs need from GPU-CorruptNet's trained model.

Runs the fine-tuned ResNet-50 once on CPU over the same seen- and unseen-content
test frames GPU-CorruptNet reports on (same seeds, same corruptions), and saves:

- the 2,048 pooled features going into the classifier head, per frame
- the head's weights and bias, and the calibration temperature
- the fp32 PyTorch logits as the reference
- for the C++ BatchNorm + ReLU kernel: the input to layer4.2.bn3 for a few
  frames, channels last, plus that BatchNorm folded into a scale and a shift

It needs GPU-CorruptNet's own environment, since it rebuilds the frames with
that package:

    ~/PycharmProjects/GPU_corruptNet/.venv/bin/python model/corruptnet_export.py \
        --corruptnet ~/PycharmProjects/GPU_corruptNet --out data/corruptnet.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corruptnet", required=True, help="path to the GPU-CorruptNet repo")
    p.add_argument("--checkpoint", default="model_resnet50_20260826-012704.pt")
    p.add_argument("--calibration", default="calibration_resnet50_20260826-012704.json")
    p.add_argument("--out", default="data/corruptnet.npz")
    p.add_argument("--bn-frames", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    repo = Path(args.corruptnet).expanduser()
    sys.path.insert(0, str(repo / "src"))
    from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
    from gpu_corruptnet.data.clean_sources import load_clean_splits
    from gpu_corruptnet.data.corrupt_dataset import CorruptionDataset
    from gpu_corruptnet.metrics import multilabel_metrics
    from gpu_corruptnet.models import load_classifier

    torch.manual_seed(0)
    net, ckpt = load_classifier(str(repo / args.checkpoint))
    net.eval()
    temperature = json.loads((repo / args.calibration).read_text())["temperature"]

    # Same splits and dataset settings as the Colab training run (seed 1337,
    # 224 px, clean fraction 0.3, at most 2 corruptions, no flips).
    splits = load_clean_splits(str(repo / "data"), seed=1337)
    datasets = {
        "seen": CorruptionDataset(splits.seen_test, base_seed=3, img_size=ckpt["img_size"]),
        "unseen": CorruptionDataset(splits.unseen_test, base_seed=4, img_size=ckpt["img_size"]),
    }

    feats: list[torch.Tensor] = []
    net.fc.register_forward_hook(lambda m, inp, out: feats.append(inp[0].detach()))
    bn_in: list[torch.Tensor] = []
    bn = net.layer4[2].bn3
    hook = bn.register_forward_hook(lambda m, inp, out: bn_in.append(inp[0].detach()))

    out: dict[str, np.ndarray] = {}
    report = {}
    for name, ds in datasets.items():
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        feats.clear()
        logits, labels = [], []
        t0 = time.time()
        with torch.no_grad():
            for x, y in loader:
                logits.append(net(x))
                labels.append(y)
                if hook is not None and sum(t.shape[0] for t in bn_in) >= args.bn_frames:
                    hook.remove()
                    hook = None
        f = torch.cat(feats).float()
        z = torch.cat(logits).float()
        y = torch.cat(labels).numpy().astype(np.uint8)
        # The head on its own, in fp32, must reproduce the full model's logits.
        z_head = f @ net.fc.weight.T.float() + net.fc.bias.float()
        assert torch.allclose(z, z_head, atol=1e-4), (z - z_head).abs().max()
        probs = torch.sigmoid(z).numpy()
        m = multilabel_metrics(y, probs, list(ARTIFACT_CLASSES))
        report[name] = {"frames": len(y), "macro_f1": round(m["macro_f1"], 4)}
        print(f"{name}: {len(y)} frames, macro-F1 {m['macro_f1']:.4f} ({time.time() - t0:.0f}s)")
        out[f"features_{name}"] = f.numpy()
        out[f"logits_{name}"] = z.numpy()
        out[f"labels_{name}"] = y

    # BatchNorm (eval) as a per-channel scale and shift: y = x * s + b.
    s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    b = (bn.bias - bn.running_mean * s).detach()
    x = torch.cat(bn_in)[: args.bn_frames]  # N, C, H, W
    out["bn_input_nhwc"] = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]).numpy()
    out["bn_scale"] = s.numpy()
    out["bn_shift"] = b.numpy()

    out["head_weight"] = net.fc.weight.detach().numpy()  # 10 x 2048
    out["head_bias"] = net.fc.bias.detach().numpy()
    out["temperature"] = np.array(temperature)
    out["class_names"] = np.array(ARTIFACT_CLASSES)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    report["bn_rows"] = int(out["bn_input_nhwc"].shape[0])
    report["temperature"] = temperature
    print(json.dumps(report), "->", dest, f"{dest.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
