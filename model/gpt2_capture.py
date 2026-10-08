"""Capture GPT-2 small's 12 MLP blocks on real text: input, output and weights.

Runs the model on CPU over the first 128 tokens of the WikiText-2 test set and
saves, for every block, what goes into its MLP (after ln_2), what comes out,
and the MLP's weights. bench/quant_gpt2.py replays these on ttsim.

    python model/gpt2_capture.py --out data/gpt2_mlp.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--out", default="data/gpt2_mlp.npz")
    args = p.parse_args()

    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "".join(test["text"][:200])
    ids = tok(text, return_tensors="pt").input_ids[:, : args.tokens]
    assert ids.shape[1] == args.tokens

    captured: dict[str, np.ndarray] = {}

    def hook(i):
        def fn(module, inputs, output):
            captured[f"x{i}"] = inputs[0][0].detach().float().numpy()
            captured[f"y{i}"] = output[0].detach().float().numpy()

        return fn

    handles = [b.mlp.register_forward_hook(hook(i)) for i, b in enumerate(model.transformer.h)]
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()

    out = dict(captured)
    for i, block in enumerate(model.transformer.h):
        # GPT-2's Conv1D stores weights as [in, out], the layout a matmul wants.
        out[f"w_fc{i}"] = block.mlp.c_fc.weight.detach().numpy()
        out[f"b_fc{i}"] = block.mlp.c_fc.bias.detach().numpy()
        out[f"w_proj{i}"] = block.mlp.c_proj.weight.detach().numpy()
        out[f"b_proj{i}"] = block.mlp.c_proj.bias.detach().numpy()
    out["token_ids"] = ids[0].numpy()

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print(f"{len(model.transformer.h)} blocks, {args.tokens} tokens -> {dest}")


if __name__ == "__main__":
    main()
