"""Run one fused or unfused y = act(a @ b + c) under TT-Lang and check it.

Meant to be launched by bench/fusion.py through the simulator, e.g.

    tt-lang-sim kernels/ttlang/run_matmul.py --trace t.jsonl --trace-events copy,pipe \
        -- --M 1024 --K 1024 --N 1024 --block 4,4,4 --variant fused --out r.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simfix  # noqa: F401  (readable kernel errors under tt-lang-sim 1.1.6)
from ops import make_eltwise, make_matmul

from tensixfuse.traffic import MatmulConfig

TORCH_ACT = {
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
    "sigmoid": torch.sigmoid,
    "none": lambda x: x,
}


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=256)
    p.add_argument("--K", type=int, default=256)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--block", default="4,4,4", help="bm,bn,bk in tiles")
    p.add_argument("--grid", default="1,1", help="columns,rows")
    p.add_argument("--mcast", action="store_true")
    p.add_argument("--variant", choices=("fused", "unfused"), default="fused")
    p.add_argument("--act", default="relu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-check", action="store_true", help="skip the PyTorch comparison")
    p.add_argument("--out", default=None)
    if argv and argv[0] == "--":  # tt-lang-sim passes its separator through
        argv = argv[1:]
    return p.parse_args(argv)


def pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    return torch.corrcoef(torch.stack([x.flatten().float(), y.flatten().float()]))[0, 1].item()


def to_device(t: torch.Tensor, device, name: str):
    tt = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    try:
        tt._name = name  # labels the tensor in tt-lang-sim-stats
    except AttributeError:
        pass
    return tt


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    bm, bn, bk = (int(v) for v in args.block.split(","))
    cols, rows = (int(v) for v in args.grid.split(","))
    cfg = MatmulConfig(args.M, args.K, args.N, bm, bn, bk, grid=(cols, rows), mcast=args.mcast)

    torch.manual_seed(args.seed)
    a = (torch.randn(args.M, args.K) / math.sqrt(args.K)).to(torch.bfloat16)
    b = torch.randn(args.K, args.N).to(torch.bfloat16)
    c = torch.randn(args.M, args.N).to(torch.bfloat16)
    expected = TORCH_ACT[args.act](a.float() @ b.float() + c.float())

    device = ttnn.open_device(device_id=0)
    try:
        a_t, b_t, c_t = (
            to_device(a, device, "a"),
            to_device(b, device, "b"),
            to_device(c, device, "c"),
        )
        zeros = torch.zeros(args.M, args.N, dtype=torch.bfloat16)
        if args.variant == "fused":
            y_t = to_device(zeros, device, "y")
            make_matmul(cfg, epilogue="bias_act", act=args.act)(a_t, b_t, c_t, y_t)
        else:
            y1_t = to_device(zeros, device, "y1")
            y2_t = to_device(zeros, device, "y2")
            y_t = to_device(zeros, device, "y")
            make_matmul(cfg)(a_t, b_t, c_t, y1_t)
            make_eltwise(cfg, op="add")(y1_t, c_t, y2_t)
            make_eltwise(cfg, op="act", act=args.act)(y2_t, c_t, y_t)
        y = ttnn.to_torch(y_t)
    finally:
        ttnn.close_device(device)

    result = {
        "M": args.M,
        "K": args.K,
        "N": args.N,
        "block": [bm, bn, bk],
        "grid": [cols, rows],
        "mcast": args.mcast,
        "variant": args.variant,
        "act": args.act,
    }
    if not args.no_check:
        result["pcc"] = pcc(y, expected)
        result["max_abs_err"] = (y.float() - expected).abs().max().item()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f)
    print(json.dumps(result))


main()
