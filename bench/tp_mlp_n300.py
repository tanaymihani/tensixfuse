"""GPT-2's MLP split across the two chips of a simulated N300, Megatron style.

c_fc is split by columns: each chip holds half of the 3,072 hidden units and
applies GELU to its own half, with no communication. c_proj is split by rows:
each chip ends up with a partial 128 x 768 output, and one all-reduce over the
chips' Ethernet link adds the two partials. Checks the result against the
fp32 PyTorch block and that both chips end up with the same bits.

Needs ttsim's 2-chip build (scripts/setup_ttsim.sh n300) and an N300 cluster
descriptor in TT_METAL_MOCK_CLUSTER_DESC_PATH.

    python bench/tp_mlp_n300.py data/gpt2_mlp.npz --out results/tp_n300.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

from tensixfuse.metrics import pcc


def shard(mesh, dim):
    if hasattr(ttnn, "shard_tensor_to_mesh_mapper"):
        return ttnn.shard_tensor_to_mesh_mapper(mesh, dim=dim)
    return ttnn.ShardTensorToMesh(mesh, dim=dim)


def replicate(mesh):
    if hasattr(ttnn, "replicate_tensor_to_mesh_mapper"):
        return ttnn.replicate_tensor_to_mesh_mapper(mesh)
    return ttnn.ReplicateTensorToMesh(mesh)


def concat(mesh, dim):
    if hasattr(ttnn, "concat_mesh_to_tensor_composer"):
        return ttnn.concat_mesh_to_tensor_composer(mesh, dim=dim)
    return ttnn.ConcatMeshToTensor(mesh, dim=dim)


def all_reduce(t):
    """Sum the per-chip partials. Tries the collective spellings across ttnn versions."""
    attempts = (
        ("all_reduce", lambda: ttnn.all_reduce(t)),
        ("all_reduce(cluster_axis=1)", lambda: ttnn.all_reduce(t, cluster_axis=1)),
        ("reduce_scatter + all_gather", lambda: ttnn.all_gather(ttnn.reduce_scatter(t, 3), 3)),
    )
    errors = []
    for name, fn in attempts:
        try:
            return fn(), name
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            errors.append(f"{name}: {str(e).splitlines()[0][:300]}")
    raise RuntimeError("no all-reduce worked:\n" + "\n".join(errors))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="?", default="data/gpt2_mlp.npz")
    p.add_argument("--blocks", type=int, nargs="+", default=[0, 6, 11])
    p.add_argument("--out", default="results/tp_n300.json")
    args = p.parse_args()
    d = np.load(args.data)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
    results: dict = {"chips": mesh.get_num_devices(), "blocks": []}

    def to_mesh(a, mapper):
        return ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(a, np.float32)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=mapper,
        )

    try:
        for i in args.blocks:
            t0 = time.perf_counter()
            x = to_mesh(d[f"x{i}"][None, None], replicate(mesh))
            w_fc = to_mesh(d[f"w_fc{i}"][None, None], shard(mesh, 3))  # columns
            b_fc = to_mesh(d[f"b_fc{i}"].reshape(1, 1, 1, -1), shard(mesh, 3))
            w_proj = to_mesh(d[f"w_proj{i}"][None, None], shard(mesh, 2))  # rows
            b_proj = to_mesh(d[f"b_proj{i}"].reshape(1, 1, 1, -1), replicate(mesh))

            h = ttnn.linear(x, w_fc, bias=b_fc, activation="gelu")  # 128 x 1536 per chip
            partial = ttnn.linear(h, w_proj)  # 128 x 768 per chip, to be summed
            y, how = all_reduce(partial)
            y = ttnn.add(y, b_proj)

            both = ttnn.to_torch(y, mesh_composer=concat(mesh, 0)).float().numpy()
            y0, y1 = both[0, 0], both[1, 0]
            row = {
                "block": i,
                "pcc_vs_fp32": pcc(y0, d[f"y{i}"]),
                "chips_bit_identical": bool(np.array_equal(y0, y1)),
                "collective": how,
                "all_reduced_kib_per_chip": partial.shape[-2] * partial.shape[-1] * 2 / 1024,
                "seconds": round(time.perf_counter() - t0, 1),
            }
            results["blocks"].append(row)
            print(json.dumps(row), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
