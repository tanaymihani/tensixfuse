"""ttnn.linear's activation, with and without core_grid: a standalone repro.

Without core_grid, ttnn's matmul.cpp applies `activation` as a separate op after
the matmul, on the output already rounded to bf16; with core_grid the activation
goes into the matmul kernel and runs on the fp32 sum. With fp32 accumulation the
two give different bits. bench/linear_bias_act.py is the full comparison; this is
the short version for an upstream report. Runs on a card or on ttsim
(.github/workflows/repro.yml).

    python bench/repro_linear_activation.py
"""

import torch
import ttnn

dev = ttnn.open_device(device_id=0)
cfg = ttnn.init_device_compute_kernel_config(
    dev.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)
torch.manual_seed(0)
x = torch.randn(128, 768).bfloat16()
w = (torch.randn(768, 3072) * 0.02).bfloat16()
b = (torch.randn(3072) * 0.02).bfloat16()
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
tb = ttnn.from_torch(b.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

default = ttnn.to_torch(ttnn.linear(tx, tw, bias=tb, activation="gelu", compute_kernel_config=cfg))
separate = ttnn.to_torch(ttnn.gelu(ttnn.linear(tx, tw, bias=tb, compute_kernel_config=cfg)))
grid = ttnn.to_torch(
    ttnn.linear(
        tx,
        tw,
        bias=tb,
        activation="gelu",
        compute_kernel_config=cfg,
        core_grid=ttnn.CoreGrid(y=4, x=8),
    )
)
ttnn.close_device(dev)
print("default == ttnn.gelu(ttnn.linear(...)):", (default == separate).float().mean().item())
print("default == with core_grid:             ", (default == grid).float().mean().item())

# fp32 math on the same rounded inputs (w as the chip stores it), GELU in its exact form.
wq = ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
ref = torch.nn.functional.gelu(x.double() @ wq.double() + b.double()).float().bfloat16()
print("equal to fp32 math rounded once, default:  ", (default == ref).float().mean().item())
print("equal to fp32 math rounded once, core_grid:", (grid == ref).float().mean().item())
