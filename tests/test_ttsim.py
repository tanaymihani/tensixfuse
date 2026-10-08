"""Checks that run on ttsim, the bit-exact Wormhole/Blackhole simulator.

Need Linux x86_64 with the ttnn wheel, SFPI and ttsim set up
(scripts/install_sfpi.sh, scripts/setup_ttsim.sh). Skipped anywhere else.
TENSIXFUSE_FORMATS limits the weight formats, so CI can split them into jobs.
"""

import json
import os

import numpy as np
import pytest

from tensixfuse.blockfloat import quantize
from tensixfuse.metrics import pcc
from tensixfuse.simrun import REPO

pytestmark = pytest.mark.ttsim

if not os.environ.get("TT_METAL_SIMULATOR"):
    pytest.skip("TT_METAL_SIMULATOR is not set", allow_module_level=True)
ttnn = pytest.importorskip("ttnn")
torch = pytest.importorskip("torch")

FORMATS = os.environ.get("TENSIXFUSE_FORMATS", "bfloat16,bfloat8_b,bfloat4_b").split(",")
FLOORS = json.loads((REPO / "bench" / "baselines.json").read_text())["pcc_floor"]


def tt_dtype(name):
    return getattr(ttnn, name)


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def to_dev(a, device, dtype="bfloat16"):
    return ttnn.from_torch(
        torch.as_tensor(np.ascontiguousarray(a, np.float32)),
        dtype=tt_dtype(dtype),
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def test_matmul_matches_torch(device):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 128)).astype(np.float32)
    b = rng.standard_normal((128, 96)).astype(np.float32)
    out = ttnn.to_torch(ttnn.matmul(to_dev(a, device), to_dev(b, device))).float().numpy()
    assert pcc(out, a @ b) > FLOORS["matmul"]


@pytest.mark.parametrize("fmt", [f for f in FORMATS if f != "bfloat16"])
def test_block_float_emulation_matches_tt_metal(device, fmt):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((128, 256)).astype(np.float32)
    x.flat[rng.integers(0, x.size, 20)] *= 40.0  # outliers, the hard case
    got = ttnn.to_torch(to_dev(x, device, fmt)).float().numpy()
    want = quantize(x, fmt)
    same = (got.view(np.uint32) == want.view(np.uint32)) | ((got == 0) & (want == 0))
    assert same.all(), f"{(~same).sum()} of {same.size} values differ"


@pytest.mark.parametrize("fmt", FORMATS)
def test_gpt2_sized_linear_gelu_stays_above_floor(device, fmt):
    # GPT-2 small's c_fc: 128 tokens, 768 -> 3072, with GELU fused into the matmul.
    rng = np.random.default_rng(2)
    x = rng.standard_normal((128, 768)).astype(np.float32)
    w = (rng.standard_normal((768, 3072)) / np.sqrt(768)).astype(np.float32)
    b = (rng.standard_normal((1, 3072)) * 0.1).astype(np.float32)
    y = ttnn.linear(
        to_dev(x, device), to_dev(w, device, fmt), bias=to_dev(b, device), activation="gelu"
    )
    expected = torch.nn.functional.gelu(torch.from_numpy(x @ w + b)).numpy()
    assert pcc(ttnn.to_torch(y).float().numpy(), expected) > FLOORS["linear_gelu"][fmt]


def test_padded_head_columns_come_out_as_exactly_one_half(device):
    # Zero-padded weight columns give sigmoid(0) = 0.5, which a >= 0.5 threshold
    # counts as a prediction. The head has to be sliced before thresholding.
    rng = np.random.default_rng(3)
    x = rng.standard_normal((32, 64)).astype(np.float32)
    w = np.zeros((64, 32), np.float32)
    w[:, :10] = rng.standard_normal((64, 10))
    probs = ttnn.to_torch(ttnn.sigmoid(ttnn.linear(to_dev(x, device), to_dev(w, device))))
    assert torch.all(probs[:, 10:].float() == 0.5)
