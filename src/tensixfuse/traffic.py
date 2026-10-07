"""Analytic DRAM traffic model for the TT-Lang matmul kernels in kernels/ttlang.

Counts 32x32 tiles moved between DRAM and L1, which is what tt-lang-sim-stats
reports, so the simulator's numbers can be checked against this tile for tile.

The matmul is output-stationary, like TT-Lang's matmul tutorial: every block of
bm x bn output tiles streams its row panel of A and column panel of B from DRAM
in steps of bk tiles. Without multicast, every output block fetches its own
panels, so A is read once per block column and B once per block row. With
multicast (SUMMA style), one core per grid row reads each A block and sends it
along the row over the NoC, and one core per grid column does the same for B.
"""

from __future__ import annotations

from dataclasses import dataclass

TILE = 32

# Bytes per 32x32 tile. Block-float formats store one 8-bit exponent per 16
# values (64 per tile) next to one byte (bfp8_b) or half a byte (bfp4_b) per value.
TILE_BYTES = {
    "float32": 4096,
    "bfloat16": 2048,
    "bfloat8_b": 1024 + 64,
    "bfloat4_b": 512 + 64,
}


@dataclass(frozen=True)
class MatmulConfig:
    """Shape (in elements), block (in tiles) and core grid of one matmul."""

    M: int
    K: int
    N: int
    bm: int
    bn: int
    bk: int
    grid: tuple[int, int] = (1, 1)  # (columns, rows), as ttl.node(dims=2) returns
    mcast: bool = False

    def __post_init__(self) -> None:
        for name, dim in (("M", self.M), ("K", self.K), ("N", self.N)):
            if dim % TILE:
                raise ValueError(f"{name}={dim} is not a multiple of {TILE}")
        Mt, Kt, Nt = self.tiles
        if Mt % self.bm or Nt % self.bn or Kt % self.bk:
            raise ValueError(
                f"block {(self.bm, self.bn, self.bk)} must divide {(Mt, Nt, Kt)} tiles"
            )
        cols, rows = self.grid
        if self.mcast and (self.blocks[0] % rows or self.blocks[1] % cols):
            raise ValueError("multicast needs the output blocks to split evenly over the grid")

    @property
    def tiles(self) -> tuple[int, int, int]:
        return self.M // TILE, self.K // TILE, self.N // TILE

    @property
    def blocks(self) -> tuple[int, int, int]:
        Mt, Kt, Nt = self.tiles
        return Mt // self.bm, Nt // self.bn, Kt // self.bk

    @property
    def cores(self) -> int:
        return self.grid[0] * self.grid[1]


def matmul_input_reads(cfg: MatmulConfig) -> dict[str, int]:
    """DRAM tiles read for A and B by the blocked matmul."""
    Mt, Kt, Nt = cfg.tiles
    Mb, Nb, _ = cfg.blocks
    if cfg.mcast:
        cols, rows = cfg.grid
        # The row source re-sends A for each of its node's block columns, and
        # the column source re-sends B for each block row.
        return {"a": Mt * Kt * (Nb // cols), "b": Kt * Nt * (Mb // rows)}
    return {"a": Mt * Kt * Nb, "b": Kt * Nt * Mb}


def fused_traffic(cfg: MatmulConfig) -> dict[str, int]:
    """y = act(a @ b + c) in one operation: c read once, y written once."""
    Mt, _, Nt = cfg.tiles
    out = Mt * Nt
    reads = matmul_input_reads(cfg)
    t = {"a": reads["a"], "b": reads["b"], "c": out, "y": out}
    t["total"] = sum(t.values())
    return t


def unfused_traffic(cfg: MatmulConfig) -> dict[str, int]:
    """y1 = a @ b; y2 = y1 + c; y = relu(y2), each op going through DRAM.

    y1 and y2 are each written once and read back once.
    """
    Mt, _, Nt = cfg.tiles
    out = Mt * Nt
    reads = matmul_input_reads(cfg)
    t = {"a": reads["a"], "b": reads["b"], "y1": 2 * out, "c": out, "y2": 2 * out, "y": out}
    t["total"] = sum(t.values())
    return t


def fusion_saving_tiles(cfg: MatmulConfig) -> int:
    """Tiles fusion removes: the two intermediates, each written and read back."""
    return unfused_traffic(cfg)["total"] - fused_traffic(cfg)["total"]


def compulsory_tiles(cfg: MatmulConfig) -> int:
    """Floor for the fused op: every input tile read once, every output tile written once."""
    Mt, Kt, Nt = cfg.tiles
    return Mt * Kt + Kt * Nt + 2 * Mt * Nt


def fused_l1_bytes(cfg: MatmulConfig, dtype: str = "bfloat16", block_count: int = 2) -> int:
    """L1 taken by the fused kernel's dataflow buffers on one core.

    A (bm x bk), B (bk x bn), and bm x bn buffers for C, the accumulator and Y,
    each holding block_count blocks.
    """
    tiles = cfg.bm * cfg.bk + cfg.bk * cfg.bn + 3 * cfg.bm * cfg.bn
    return block_count * tiles * TILE_BYTES[dtype]


def tiles_to_mib(tiles: int, dtype: str = "bfloat16") -> float:
    return tiles * TILE_BYTES[dtype] / 2**20
