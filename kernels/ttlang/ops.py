"""TT-Lang operations for the fusion study.

Every operation has the usual Tensix split: a reader that brings tiles in over
the NoC, a compute kernel, and a writer that sends results back. Shapes are in
elements, blocks in 32x32 tiles, and the grid is (columns, rows).

make_matmul(..., epilogue="bias_act") is the fused kernel: the bias add and the
activation run on the accumulator before it leaves L1. The unfused baseline is
make_matmul(..., epilogue=None) followed by make_eltwise_add and
make_eltwise_act, with the same blocking, so fusion is the only difference.
"""

from __future__ import annotations

import ttl

ACTIVATIONS = ("relu", "gelu", "sigmoid", "none")


def _activation(name):
    if name == "none":
        return lambda x: x
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}")
    return getattr(ttl.math, name)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def make_matmul(cfg, epilogue=None, act="relu"):
    """Blocked, output-stationary matmul over a core grid.

    epilogue=None writes a @ b. epilogue="bias_act" writes act(a @ b + c), where
    c has the same shape as the output.
    """
    bm, bn, bk = cfg.bm, cfg.bn, cfg.bk
    Mb, Nb, Kb = cfg.blocks
    cols, rows = cfg.grid
    m_per = _cdiv(Mb, rows)
    n_per = _cdiv(Nb, cols)
    fused = epilogue == "bias_act"
    act_fn = _activation(act)

    if cfg.mcast:
        return _make_matmul_mcast(cfg, fused, act_fn)

    @ttl.operation(grid=(cols, rows))
    def matmul(a, b, c, y):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(bk, bn), block_count=2)
        acc_dfb = ttl.make_dataflow_buffer_like(y, shape=(bm, bn), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(bm, bn), block_count=2)
        if fused:
            c_dfb = ttl.make_dataflow_buffer_like(c, shape=(bm, bn), block_count=2)

        @ttl.datamovement()
        def read():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                mb = row * m_per + lm
                if mb < Mb:
                    for ln in range(n_per):
                        nb = col * n_per + ln
                        if nb < Nb:
                            if fused:
                                with c_dfb.reserve() as c_blk:
                                    ttl.copy(
                                        c[mb * bm : (mb + 1) * bm, nb * bn : (nb + 1) * bn], c_blk
                                    ).wait()
                            for kb in range(Kb):
                                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                                    tx_a = ttl.copy(
                                        a[mb * bm : (mb + 1) * bm, kb * bk : (kb + 1) * bk], a_blk
                                    )
                                    tx_b = ttl.copy(
                                        b[kb * bk : (kb + 1) * bk, nb * bn : (nb + 1) * bn], b_blk
                                    )
                                    tx_a.wait()
                                    tx_b.wait()

        @ttl.compute()
        def compute():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                mb = row * m_per + lm
                if mb < Mb:
                    for ln in range(n_per):
                        nb = col * n_per + ln
                        if nb < Nb:
                            _accumulate(a_dfb, b_dfb, acc_dfb, Kb)
                            _epilogue(acc_dfb, y_dfb, c_dfb if fused else None, act_fn)

        @ttl.datamovement()
        def write():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                mb = row * m_per + lm
                if mb < Mb:
                    for ln in range(n_per):
                        nb = col * n_per + ln
                        if nb < Nb:
                            with y_dfb.wait() as y_blk:
                                ttl.copy(
                                    y_blk, y[mb * bm : (mb + 1) * bm, nb * bn : (nb + 1) * bn]
                                ).wait()

    return matmul


def _make_matmul_mcast(cfg, fused, act_fn):
    """SUMMA-style matmul: A multicast along grid rows, B down grid columns.

    The core in column 0 of each row reads A from DRAM and sends it to every
    core in its row; the core in row 0 of each column does the same for B. Each
    input block leaves DRAM once per time it's needed by a whole row or column,
    instead of once per core.
    """
    bm, bn, bk = cfg.bm, cfg.bn, cfg.bk
    Mb, Nb, Kb = cfg.blocks
    cols, rows = cfg.grid
    m_per = Mb // rows
    n_per = Nb // cols

    @ttl.operation(grid=(cols, rows))
    def matmul_mcast(a, b, c, y):
        a_net = ttl.PipeNet([ttl.Pipe(src=(0, r), dst=(slice(0, cols), r)) for r in range(rows)])
        b_net = ttl.PipeNet([ttl.Pipe(src=(cc, 0), dst=(cc, slice(0, rows))) for cc in range(cols)])
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(bk, bn), block_count=2)
        acc_dfb = ttl.make_dataflow_buffer_like(y, shape=(bm, bn), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(bm, bn), block_count=2)
        if fused:
            c_dfb = ttl.make_dataflow_buffer_like(c, shape=(bm, bn), block_count=2)

        # Reader 1: C for this core's own block, plus A over the row pipe.
        @ttl.datamovement()
        def read_a():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                mb = row * m_per + lm
                for ln in range(n_per):
                    nb = col * n_per + ln
                    if fused:
                        with c_dfb.reserve() as c_blk:
                            ttl.copy(
                                c[mb * bm : (mb + 1) * bm, nb * bn : (nb + 1) * bn], c_blk
                            ).wait()
                    for kb in range(Kb):
                        with a_dfb.reserve() as a_blk:

                            def send_a(pipe):
                                ttl.copy(
                                    a[mb * bm : (mb + 1) * bm, kb * bk : (kb + 1) * bk], a_blk
                                ).wait()
                                ttl.copy(a_blk, pipe).wait()

                            def recv_a(pipe):
                                ttl.copy(pipe, a_blk).wait()

                            a_net.if_src(send_a)
                            a_net.if_dst(recv_a)

        @ttl.compute()
        def compute():
            for _ in range(m_per):
                for _ in range(n_per):
                    _accumulate(a_dfb, b_dfb, acc_dfb, Kb)
                    _epilogue(acc_dfb, y_dfb, c_dfb if fused else None, act_fn)

        # Reader 2: B over the column pipe, then the finished output block.
        @ttl.datamovement()
        def read_b_write_y():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                mb = row * m_per + lm
                for ln in range(n_per):
                    nb = col * n_per + ln
                    for kb in range(Kb):
                        with b_dfb.reserve() as b_blk:

                            def send_b(pipe):
                                ttl.copy(
                                    b[kb * bk : (kb + 1) * bk, nb * bn : (nb + 1) * bn], b_blk
                                ).wait()
                                ttl.copy(b_blk, pipe).wait()

                            def recv_b(pipe):
                                ttl.copy(pipe, b_blk).wait()

                            b_net.if_src(send_b)
                            b_net.if_dst(recv_b)
                    with y_dfb.wait() as y_blk:
                        ttl.copy(y_blk, y[mb * bm : (mb + 1) * bm, nb * bn : (nb + 1) * bn]).wait()

    return matmul_mcast


def _accumulate(a_dfb, b_dfb, acc_dfb, Kb):
    """acc = sum over K blocks of a_blk @ b_blk, kept in an L1 buffer."""
    with acc_dfb.reserve() as acc_blk:
        acc_blk.store(ttl.block.fill(0, shape=acc_blk.shape))
    for _ in range(Kb):
        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk, acc_dfb.wait() as pre:
            with acc_dfb.reserve() as acc_blk:
                acc_blk.store(pre + a_blk @ b_blk)


def _epilogue(acc_dfb, y_dfb, c_dfb, act_fn):
    """Fused: y = act(c + acc). Unfused: y = acc."""
    if c_dfb is not None:
        with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
            with y_dfb.reserve() as y_blk:
                y_blk.store(act_fn(c_blk + acc_blk))
    else:
        with acc_dfb.wait() as acc_blk:
            with y_dfb.reserve() as y_blk:
                y_blk.store(acc_blk)


def make_eltwise(cfg, op="add", act="relu"):
    """Elementwise op over the output in (bm, bn) blocks: x + c, or act(x)."""
    bm, bn = cfg.bm, cfg.bn
    Mb, Nb, _ = cfg.blocks
    cols, rows = cfg.grid
    m_per = _cdiv(Mb, rows)
    n_per = _cdiv(Nb, cols)
    binary = op == "add"
    act_fn = _activation(act)

    def blocks():
        col, row = ttl.node(dims=2)
        for lm in range(m_per):
            mb = row * m_per + lm
            if mb < Mb:
                for ln in range(n_per):
                    nb = col * n_per + ln
                    if nb < Nb:
                        yield mb, nb

    @ttl.operation(grid=(cols, rows))
    def eltwise(x, c, out):
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(bm, bn), block_count=2)
        o_dfb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)
        if binary:
            c_dfb = ttl.make_dataflow_buffer_like(c, shape=(bm, bn), block_count=2)

        @ttl.datamovement()
        def read():
            for mb, nb in blocks():
                rs, cs = slice(mb * bm, (mb + 1) * bm), slice(nb * bn, (nb + 1) * bn)
                with x_dfb.reserve() as x_blk:
                    ttl.copy(x[rs, cs], x_blk).wait()
                if binary:
                    with c_dfb.reserve() as c_blk:
                        ttl.copy(c[rs, cs], c_blk).wait()

        @ttl.compute()
        def compute():
            for _ in blocks():
                if binary:
                    with x_dfb.wait() as x_blk, c_dfb.wait() as c_blk:
                        with o_dfb.reserve() as o_blk:
                            o_blk.store(x_blk + c_blk)
                else:
                    with x_dfb.wait() as x_blk:
                        with o_dfb.reserve() as o_blk:
                            o_blk.store(act_fn(x_blk))

        @ttl.datamovement()
        def write():
            for mb, nb in blocks():
                with o_dfb.wait() as o_blk:
                    ttl.copy(o_blk, out[mb * bm : (mb + 1) * bm, nb * bn : (nb + 1) * bn]).wait()

    return eltwise
