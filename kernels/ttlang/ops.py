"""TT-Lang operations for the fusion study.

Every operation has the usual Tensix split: a reader that brings tiles in over
the NoC, a compute kernel, and a writer that sends results back. Shapes are in
elements, blocks in 32x32 tiles, and the grid is (columns, rows).

make_matmul(..., epilogue="bias_act") is the fused kernel: the bias add and the
activation run on the accumulator before it leaves L1. The unfused baseline is
make_matmul(..., epilogue=None) followed by make_eltwise(op="add") and
make_eltwise(op="act"), with the same blocking, so fusion is the only difference.

The kernels are written so the TT-Lang compiler takes them as well as the
functional simulator: a kernel may only capture ints, floats, tensors, dataflow
buffers and pipes, and can only call TT-Lang's own functions. So each variant
(fused or not, which activation) gets its own kernel body, picked in Python
inside the operation, instead of a flag or a function passed into one body.
"""

from __future__ import annotations

import ttl

ACTIVATIONS = ("relu", "gelu", "sigmoid", "none")


def _check_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}")


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


# Compute kernels shared by both matmuls. Each output block accumulates over K
# in an L1 buffer, then either goes out as is or gets c added and the
# activation applied first. Blocks past the edge of the output are skipped.


def _matmul_compute_plain(a_dfb, b_dfb, acc_dfb, y_dfb, Mb, Nb, Kb, m_per, n_per):
    @ttl.compute()
    def compute():
        col, row = ttl.node(dims=2)
        for lm in range(m_per):
            mb = row * m_per + lm
            if mb < Mb:
                for ln in range(n_per):
                    nb = col * n_per + ln
                    if nb < Nb:
                        with acc_dfb.reserve() as acc_blk:
                            acc_blk.store(ttl.block.fill(0, shape=acc_blk.shape))
                        for _ in range(Kb):
                            with (
                                a_dfb.wait() as a_blk,
                                b_dfb.wait() as b_blk,
                                acc_dfb.wait() as pre,
                            ):
                                with acc_dfb.reserve() as acc_blk:
                                    acc_blk.store(pre + a_blk @ b_blk)
                        with acc_dfb.wait() as acc_blk:
                            with y_dfb.reserve() as y_blk:
                                y_blk.store(acc_blk)

    return compute


def _matmul_compute_fused(act, a_dfb, b_dfb, acc_dfb, c_dfb, y_dfb, Mb, Nb, Kb, m_per, n_per):
    if act == "relu":

        @ttl.compute()
        def compute():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                mb = row * m_per + lm
                if mb < Mb:
                    for ln in range(n_per):
                        nb = col * n_per + ln
                        if nb < Nb:
                            with acc_dfb.reserve() as acc_blk:
                                acc_blk.store(ttl.block.fill(0, shape=acc_blk.shape))
                            for _ in range(Kb):
                                with (
                                    a_dfb.wait() as a_blk,
                                    b_dfb.wait() as b_blk,
                                    acc_dfb.wait() as pre,
                                ):
                                    with acc_dfb.reserve() as acc_blk:
                                        acc_blk.store(pre + a_blk @ b_blk)
                            with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
                                with y_dfb.reserve() as y_blk:
                                    y_blk.store(ttl.math.relu(c_blk + acc_blk))

    elif act == "gelu":

        @ttl.compute()
        def compute():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                mb = row * m_per + lm
                if mb < Mb:
                    for ln in range(n_per):
                        nb = col * n_per + ln
                        if nb < Nb:
                            with acc_dfb.reserve() as acc_blk:
                                acc_blk.store(ttl.block.fill(0, shape=acc_blk.shape))
                            for _ in range(Kb):
                                with (
                                    a_dfb.wait() as a_blk,
                                    b_dfb.wait() as b_blk,
                                    acc_dfb.wait() as pre,
                                ):
                                    with acc_dfb.reserve() as acc_blk:
                                        acc_blk.store(pre + a_blk @ b_blk)
                            with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
                                with y_dfb.reserve() as y_blk:
                                    y_blk.store(ttl.math.gelu(c_blk + acc_blk))

    elif act == "sigmoid":

        @ttl.compute()
        def compute():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                mb = row * m_per + lm
                if mb < Mb:
                    for ln in range(n_per):
                        nb = col * n_per + ln
                        if nb < Nb:
                            with acc_dfb.reserve() as acc_blk:
                                acc_blk.store(ttl.block.fill(0, shape=acc_blk.shape))
                            for _ in range(Kb):
                                with (
                                    a_dfb.wait() as a_blk,
                                    b_dfb.wait() as b_blk,
                                    acc_dfb.wait() as pre,
                                ):
                                    with acc_dfb.reserve() as acc_blk:
                                        acc_blk.store(pre + a_blk @ b_blk)
                            with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
                                with y_dfb.reserve() as y_blk:
                                    y_blk.store(ttl.math.sigmoid(c_blk + acc_blk))

    else:  # "none": just the bias add

        @ttl.compute()
        def compute():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                mb = row * m_per + lm
                if mb < Mb:
                    for ln in range(n_per):
                        nb = col * n_per + ln
                        if nb < Nb:
                            with acc_dfb.reserve() as acc_blk:
                                acc_blk.store(ttl.block.fill(0, shape=acc_blk.shape))
                            for _ in range(Kb):
                                with (
                                    a_dfb.wait() as a_blk,
                                    b_dfb.wait() as b_blk,
                                    acc_dfb.wait() as pre,
                                ):
                                    with acc_dfb.reserve() as acc_blk:
                                        acc_blk.store(pre + a_blk @ b_blk)
                            with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
                                with y_dfb.reserve() as y_blk:
                                    y_blk.store(c_blk + acc_blk)

    return compute


def _matmul_compute_even(act, a_dfb, b_dfb, acc_dfb, c_dfb, y_dfb, Kb, m_per, n_per):
    """Compute for the multicast matmul, where the blocks divide evenly over the
    grid. No bounds checks: with pipes in the operation, the compiler has to
    prove which nodes produce and consume each buffer, and guards built from
    node coordinates are more than it can analyze. act=None is the plain matmul.
    """
    if act is None:

        @ttl.compute()
        def compute():
            for _ in range(m_per):
                for _ in range(n_per):
                    with acc_dfb.reserve() as acc_blk:
                        acc_blk.store(ttl.block.fill(0, shape=acc_blk.shape))
                    for _ in range(Kb):
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            acc_dfb.wait() as pre,
                        ):
                            with acc_dfb.reserve() as acc_blk:
                                acc_blk.store(pre + a_blk @ b_blk)
                    with acc_dfb.wait() as acc_blk:
                        with y_dfb.reserve() as y_blk:
                            y_blk.store(acc_blk)

    elif act == "relu":

        @ttl.compute()
        def compute():
            for _ in range(m_per):
                for _ in range(n_per):
                    with acc_dfb.reserve() as acc_blk:
                        acc_blk.store(ttl.block.fill(0, shape=acc_blk.shape))
                    for _ in range(Kb):
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            acc_dfb.wait() as pre,
                        ):
                            with acc_dfb.reserve() as acc_blk:
                                acc_blk.store(pre + a_blk @ b_blk)
                    with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
                        with y_dfb.reserve() as y_blk:
                            y_blk.store(ttl.math.relu(c_blk + acc_blk))

    elif act == "gelu":

        @ttl.compute()
        def compute():
            for _ in range(m_per):
                for _ in range(n_per):
                    with acc_dfb.reserve() as acc_blk:
                        acc_blk.store(ttl.block.fill(0, shape=acc_blk.shape))
                    for _ in range(Kb):
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            acc_dfb.wait() as pre,
                        ):
                            with acc_dfb.reserve() as acc_blk:
                                acc_blk.store(pre + a_blk @ b_blk)
                    with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
                        with y_dfb.reserve() as y_blk:
                            y_blk.store(ttl.math.gelu(c_blk + acc_blk))

    elif act == "sigmoid":

        @ttl.compute()
        def compute():
            for _ in range(m_per):
                for _ in range(n_per):
                    with acc_dfb.reserve() as acc_blk:
                        acc_blk.store(ttl.block.fill(0, shape=acc_blk.shape))
                    for _ in range(Kb):
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            acc_dfb.wait() as pre,
                        ):
                            with acc_dfb.reserve() as acc_blk:
                                acc_blk.store(pre + a_blk @ b_blk)
                    with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
                        with y_dfb.reserve() as y_blk:
                            y_blk.store(ttl.math.sigmoid(c_blk + acc_blk))

    else:  # "none": just the bias add

        @ttl.compute()
        def compute():
            for _ in range(m_per):
                for _ in range(n_per):
                    with acc_dfb.reserve() as acc_blk:
                        acc_blk.store(ttl.block.fill(0, shape=acc_blk.shape))
                    for _ in range(Kb):
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            acc_dfb.wait() as pre,
                        ):
                            with acc_dfb.reserve() as acc_blk:
                                acc_blk.store(pre + a_blk @ b_blk)
                    with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
                        with y_dfb.reserve() as y_blk:
                            y_blk.store(c_blk + acc_blk)

    return compute


def make_matmul(cfg, epilogue=None, act="relu"):
    """Blocked, output-stationary matmul over a core grid.

    epilogue=None writes a @ b. epilogue="bias_act" writes act(a @ b + c), where
    c has the same shape as the output.
    """
    _check_activation(act)
    bm, bn, bk = cfg.bm, cfg.bn, cfg.bk
    Mb, Nb, Kb = cfg.blocks
    cols, rows = cfg.grid
    m_per = _cdiv(Mb, rows)
    n_per = _cdiv(Nb, cols)
    fused = epilogue == "bias_act"

    if cfg.mcast:
        return _make_matmul_mcast(cfg, fused, act)

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
                                with c_dfb.reserve() as c_blk:
                                    ttl.copy(
                                        c[mb * bm : (mb + 1) * bm, nb * bn : (nb + 1) * bn], c_blk
                                    ).wait()
                                for kb in range(Kb):
                                    with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                                        tx_a = ttl.copy(
                                            a[mb * bm : (mb + 1) * bm, kb * bk : (kb + 1) * bk],
                                            a_blk,
                                        )
                                        tx_b = ttl.copy(
                                            b[kb * bk : (kb + 1) * bk, nb * bn : (nb + 1) * bn],
                                            b_blk,
                                        )
                                        tx_a.wait()
                                        tx_b.wait()

            _matmul_compute_fused(
                act, a_dfb, b_dfb, acc_dfb, c_dfb, y_dfb, Mb, Nb, Kb, m_per, n_per
            )
        else:

            @ttl.datamovement()
            def read():
                col, row = ttl.node(dims=2)
                for lm in range(m_per):
                    mb = row * m_per + lm
                    if mb < Mb:
                        for ln in range(n_per):
                            nb = col * n_per + ln
                            if nb < Nb:
                                for kb in range(Kb):
                                    with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                                        tx_a = ttl.copy(
                                            a[mb * bm : (mb + 1) * bm, kb * bk : (kb + 1) * bk],
                                            a_blk,
                                        )
                                        tx_b = ttl.copy(
                                            b[kb * bk : (kb + 1) * bk, nb * bn : (nb + 1) * bn],
                                            b_blk,
                                        )
                                        tx_a.wait()
                                        tx_b.wait()

            _matmul_compute_plain(a_dfb, b_dfb, acc_dfb, y_dfb, Mb, Nb, Kb, m_per, n_per)

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


def _make_matmul_mcast(cfg, fused, act):
    """SUMMA-style matmul: A multicast along grid rows, B down grid columns.

    The core in column 0 of each row reads A from DRAM, keeps it, and sends it
    to the other cores in its row; the core in row 0 of each column does the
    same for B. Each input block leaves DRAM once per time it's needed by a
    whole row or column, instead of once per core. The source isn't one of its
    own pipe's destinations: receiving into the block it just read would make
    it wait on itself, which the compiler rejects.
    """
    bm, bn, bk = cfg.bm, cfg.bn, cfg.bk
    Mb, Nb, Kb = cfg.blocks
    cols, rows = cfg.grid
    if cols < 2 or rows < 2:
        raise ValueError("multicast needs at least 2 columns and 2 rows")
    m_per = Mb // rows
    n_per = Nb // cols

    @ttl.operation(grid=(cols, rows))
    def matmul_mcast(a, b, c, y):
        a_net = ttl.PipeNet([ttl.Pipe(src=(0, r), dst=(slice(1, cols), r)) for r in range(rows)])
        b_net = ttl.PipeNet([ttl.Pipe(src=(cc, 0), dst=(cc, slice(1, rows))) for cc in range(cols)])
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(bk, bn), block_count=2)
        acc_dfb = ttl.make_dataflow_buffer_like(y, shape=(bm, bn), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(bm, bn), block_count=2)

        # Reader 1: C for this core's own block, plus A over the row pipe.
        if fused:
            c_dfb = ttl.make_dataflow_buffer_like(c, shape=(bm, bn), block_count=2)

            @ttl.datamovement()
            def read_a():
                col, row = ttl.node(dims=2)
                for lm in range(m_per):
                    mb = row * m_per + lm
                    for ln in range(n_per):
                        nb = col * n_per + ln
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

            _matmul_compute_even(act, a_dfb, b_dfb, acc_dfb, c_dfb, y_dfb, Kb, m_per, n_per)
        else:

            @ttl.datamovement()
            def read_a():
                _col, row = ttl.node(dims=2)
                for lm in range(m_per):
                    mb = row * m_per + lm
                    for ln in range(n_per):
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

            _matmul_compute_even(None, a_dfb, b_dfb, acc_dfb, None, y_dfb, Kb, m_per, n_per)

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


def _eltwise_compute(op, x_dfb, c_dfb, o_dfb, Mb, Nb, m_per, n_per):
    if op == "add":

        @ttl.compute()
        def compute():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                if row * m_per + lm < Mb:
                    for ln in range(n_per):
                        if col * n_per + ln < Nb:
                            with x_dfb.wait() as x_blk, c_dfb.wait() as c_blk:
                                with o_dfb.reserve() as o_blk:
                                    o_blk.store(x_blk + c_blk)

    elif op == "relu":

        @ttl.compute()
        def compute():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                if row * m_per + lm < Mb:
                    for ln in range(n_per):
                        if col * n_per + ln < Nb:
                            with x_dfb.wait() as x_blk:
                                with o_dfb.reserve() as o_blk:
                                    o_blk.store(ttl.math.relu(x_blk))

    elif op == "gelu":

        @ttl.compute()
        def compute():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                if row * m_per + lm < Mb:
                    for ln in range(n_per):
                        if col * n_per + ln < Nb:
                            with x_dfb.wait() as x_blk:
                                with o_dfb.reserve() as o_blk:
                                    o_blk.store(ttl.math.gelu(x_blk))

    elif op == "sigmoid":

        @ttl.compute()
        def compute():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                if row * m_per + lm < Mb:
                    for ln in range(n_per):
                        if col * n_per + ln < Nb:
                            with x_dfb.wait() as x_blk:
                                with o_dfb.reserve() as o_blk:
                                    o_blk.store(ttl.math.sigmoid(x_blk))

    else:  # "none": a copy

        @ttl.compute()
        def compute():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                if row * m_per + lm < Mb:
                    for ln in range(n_per):
                        if col * n_per + ln < Nb:
                            with x_dfb.wait() as x_blk:
                                with o_dfb.reserve() as o_blk:
                                    o_blk.store(x_blk)

    return compute


def make_eltwise(cfg, op="add", act="relu"):
    """Elementwise op over the output in (bm, bn) blocks: x + c, or act(x)."""
    _check_activation(act)
    bm, bn = cfg.bm, cfg.bn
    Mb, Nb, _ = cfg.blocks
    cols, rows = cfg.grid
    m_per = _cdiv(Mb, rows)
    n_per = _cdiv(Nb, cols)
    binary = op == "add"

    @ttl.operation(grid=(cols, rows))
    def eltwise(x, c, out):
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(bm, bn), block_count=2)
        o_dfb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        if binary:
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
                                with x_dfb.reserve() as x_blk:
                                    ttl.copy(
                                        x[mb * bm : (mb + 1) * bm, nb * bn : (nb + 1) * bn], x_blk
                                    ).wait()
                                with c_dfb.reserve() as c_blk:
                                    ttl.copy(
                                        c[mb * bm : (mb + 1) * bm, nb * bn : (nb + 1) * bn], c_blk
                                    ).wait()

            _eltwise_compute("add", x_dfb, c_dfb, o_dfb, Mb, Nb, m_per, n_per)
        else:

            @ttl.datamovement()
            def read():
                col, row = ttl.node(dims=2)
                for lm in range(m_per):
                    mb = row * m_per + lm
                    if mb < Mb:
                        for ln in range(n_per):
                            nb = col * n_per + ln
                            if nb < Nb:
                                with x_dfb.reserve() as x_blk:
                                    ttl.copy(
                                        x[mb * bm : (mb + 1) * bm, nb * bn : (nb + 1) * bn], x_blk
                                    ).wait()

            _eltwise_compute(act, x_dfb, None, o_dfb, Mb, Nb, m_per, n_per)

        @ttl.datamovement()
        def write():
            col, row = ttl.node(dims=2)
            for lm in range(m_per):
                mb = row * m_per + lm
                if mb < Mb:
                    for ln in range(n_per):
                        nb = col * n_per + ln
                        if nb < Nb:
                            with o_dfb.wait() as o_blk:
                                ttl.copy(
                                    o_blk, out[mb * bm : (mb + 1) * bm, nb * bn : (nb + 1) * bn]
                                ).wait()

    return eltwise
