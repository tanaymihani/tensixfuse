// Compute for y = act(x @ w + b), in the order ttnn.linear's fused-bias kernel
// uses (bmm_large_block_zm_fused_bias_activation.cpp, the same in v0.79.0 and
// v0.81.1), so the two can match bit for bit:
//
// 1. x @ w over all K tiles, accumulated in fp32 destination registers
//    (fp32_dest_acc_en), packed to an fp32 buffer in L1.
// 2. That buffer + b on the matrix engine, b broadcast down from row 0.
// 3. The activation on the vector engine (GELU in its accurate form, which is
//    what ttnn.linear's activation="gelu" selects), packed to bf16.
//
// All of x stays in L1 for the whole call, and each column of w (with its bias
// tile) is read once and used for every row of the output; each output tile
// still sums its K tiles in order, so the bits are the same as one tile at a time.
//
// Compile-time arg 0: 0 none, 1 relu, 2 gelu, 3 gelu on x @ w + b rounded to
// bf16 first (the sum packed to bf16, read back, then GELU), to test whether
// that's what ttnn.linear does with activation="gelu".

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    const uint32_t n_cols = get_arg_val<uint32_t>(0);  // output tile columns on this core
    const uint32_t Kt = get_arg_val<uint32_t>(1);
    const uint32_t Mt = get_arg_val<uint32_t>(2);
    constexpr uint32_t act = get_compile_time_arg_val(0);

    constexpr auto cb_x = tt::CBIndex::c_0;
    constexpr auto cb_w = tt::CBIndex::c_1;
    constexpr auto cb_b = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_mm = tt::CBIndex::c_24;     // x @ w in fp32
    constexpr auto cb_sum16 = tt::CBIndex::c_25;  // x @ w + b in bf16 (act 3 only)
    constexpr uint32_t dst = 0;

    // Like ttnn's matmul: x feeds srcB and w srcA.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_mm);

    // All of x stays in L1; each column of w and its bias tile serve all Mt rows.
    cb_wait_front(cb_x, Mt * Kt);
    for (uint32_t c = 0; c < n_cols; ++c) {
        cb_wait_front(cb_w, Kt);
        cb_wait_front(cb_b, 1);
        for (uint32_t mt = 0; mt < Mt; ++mt) {
            // 1. x @ w, the K tiles in the same order as before, so the sum is the same
            matmul_init(cb_x, cb_w);
            pack_reconfig_data_format(cb_mm);
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                matmul_tiles(cb_x, cb_w, mt * Kt + kt, kt, dst);
            }
            tile_regs_commit();
            cb_reserve_back(cb_mm, 1);
            tile_regs_wait();
            pack_tile(dst, cb_mm);
            tile_regs_release();
            cb_push_back(cb_mm, 1);

            // 2. + b, 3. the activation
            reconfig_data_format(cb_w, cb_mm, cb_x, cb_b);
            pack_reconfig_data_format(cb_out);
            add_bcast_rows_init(cb_mm, cb_b);
            cb_wait_front(cb_mm, 1);
            tile_regs_acquire();
            add_tiles_bcast_rows(cb_mm, cb_b, 0, 0, dst);
            if constexpr (act == 1) {
                relu_tile_init();
                relu_tile(dst);
            } else if constexpr (act == 2) {
                gelu_tile_init<false>();
                gelu_tile<false>(dst);
            }
            tile_regs_commit();
            cb_pop_front(cb_mm, 1);

            if constexpr (act == 3) {
                // The sum goes out to a bf16 buffer first...
                cb_reserve_back(cb_sum16, 1);
                tile_regs_wait();
                pack_tile(dst, cb_sum16);
                tile_regs_release();
                cb_push_back(cb_sum16, 1);
                // ...and comes back for GELU.
                reconfig_data_format_srca(cb_mm, cb_sum16);
                copy_init(cb_sum16);
                cb_wait_front(cb_sum16, 1);
                tile_regs_acquire();
                copy_tile(cb_sum16, 0, dst);
                gelu_tile_init<false>();
                gelu_tile<false>(dst);
                tile_regs_commit();
                cb_pop_front(cb_sum16, 1);
                reconfig_data_format_srca(cb_sum16, cb_mm);
            }
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(dst, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);

            reconfig_data_format(cb_mm, cb_w, cb_b, cb_x);
        }
        cb_pop_front(cb_w, Kt);
        cb_pop_front(cb_b, 1);
    }
    cb_pop_front(cb_x, Mt * Kt);
}
