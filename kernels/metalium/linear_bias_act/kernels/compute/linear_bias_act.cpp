// Compute for y = act(x @ w + b), in the order ttnn.linear's fused-bias kernel
// uses (bmm_large_block_zm_fused_bias_activation.cpp at v0.79.0), so the two
// can match bit for bit:
//
// 1. x @ w over all K tiles, accumulated in fp32 destination registers
//    (fp32_dest_acc_en), packed to an fp32 buffer in L1.
// 2. That buffer + b on the matrix engine, b broadcast down from row 0.
// 3. The activation on the vector engine (GELU in its accurate form, which is
//    what ttnn.linear's activation="gelu" selects), packed to bf16.
//
// Compile-time arg 0: 0 none, 1 relu, 2 gelu.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);  // output tiles on this core
    const uint32_t Kt = get_arg_val<uint32_t>(1);
    constexpr uint32_t act = get_compile_time_arg_val(0);

    constexpr auto cb_x = tt::CBIndex::c_0;
    constexpr auto cb_w = tt::CBIndex::c_1;
    constexpr auto cb_b = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_mm = tt::CBIndex::c_24;  // x @ w in fp32
    constexpr uint32_t dst = 0;

    // Like ttnn's matmul: x feeds srcB and w srcA.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_mm);

    for (uint32_t t = 0; t < n_tiles; ++t) {
        // 1. x @ w
        matmul_init(cb_x, cb_w);
        pack_reconfig_data_format(cb_mm);
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_wait_front(cb_x, 1);
            cb_wait_front(cb_w, 1);
            matmul_tiles(cb_x, cb_w, 0, 0, dst);
            cb_pop_front(cb_x, 1);
            cb_pop_front(cb_w, 1);
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
        cb_wait_front(cb_b, 1);
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
        cb_pop_front(cb_b, 1);

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(dst, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);

        reconfig_data_format(cb_mm, cb_w, cb_b, cb_x);
    }
}
