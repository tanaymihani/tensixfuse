// Compute, fp32 variant: y = relu(x * scale + shift) on the SFPU (vector engine).
//
// The FPU version (bn_relu.cpp) is faster on silicon but its operands go through
// 16- or 19-bit source registers, and x * scale is rounded to bf16 in L1 before
// the add. Here all three steps happen in fp32 destination registers
// (fp32_dest_acc_en), so the only rounding is the final pack to bf16. scale
// and shift arrive as full tiles (each row a copy of the channel values),
// because the SFPU binary ops work tile by tile with no row broadcast.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t n_rows = get_arg_val<uint32_t>(0);
    const uint32_t n_cols = get_arg_val<uint32_t>(1);

    constexpr auto cb_x = tt::CBIndex::c_0;
    constexpr auto cb_scale = tt::CBIndex::c_1;
    constexpr auto cb_shift = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_x, cb_out);

    for (uint32_t c = 0; c < n_cols; ++c) {
        cb_wait_front(cb_scale, 1);
        cb_wait_front(cb_shift, 1);

        for (uint32_t r = 0; r < n_rows; ++r) {
            cb_wait_front(cb_x, 1);
            tile_regs_acquire();
            copy_init(cb_x);
            copy_tile(cb_x, 0, 0);
            copy_init(cb_scale);
            copy_tile(cb_scale, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);  // dst0 = x * scale, fp32
            copy_init(cb_shift);
            copy_tile(cb_shift, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);  // dst0 += shift, fp32
            relu_tile_init();
            relu_tile(0);
            tile_regs_commit();
            cb_pop_front(cb_x, 1);

            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(0, cb_out);  // the one rounding, fp32 -> bf16
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }

        cb_pop_front(cb_scale, 1);
        cb_pop_front(cb_shift, 1);
    }
}
