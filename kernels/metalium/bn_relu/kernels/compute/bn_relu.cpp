// Compute (unpack, math and pack RISC-V cores): y = relu(x * scale + shift).
//
// The FPU multiplies x by the scale row broadcast down the tile, the result
// goes to an L1 circular buffer, the FPU adds the shift row the same way, and
// the SFPU applies ReLU in the destination registers before the tile is packed
// out. Nothing in between touches DRAM.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/relu.h"

void kernel_main() {
    const uint32_t n_rows = get_arg_val<uint32_t>(0);
    const uint32_t n_cols = get_arg_val<uint32_t>(1);

    constexpr auto cb_x = tt::CBIndex::c_0;
    constexpr auto cb_scale = tt::CBIndex::c_1;
    constexpr auto cb_shift = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_tmp = tt::CBIndex::c_24;
    constexpr uint32_t dst = 0;

    compute_kernel_hw_startup(cb_x, cb_scale, cb_out);

    for (uint32_t c = 0; c < n_cols; ++c) {
        cb_wait_front(cb_scale, 1);  // held for the whole column
        cb_wait_front(cb_shift, 1);

        for (uint32_t r = 0; r < n_rows; ++r) {
            // tmp = x * scale
            cb_wait_front(cb_x, 1);
            mul_bcast_rows_init(cb_x, cb_scale);
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_x, cb_scale, 0, 0, dst);
            tile_regs_commit();
            cb_pop_front(cb_x, 1);

            cb_reserve_back(cb_tmp, 1);
            tile_regs_wait();
            pack_tile(dst, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);

            // y = relu(tmp + shift)
            cb_wait_front(cb_tmp, 1);
            add_bcast_rows_init(cb_tmp, cb_shift);
            tile_regs_acquire();
            add_tiles_bcast_rows(cb_tmp, cb_shift, 0, 0, dst);
            relu_tile_init();
            relu_tile(dst);
            tile_regs_commit();
            cb_pop_front(cb_tmp, 1);

            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(dst, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }

        cb_pop_front(cb_scale, 1);
        cb_pop_front(cb_shift, 1);
    }
}
