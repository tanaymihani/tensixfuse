// Reader (RISC-V 0): DRAM -> L1 over the NoC.
//
// For each channel tile this core owns, pushes that column's scale and shift
// rows once, then every x tile in the column. Compute holds the scale and
// shift for the whole column, so they cost two tile reads per column instead
// of two per x tile.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t scale_addr = get_arg_val<uint32_t>(1);
    const uint32_t shift_addr = get_arg_val<uint32_t>(2);
    const uint32_t n_rows = get_arg_val<uint32_t>(3);      // tile rows in x
    const uint32_t row_stride = get_arg_val<uint32_t>(4);  // tiles per tile row of x
    const uint32_t col_start = get_arg_val<uint32_t>(5);
    const uint32_t n_cols = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_scale = tt::CBIndex::c_1;
    constexpr uint32_t cb_shift = tt::CBIndex::c_2;
    const uint32_t tile_bytes = get_tile_size(cb_x);

    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, x_addr);
    constexpr auto scale_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto scale = TensorAccessor(scale_args, scale_addr);
    constexpr auto shift_args = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();
    const auto shift = TensorAccessor(shift_args, shift_addr);

    Noc noc;
    CircularBuffer x_cb(cb_x);
    CircularBuffer scale_cb(cb_scale);
    CircularBuffer shift_cb(cb_shift);

    for (uint32_t c = col_start; c < col_start + n_cols; ++c) {
        scale_cb.reserve_back(1);
        shift_cb.reserve_back(1);
        noc.async_read(scale, scale_cb, tile_bytes, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read(shift, shift_cb, tile_bytes, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read_barrier();  // data must land in L1 before compute is told it's there
        scale_cb.push_back(1);
        shift_cb.push_back(1);

        for (uint32_t r = 0; r < n_rows; ++r) {
            x_cb.reserve_back(1);
            noc.async_read(x, x_cb, tile_bytes, {.page_id = r * row_stride + c}, {.offset_bytes = 0});
            noc.async_read_barrier();
            x_cb.push_back(1);
        }
    }
}
