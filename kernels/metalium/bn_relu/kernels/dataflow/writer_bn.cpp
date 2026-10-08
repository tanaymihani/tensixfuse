// Writer (RISC-V 1): finished y tiles from L1 back to DRAM over the NoC, in
// the same column-by-column order the reader used.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_rows = get_arg_val<uint32_t>(1);
    const uint32_t row_stride = get_arg_val<uint32_t>(2);
    const uint32_t col_start = get_arg_val<uint32_t>(3);
    const uint32_t n_cols = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr auto y_args = TensorAccessorArgs<0>();
    const auto y = TensorAccessor(y_args, y_addr);

    Noc noc;
    CircularBuffer out_cb(cb_out);
    for (uint32_t c = col_start; c < col_start + n_cols; ++c) {
        for (uint32_t r = 0; r < n_rows; ++r) {
            out_cb.wait_front(1);
            noc.async_write(out_cb, y, tile_bytes, {}, {.page_id = r * row_stride + c});
            noc.async_write_barrier();
            out_cb.pop_front(1);
        }
    }
}
