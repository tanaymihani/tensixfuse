// Writer for y = act(x @ w + b): output tiles back to DRAM in the order compute
// makes them, column by column.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t Mt = get_arg_val<uint32_t>(1);
    const uint32_t Nt = get_arg_val<uint32_t>(2);
    const uint32_t n_start = get_arg_val<uint32_t>(3);
    const uint32_t n_count = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out = 16;
    constexpr auto y_args = TensorAccessorArgs<0>();
    const auto ys = TensorAccessor(y_args, y_addr);

    for (uint32_t nt = n_start; nt < n_start + n_count; ++nt) {
        for (uint32_t mt = 0; mt < Mt; ++mt) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(mt * Nt + nt, ys, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
