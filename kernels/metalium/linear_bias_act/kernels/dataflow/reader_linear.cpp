// Reader for y = act(x @ w + b): for each output tile in this core's columns,
// the x and w tiles along K, then that column's bias tile.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t b_addr = get_arg_val<uint32_t>(2);
    const uint32_t Mt = get_arg_val<uint32_t>(3);
    const uint32_t Kt = get_arg_val<uint32_t>(4);
    const uint32_t Nt = get_arg_val<uint32_t>(5);
    const uint32_t n_start = get_arg_val<uint32_t>(6);
    const uint32_t n_count = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_w = 1;
    constexpr uint32_t cb_b = 2;

    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto xs = TensorAccessor(x_args, x_addr);
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto ws = TensorAccessor(w_args, w_addr);
    constexpr auto b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto bs = TensorAccessor(b_args, b_addr);

    for (uint32_t nt = n_start; nt < n_start + n_count; ++nt) {
        for (uint32_t mt = 0; mt < Mt; ++mt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_reserve_back(cb_x, 1);
                cb_reserve_back(cb_w, 1);
                noc_async_read_page(mt * Kt + kt, xs, get_write_ptr(cb_x));
                noc_async_read_page(kt * Nt + nt, ws, get_write_ptr(cb_w));
                noc_async_read_barrier();
                cb_push_back(cb_x, 1);
                cb_push_back(cb_w, 1);
            }
            cb_reserve_back(cb_b, 1);
            noc_async_read_page(nt, bs, get_write_ptr(cb_b));
            noc_async_read_barrier();
            cb_push_back(cb_b, 1);
        }
    }
}
