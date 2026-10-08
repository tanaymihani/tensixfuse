// Fused BatchNorm (inference) + ReLU as a TT-Metalium program:
//
//     y = relu(x * scale + shift)
//
// x is channels-last activations, [rows, channels] (rows = frames * H * W), in
// bf16. scale and shift are per channel, BatchNorm folded at export time:
// scale = gamma / sqrt(var + eps), shift = beta - mean * scale.
//
// Each core gets a slice of the channel tiles. For every column of tiles it
// reads that column's scale and shift once, then streams the column's x tiles
// through compute and back to DRAM. x is read once and y written once; the
// intermediate x * scale never leaves the core's L1.
//
// usage: bn_relu IN.bin SCALE.bin SHIFT.bin OUT.bin ROWS CHANNELS [CORES]
// All .bin files are raw little-endian bf16, row-major.

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tt::tt_metal;

#ifndef KERNEL_DIR
#define KERNEL_DIR "kernels/"
#endif

namespace {

constexpr uint32_t TILE = 32;
constexpr uint32_t FACE = 16;
constexpr uint32_t TILE_ELEMS = TILE * TILE;
constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(uint16_t);

std::vector<uint16_t> read_bf16(const std::string& path, size_t expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint16_t> v(expected);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(expected * sizeof(uint16_t)));
    if (static_cast<size_t>(f.gcount()) != expected * sizeof(uint16_t)) {
        throw std::runtime_error(path + ": expected " + std::to_string(expected) + " bf16 values");
    }
    return v;
}

// Position of element (r, c) of a 32x32 tile in tile order. A tile is four
// 16x16 faces stored one after another (top-left, top-right, bottom-left,
// bottom-right), each face row-major. Row 0 of the tile is therefore the first
// row of face 0 followed by the first row of face 1, which is what the FPU's
// row broadcast reads.
inline size_t index_in_tile(uint32_t r, uint32_t c) {
    const uint32_t face = (r / FACE) * 2 + (c / FACE);
    return face * FACE * FACE + (r % FACE) * FACE + (c % FACE);
}

// Row-major [rows, cols] -> tiles in row-major tile order.
std::vector<uint16_t> tilize(const std::vector<uint16_t>& x, uint32_t rows, uint32_t cols) {
    std::vector<uint16_t> out(x.size());
    const uint32_t tiles_per_row = cols / TILE;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const size_t tile = static_cast<size_t>(r / TILE) * tiles_per_row + c / TILE;
            out[tile * TILE_ELEMS + index_in_tile(r % TILE, c % TILE)] = x[static_cast<size_t>(r) * cols + c];
        }
    }
    return out;
}

std::vector<uint16_t> untilize(const std::vector<uint16_t>& t, uint32_t rows, uint32_t cols) {
    std::vector<uint16_t> out(t.size());
    const uint32_t tiles_per_row = cols / TILE;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const size_t tile = static_cast<size_t>(r / TILE) * tiles_per_row + c / TILE;
            out[static_cast<size_t>(r) * cols + c] = t[tile * TILE_ELEMS + index_in_tile(r % TILE, c % TILE)];
        }
    }
    return out;
}

// One tile per channel tile, with the 32 per-channel values in row 0 and zeros
// below. mul/add_tiles_bcast_rows repeat row 0 down the whole tile.
std::vector<uint16_t> row_tiles(const std::vector<uint16_t>& per_channel, uint32_t cols) {
    std::vector<uint16_t> out(static_cast<size_t>(cols / TILE) * TILE_ELEMS, 0);
    for (uint32_t c = 0; c < cols; ++c) {
        out[static_cast<size_t>(c / TILE) * TILE_ELEMS + index_in_tile(0, c % TILE)] = per_channel[c];
    }
    return out;
}

std::shared_ptr<distributed::MeshBuffer> dram_buffer(distributed::MeshDevice* device, uint32_t n_tiles) {
    distributed::DeviceLocalBufferConfig local{.page_size = TILE_BYTES, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig whole{.size = static_cast<uint64_t>(n_tiles) * TILE_BYTES};
    return distributed::MeshBuffer::create(whole, local, device);
}

void make_cb(Program& program, const CoreRangeSet& cores, tt::CBIndex index, uint32_t n_tiles) {
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(n_tiles * TILE_BYTES, {{index, tt::DataFormat::Float16_b}})
            .set_page_size(index, TILE_BYTES));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s IN.bin SCALE.bin SHIFT.bin OUT.bin ROWS CHANNELS [CORES]\n", argv[0]);
        return 2;
    }
    const uint32_t rows = std::stoul(argv[5]);
    const uint32_t cols = std::stoul(argv[6]);
    const uint32_t requested_cores = argc > 7 ? std::stoul(argv[7]) : 8;
    if (rows % TILE || cols % TILE) {
        std::fprintf(stderr, "ROWS and CHANNELS must be multiples of 32 (pad first)\n");
        return 2;
    }
    const uint32_t row_tiles_n = rows / TILE;
    const uint32_t col_tiles_n = cols / TILE;

    const auto x = read_bf16(argv[1], static_cast<size_t>(rows) * cols);
    const auto scale = read_bf16(argv[2], cols);
    const auto shift = read_bf16(argv[3], cols);

    auto device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = device->mesh_command_queue();

    auto x_buf = dram_buffer(device.get(), row_tiles_n * col_tiles_n);
    auto s_buf = dram_buffer(device.get(), col_tiles_n);
    auto b_buf = dram_buffer(device.get(), col_tiles_n);
    auto y_buf = dram_buffer(device.get(), row_tiles_n * col_tiles_n);

    auto x_tiles = tilize(x, rows, cols);
    auto s_tiles = row_tiles(scale, cols);
    auto b_tiles = row_tiles(shift, cols);
    distributed::EnqueueWriteMeshBuffer(cq, x_buf, x_tiles, false);
    distributed::EnqueueWriteMeshBuffer(cq, s_buf, s_tiles, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_tiles, false);

    // Spread the channel tiles over a row of cores.
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t n_cores = std::max<uint32_t>(1, std::min({requested_cores, col_tiles_n, grid.x * grid.y}));
    std::vector<CoreCoord> core_list;
    for (uint32_t i = 0; i < n_cores; ++i) core_list.push_back({i % grid.x, i / grid.x});
    std::vector<CoreRange> ranges;
    for (const auto& c : core_list) ranges.emplace_back(c, c);
    const CoreRangeSet cores(ranges);

    Program program = CreateProgram();
    make_cb(program, cores, tt::CBIndex::c_0, 2);   // x
    make_cb(program, cores, tt::CBIndex::c_1, 2);   // scale rows, next column prefetched
    make_cb(program, cores, tt::CBIndex::c_2, 2);   // shift rows
    make_cb(program, cores, tt::CBIndex::c_24, 2);  // x * scale, never leaves L1
    make_cb(program, cores, tt::CBIndex::c_16, 2);  // y

    std::vector<uint32_t> reader_ct;
    TensorAccessorArgs(*x_buf).append_to(reader_ct);
    TensorAccessorArgs(*s_buf).append_to(reader_ct);
    TensorAccessorArgs(*b_buf).append_to(reader_ct);
    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs(*y_buf).append_to(writer_ct);

    auto reader = CreateKernel(
        program,
        KERNEL_DIR "dataflow/reader_bn.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct});
    auto writer = CreateKernel(
        program,
        KERNEL_DIR "dataflow/writer_bn.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_ct});
    auto compute = CreateKernel(
        program, KERNEL_DIR "compute/bn_relu.cpp", cores, ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

    const uint32_t per_core = col_tiles_n / n_cores;
    const uint32_t extra = col_tiles_n % n_cores;
    uint32_t col_start = 0;
    for (uint32_t i = 0; i < n_cores; ++i) {
        const uint32_t n_cols = per_core + (i < extra ? 1 : 0);
        const CoreCoord& core = core_list[i];
        SetRuntimeArgs(
            program,
            reader,
            core,
            {x_buf->address(), s_buf->address(), b_buf->address(), row_tiles_n, col_tiles_n, col_start, n_cols});
        SetRuntimeArgs(program, compute, core, {row_tiles_n, n_cols});
        SetRuntimeArgs(program, writer, core, {y_buf->address(), row_tiles_n, col_tiles_n, col_start, n_cols});
        col_start += n_cols;
    }

    const auto t0 = std::chrono::steady_clock::now();
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::vector<uint16_t> y_tiles;
    distributed::EnqueueReadMeshBuffer(cq, y_tiles, y_buf, true);
    const auto y = untilize(y_tiles, rows, cols);
    std::ofstream(argv[4], std::ios::binary)
        .write(reinterpret_cast<const char*>(y.data()), static_cast<std::streamsize>(y.size() * sizeof(uint16_t)));

    device->close();
    std::printf(
        "{\"rows\": %u, \"channels\": %u, \"tiles\": %u, \"cores\": %u, \"seconds\": %.2f}\n",
        rows,
        cols,
        row_tiles_n * col_tiles_n,
        n_cores,
        secs);
    return 0;
}
