// A linear layer with the bias and the activation fused in, as a TT-Metalium
// program:
//
//     y = act(x @ w + b)
//
// x is [M, K] activations in bf16, w is [K, N] weights in bfloat8_b (or bf16),
// b is [N] in bf16. It follows the steps of ttnn.linear's own kernel for a
// fused bias, so the two can be compared bit for bit: the K tiles accumulate in
// fp32 destination registers, the sum goes to an fp32 buffer in L1, the matrix
// engine adds b as a row broadcast, the activation runs on the vector engine,
// and the result is packed to bf16.
//
// Each core gets a range of output tile columns. For every output tile it
// streams the x and w tiles along K and the bias tile for that column.
//
// usage: linear_bias_act X.bin W.bin B.bin OUT.bin M K N [none|relu|gelu|gelu_after_round] [bfp8|bf16] [CORES]
// gelu_after_round rounds x @ w + b to bf16 before GELU.
// X.bin, B.bin and OUT.bin are raw little-endian bf16, row-major. W.bin is raw
// float32, row-major; the host packs it as bfloat8_b with tt-metal's packer.

#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/span.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
constexpr uint32_t BF16_TILE_BYTES = TILE_ELEMS * sizeof(uint16_t);
constexpr uint32_t FP32_TILE_BYTES = TILE_ELEMS * sizeof(float);
constexpr uint32_t BFP8_TILE_BYTES = TILE_ELEMS + TILE_ELEMS / FACE;  // mantissas + one exponent per 16

template <typename T>
std::vector<T> read_raw(const std::string& path, size_t expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<T> v(expected);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(expected * sizeof(T)));
    if (static_cast<size_t>(f.gcount()) != expected * sizeof(T)) {
        throw std::runtime_error(path + ": expected " + std::to_string(expected) + " values");
    }
    return v;
}

// Position of element (r, c) of a 32x32 tile in tile order: four 16x16 faces
// one after another (top-left, top-right, bottom-left, bottom-right), each
// row-major.
inline size_t index_in_tile(uint32_t r, uint32_t c) {
    const uint32_t face = (r / FACE) * 2 + (c / FACE);
    return face * FACE * FACE + (r % FACE) * FACE + (c % FACE);
}

// Row-major [rows, cols] -> tiles in row-major tile order, faces inside.
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

// Row-major [rows, cols] -> tiles in row-major tile order, each tile itself
// row-major: the input pack_as_bfp8_tiles(..., row_major_input=true) expects.
std::vector<float> tiles_row_major(const std::vector<float>& x, uint32_t rows, uint32_t cols) {
    std::vector<float> out(x.size());
    const uint32_t tiles_per_row = cols / TILE;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const size_t tile = static_cast<size_t>(r / TILE) * tiles_per_row + c / TILE;
            out[tile * TILE_ELEMS + (r % TILE) * TILE + (c % TILE)] = x[static_cast<size_t>(r) * cols + c];
        }
    }
    return out;
}

// bf16 (round to nearest even) of a float, as raw bits.
uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7FFF + ((u >> 16) & 1);
    return static_cast<uint16_t>(u >> 16);
}

// One tile per output tile column, with the column's 32 bias values in row 0:
// add_tiles_bcast_rows repeats row 0 down the tile.
std::vector<uint16_t> bias_tiles(const std::vector<uint16_t>& b, uint32_t cols) {
    std::vector<uint16_t> out(static_cast<size_t>(cols / TILE) * TILE_ELEMS, 0);
    for (uint32_t c = 0; c < cols; ++c) {
        out[static_cast<size_t>(c / TILE) * TILE_ELEMS + index_in_tile(0, c % TILE)] = b[c];
    }
    return out;
}

std::shared_ptr<distributed::MeshBuffer> dram_buffer(
    distributed::MeshDevice* device, uint32_t n_tiles, uint32_t tile_bytes) {
    distributed::DeviceLocalBufferConfig local{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig whole{.size = static_cast<uint64_t>(n_tiles) * tile_bytes};
    return distributed::MeshBuffer::create(whole, local, device);
}

void make_cb(
    Program& program,
    const CoreRangeSet& cores,
    tt::CBIndex index,
    uint32_t n_tiles,
    tt::DataFormat format,
    uint32_t tile_bytes) {
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(n_tiles * tile_bytes, {{index, format}}).set_page_size(index, tile_bytes));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(
            stderr,
            "usage: %s X.bin W.bin B.bin OUT.bin M K N [none|relu|gelu|gelu_after_round] [bfp8|bf16] [CORES]\n",
            argv[0]);
        return 2;
    }
    const uint32_t M = std::stoul(argv[5]);
    const uint32_t K = std::stoul(argv[6]);
    const uint32_t N = std::stoul(argv[7]);
    const std::string act = argc > 8 ? argv[8] : "gelu";
    const std::string fmt = argc > 9 ? argv[9] : "bfp8";
    const uint32_t requested_cores = argc > 10 ? std::stoul(argv[10]) : 8;
    uint32_t act_id;
    if (act == "none") {
        act_id = 0;
    } else if (act == "relu") {
        act_id = 1;
    } else if (act == "gelu") {
        act_id = 2;
    } else if (act == "gelu_after_round") {
        act_id = 3;
    } else {
        std::fprintf(stderr, "activation must be none, relu, gelu or gelu_after_round\n");
        return 2;
    }
    if (fmt != "bfp8" && fmt != "bf16") {
        std::fprintf(stderr, "weight format must be bfp8 or bf16\n");
        return 2;
    }
    if (M % TILE || K % TILE || N % TILE) {
        std::fprintf(stderr, "M, K and N must be multiples of 32 (pad first)\n");
        return 2;
    }
    const bool bfp8 = fmt == "bfp8";
    const uint32_t Mt = M / TILE;
    const uint32_t Kt = K / TILE;
    const uint32_t Nt = N / TILE;

    const auto x = read_raw<uint16_t>(argv[1], static_cast<size_t>(M) * K);
    const auto w = read_raw<float>(argv[2], static_cast<size_t>(K) * N);
    const auto b = read_raw<uint16_t>(argv[3], N);

    auto device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = device->mesh_command_queue();

    const uint32_t w_tile_bytes = bfp8 ? BFP8_TILE_BYTES : BF16_TILE_BYTES;
    auto x_buf = dram_buffer(device.get(), Mt * Kt, BF16_TILE_BYTES);
    auto w_buf = dram_buffer(device.get(), Kt * Nt, w_tile_bytes);
    auto b_buf = dram_buffer(device.get(), Nt, BF16_TILE_BYTES);
    auto y_buf = dram_buffer(device.get(), Mt * Nt, BF16_TILE_BYTES);

    // The writes don't block, so the host copies stay alive until Finish below.
    auto x_tiles = tilize(x, M, K);
    auto b_tiles = bias_tiles(b, N);
    std::vector<uint32_t> w_bfp8;
    std::vector<uint16_t> w_bf16;
    distributed::EnqueueWriteMeshBuffer(cq, x_buf, x_tiles, false);
    if (bfp8) {
        const auto w_tiles = tiles_row_major(w, K, N);
        w_bfp8 = pack_as_bfp8_tiles<float>(
            ttsl::Span<const float>(w_tiles.data(), w_tiles.size()), /*row_major_input=*/true, /*is_exp_a=*/false);
        distributed::EnqueueWriteMeshBuffer(cq, w_buf, w_bfp8, false);
    } else {
        std::vector<uint16_t> w16(w.size());
        std::transform(w.begin(), w.end(), w16.begin(), to_bf16);
        w_bf16 = tilize(w16, K, N);
        distributed::EnqueueWriteMeshBuffer(cq, w_buf, w_bf16, false);
    }
    distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_tiles, false);

    // Spread the output tile columns over the cores.
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const auto grid_cores = static_cast<uint32_t>(grid.x * grid.y);
    const uint32_t n_cores = std::max<uint32_t>(1, std::min<uint32_t>({requested_cores, Nt, grid_cores}));
    std::vector<CoreCoord> core_list;
    for (uint32_t i = 0; i < n_cores; ++i) core_list.push_back({i % grid.x, i / grid.x});
    std::vector<CoreRange> ranges;
    for (const auto& c : core_list) ranges.emplace_back(c, c);
    const CoreRangeSet cores(ranges);

    Program program = CreateProgram();
    make_cb(program, cores, tt::CBIndex::c_0, 2, tt::DataFormat::Float16_b, BF16_TILE_BYTES);  // x
    make_cb(
        program,
        cores,
        tt::CBIndex::c_1,
        2,
        bfp8 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b,
        w_tile_bytes);                                                                          // w
    make_cb(program, cores, tt::CBIndex::c_2, 2, tt::DataFormat::Float16_b, BF16_TILE_BYTES);   // bias
    make_cb(program, cores, tt::CBIndex::c_24, 2, tt::DataFormat::Float32, FP32_TILE_BYTES);    // x @ w
    make_cb(program, cores, tt::CBIndex::c_25, 2, tt::DataFormat::Float16_b, BF16_TILE_BYTES);  // sum, bf16
    make_cb(program, cores, tt::CBIndex::c_16, 2, tt::DataFormat::Float16_b, BF16_TILE_BYTES);  // y

    std::vector<uint32_t> reader_ct;
    TensorAccessorArgs(*x_buf).append_to(reader_ct);
    TensorAccessorArgs(*w_buf).append_to(reader_ct);
    TensorAccessorArgs(*b_buf).append_to(reader_ct);
    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs(*y_buf).append_to(writer_ct);

    auto reader = CreateKernel(
        program,
        KERNEL_DIR "dataflow/reader_linear.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct});
    auto writer = CreateKernel(
        program,
        KERNEL_DIR "dataflow/writer_linear.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_ct});
    // HiFi4 with fp32 accumulation, ttnn.linear's settings in the GPT-2 study.
    auto compute = CreateKernel(
        program,
        KERNEL_DIR "compute/linear_bias_act.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = {act_id}});

    const uint32_t per_core = Nt / n_cores;
    const uint32_t extra = Nt % n_cores;
    uint32_t n_start = 0;
    for (uint32_t i = 0; i < n_cores; ++i) {
        const uint32_t n_count = per_core + (i < extra ? 1 : 0);
        const CoreCoord& core = core_list[i];
        SetRuntimeArgs(
            program,
            reader,
            core,
            {x_buf->address(), w_buf->address(), b_buf->address(), Mt, Kt, Nt, n_start, n_count});
        SetRuntimeArgs(program, compute, core, {Mt * n_count, Kt});
        SetRuntimeArgs(program, writer, core, {y_buf->address(), Mt, Nt, n_start, n_count});
        n_start += n_count;
    }

    const auto t0 = std::chrono::steady_clock::now();
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::vector<uint16_t> y_tiles;
    distributed::EnqueueReadMeshBuffer(cq, y_tiles, y_buf, true);
    const auto y = untilize(y_tiles, M, N);
    std::ofstream(argv[4], std::ios::binary)
        .write(reinterpret_cast<const char*>(y.data()), static_cast<std::streamsize>(y.size() * sizeof(uint16_t)));

    device->close();
    std::printf(
        "{\"rows\": %u, \"k\": %u, \"cols\": %u, \"tiles\": %u, \"cores\": %u, \"act\": \"%s\", "
        "\"weights\": \"%s\", \"seconds\": %.2f}\n",
        M,
        K,
        N,
        Mt * Nt,
        n_cores,
        act.c_str(),
        fmt.c_str(),
        secs);
    return 0;
}
