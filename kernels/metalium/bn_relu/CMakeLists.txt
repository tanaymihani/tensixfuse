cmake_minimum_required(VERSION 3.22)
project(tensixfuse_bn_relu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(TT-Metalium REQUIRED)

add_executable(bn_relu bn_relu.cpp)
target_link_libraries(bn_relu PRIVATE TT::Metalium)
# Device kernels are compiled at runtime by tt-metal; point it at the sources.
target_compile_definitions(bn_relu PRIVATE KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
