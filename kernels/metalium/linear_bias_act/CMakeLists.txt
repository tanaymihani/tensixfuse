cmake_minimum_required(VERSION 3.22)
project(tensixfuse_linear_bias_act LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(TT-Metalium REQUIRED)

add_executable(linear_bias_act linear_bias_act.cpp)
target_link_libraries(linear_bias_act PRIVATE TT::Metalium)
# Device kernels are compiled at runtime by tt-metal; point it at the sources.
target_compile_definitions(linear_bias_act PRIVATE KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
