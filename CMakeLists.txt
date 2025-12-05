cmake_minimum_required(VERSION 3.18)

project(CudaTransferBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

add_executable(benchmark_cpp src/cpp/main.cu)
