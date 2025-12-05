#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// 简单的错误检查宏
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

int main(int argc, char* argv[]) {
    // 默认配置：100 MB, 重复 100 次
    size_t data_size = 100 * 1024 * 1024;
    int n_repeat = 100;

    std::cout << "[C++ Native] Settings: " << data_size / (1024*1024) << " MB, " 
              << n_repeat << " iterations." << std::endl;

    // 1. 分配 Pinned Memory (Host)
    float *h_data;
    CHECK_CUDA(cudaMallocHost(&h_data, data_size)); // 关键：锁页内存

    // 2. 分配 Device Memory
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, data_size));

    // 初始化数据
    memset(h_data, 0, data_size);

    // 3. 创建 Event
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 4. 预热 (Warm up)
    for(int i=0; i<5; ++i) {
        cudaMemcpyAsync(d_data, h_data, data_size, cudaMemcpyHostToDevice, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. 测量
    CHECK_CUDA(cudaEventRecord(start));
    
    for (int i = 0; i < n_repeat; ++i) {
        CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, data_size, cudaMemcpyHostToDevice, 0));
    }
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float avg_ms = milliseconds / n_repeat;
    float bandwidth_gb = (data_size * n_repeat) / (milliseconds / 1000.0) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Avg Latency: " << avg_ms << " ms" << std::endl;
    std::cout << "Bandwidth:   " << bandwidth_gb << " GB/s" << std::endl;

    // 清理
    cudaFreeHost(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}