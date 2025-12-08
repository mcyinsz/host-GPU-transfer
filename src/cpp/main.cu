#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <sstream>
#include <getopt.h>
#include <algorithm>

// 安全的错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// 测试配置结构体
struct TestConfig {
    std::vector<size_t> data_sizes;  // 以字节为单位
    int default_repeats;
    bool test_async;
    bool test_sync;
    bool verbose;
    bool csv_output;  // CSV输出模式
    int device_id;
    
    TestConfig() : 
        default_repeats(100), 
        test_async(true), 
        test_sync(true), 
        verbose(false), 
        csv_output(false),
        device_id(0) {}
};

// 解析命令行参数
TestConfig parse_arguments(int argc, char* argv[]) {
    TestConfig config;
    
    // 默认测试大小（如果未指定则使用这些）
    config.data_sizes = {
        1 * 1024,           // 1 KB
        4 * 1024,           // 4 KB
        16 * 1024,          // 16 KB
        64 * 1024,          // 64 KB
        256 * 1024,         // 256 KB
        1 * 1024 * 1024,    // 1 MB
        4 * 1024 * 1024,    // 4 MB
        16 * 1024 * 1024,   // 16 MB
        64 * 1024 * 1024,   // 64 MB
        256 * 1024 * 1024,  // 256 MB
        512 * 1024 * 1024,  // 512 MB
        1ULL * 1024 * 1024 * 1024,  // 1 GB
    };
    
    // 长选项定义
    static struct option long_options[] = {
        {"size", required_argument, 0, 's'},
        {"repeats", required_argument, 0, 'n'},
        {"mode", required_argument, 0, 'm'},
        {"device", required_argument, 0, 'd'},
        {"async-only", no_argument, 0, 'a'},
        {"sync-only", no_argument, 0, 'y'},
        {"verbose", no_argument, 0, 'v'},
        {"csv", no_argument, 0, 'c'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "s:n:m:d:avyh", long_options, &option_index)) != -1) {
        switch (c) {
            case 's': {  // 数据大小
                std::string size_str = optarg;
                config.data_sizes.clear();
                
                // 解析多个大小，支持逗号分隔
                size_t start = 0;
                size_t end = size_str.find(',');
                while (end != std::string::npos) {
                    std::string token = size_str.substr(start, end - start);
                    size_t size = std::stoull(token);
                    
                    // 如果数字小于1024，假设是MB单位，否则假设是字节
                    if (size < 1024) {
                        size *= 1024 * 1024;  // 转换为MB
                    }
                    config.data_sizes.push_back(size);
                    
                    start = end + 1;
                    end = size_str.find(',', start);
                }
                
                // 最后一个token
                std::string last_token = size_str.substr(start);
                if (!last_token.empty()) {
                    size_t size = std::stoull(last_token);
                    if (size < 1024) {
                        size *= 1024 * 1024;  // 转换为MB
                    }
                    config.data_sizes.push_back(size);
                }
                break;
            }
            
            case 'n':  // 迭代次数
                config.default_repeats = std::stoi(optarg);
                break;
                
            case 'm':  // 传输模式
                if (std::string(optarg) == "async") {
                    config.test_async = true;
                    config.test_sync = false;
                } else if (std::string(optarg) == "sync") {
                    config.test_async = false;
                    config.test_sync = true;
                } else if (std::string(optarg) == "both") {
                    config.test_async = true;
                    config.test_sync = true;
                } else {
                    std::cerr << "错误: 无效的模式 '" << optarg 
                              << "'. 使用 async, sync 或 both." << std::endl;
                    exit(1);
                }
                break;
                
            case 'd':  // 设备ID
                config.device_id = std::stoi(optarg);
                break;
                
            case 'a':  // 仅异步
                config.test_async = true;
                config.test_sync = false;
                break;
                
            case 'y':  // 仅同步
                config.test_async = false;
                config.test_sync = true;
                break;
                
            case 'v':  // 详细输出模式
                config.verbose = true;
                break;
                
            case 'c':  // CSV输出模式
                config.csv_output = true;
                config.verbose = false;  // CSV模式关闭详细输出
                break;
                
            case 'h':  // 帮助
                std::cout << "用法: " << argv[0] << " [选项]" << std::endl;
                std::cout << "CUDA主机到设备传输延迟测试工具" << std::endl;
                std::cout << std::endl;
                std::cout << "选项:" << std::endl;
                std::cout << "  -s, --size SIZE    测试数据大小（单位MB，逗号分隔多个值）" << std::endl;
                std::cout << "                      例如: -s 1,4,16,64,256" << std::endl;
                std::cout << "  -n, --repeats N    每个测试的迭代次数（默认: 100）" << std::endl;
                std::cout << "  -m, --mode MODE    传输模式: async, sync 或 both（默认: both）" << std::endl;
                std::cout << "  -d, --device ID    使用的CUDA设备ID（默认: 0）" << std::endl;
                std::cout << "  -a, --async-only   仅测试异步传输" << std::endl;
                std::cout << "  -y, --sync-only    仅测试同步传输" << std::endl;
                std::cout << "  -v, --verbose      详细输出模式" << std::endl;
                std::cout << "  -c, --csv          CSV输出格式（易于解析）" << std::endl;
                std::cout << "  -h, --help         显示此帮助信息" << std::endl;
                std::cout << std::endl;
                std::cout << "示例:" << std::endl;
                std::cout << "  " << argv[0] << " -s 1,4,16 -n 1000 -m async" << std::endl;
                std::cout << "  " << argv[0] << " -s 1024 -n 10 -d 1" << std::endl;
                std::cout << "  " << argv[0] << " --size 256,512,1024 --mode both --csv" << std::endl;
                exit(0);
                
            default:
                std::cerr << "使用 -h 或 --help 查看使用帮助" << std::endl;
                exit(1);
        }
    }
    
    return config;
}

// 测试结果结构体
struct TestResult {
    size_t data_size;       // 数据大小（字节）
    std::string mode;       // 传输模式
    double avg_latency_ms;  // 平均延迟（毫秒）
    double bandwidth_gbs;   // 带宽（GB/s）
    double min_latency_ms;  // 最小延迟
    double max_latency_ms;  // 最大延迟
    double peak_bandwidth_gbs; // 峰值带宽
};

// 测量特定数据大小的延迟和带宽
TestResult run_bandwidth_test(size_t data_size, int n_repeat, bool use_async, 
                              bool verbose, bool csv_output, const cudaDeviceProp& prop) {
    TestResult result;
    result.data_size = data_size;
    result.mode = use_async ? "async" : "sync";
    
    if (verbose && !csv_output) {
        std::cout << "\n[准备测试] 数据大小: " << data_size << " 字节"
                  << ", 模式: " << (use_async ? "异步" : "同步")
                  << ", 迭代: " << n_repeat << " 次" << std::endl;
    }
    
    // 验证数据大小
    if (data_size > prop.totalGlobalMem) {
        if (!csv_output) {
            std::cerr << "错误: 数据大小 " << data_size 
                      << " 超过设备显存 " << prop.totalGlobalMem << std::endl;
        }
        result.avg_latency_ms = -1;
        result.bandwidth_gbs = -1;
        return result;
    }
    
    if (data_size < sizeof(float)) {
        if (!csv_output) {
            std::cerr << "错误: 数据大小必须至少为 " << sizeof(float) << " 字节" << std::endl;
        }
        result.avg_latency_ms = -1;
        result.bandwidth_gbs = -1;
        return result;
    }
    
    // 1. 分配锁页内存
    float *h_data = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_data, data_size));
    
    // 2. 分配设备内存
    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, data_size));
    
    // 3. 初始化数据
    size_t num_elements = data_size / sizeof(float);
    for (size_t i = 0; i < num_elements; ++i) {
        h_data[i] = static_cast<float>(i % 256) * 0.01f;
    }
    
    // 4. 创建CUDA流（仅用于异步传输）
    cudaStream_t stream = nullptr;
    if (use_async) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    
    // 5. 创建事件
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 6. 预热
    for (int i = 0; i < 3; ++i) {
        if (use_async) {
            CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, data_size, 
                                       cudaMemcpyHostToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpy(d_data, h_data, data_size, 
                                  cudaMemcpyHostToDevice));
        }
    }
    
    if (use_async) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 7. 多次测量以减少误差
    const int num_trials = 5;
    double total_ms = 0.0;
    double min_ms = 1e9;
    double max_ms = 0.0;
    double best_bandwidth = 0.0;
    
    for (int trial = 0; trial < num_trials; ++trial) {
        // 记录开始时间
        if (use_async) {
            CUDA_CHECK(cudaEventRecord(start, stream));
        } else {
            CUDA_CHECK(cudaEventRecord(start, 0));
        }
        
        // 执行多次传输
        for (int i = 0; i < n_repeat; ++i) {
            if (use_async) {
                CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, data_size, 
                                           cudaMemcpyHostToDevice, stream));
            } else {
                CUDA_CHECK(cudaMemcpy(d_data, h_data, data_size, 
                                      cudaMemcpyHostToDevice));
            }
        }
        
        // 记录结束时间
        if (use_async) {
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
        }
        
        // 计算耗时
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        double avg_ms = milliseconds / n_repeat;
        total_ms += avg_ms;
        min_ms = std::min(min_ms, avg_ms);
        max_ms = std::max(max_ms, avg_ms);
        
        // 计算带宽
        double bandwidth = data_size / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        best_bandwidth = std::max(best_bandwidth, bandwidth);
        
        if (verbose && !csv_output) {
            std::cout << "  尝试 " << trial + 1 << ": " 
                      << std::fixed << std::setprecision(6) << avg_ms << " ms, "
                      << std::fixed << std::setprecision(3) << bandwidth << " GB/s" << std::endl;
        }
    }
    
    // 8. 计算统计结果
    result.avg_latency_ms = total_ms / num_trials;
    result.min_latency_ms = min_ms;
    result.max_latency_ms = max_ms;
    result.bandwidth_gbs = data_size / (result.avg_latency_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    result.peak_bandwidth_gbs = best_bandwidth;
    
    if (verbose && !csv_output) {
        std::cout << "  结果: " << std::fixed << std::setprecision(6) << result.avg_latency_ms << " ms";
        
        // 对于小数据，显示微秒级的延迟
        if (result.avg_latency_ms < 1.0) {
            std::cout << " (" << std::fixed << std::setprecision(3) << result.avg_latency_ms * 1000.0 << " μs)";
        }
        
        std::cout << ", " << std::fixed << std::setprecision(3) << result.bandwidth_gbs << " GB/s";
        
        // 显示带宽峰值
        if (best_bandwidth > result.bandwidth_gbs * 1.05) {
            std::cout << " (峰值: " << best_bandwidth << " GB/s)";
        }
        
        std::cout << std::endl;
    }
    
    // 9. 清理资源
    if (use_async && stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return result;
}

// 主函数
int main(int argc, char* argv[]) {
    // 解析命令行参数
    TestConfig config = parse_arguments(argc, argv);
    
    // 设置CUDA设备
    CUDA_CHECK(cudaSetDevice(config.device_id));
    
    // 获取设备信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, config.device_id));
    
    // 显示标题（仅在非CSV模式下）
    if (!config.csv_output) {
        std::cout << "==================================================" << std::endl;
        std::cout << "CUDA 主机到设备传输延迟测试" << std::endl;
        std::cout << "设备 " << config.device_id << ": " << prop.name << std::endl;
        std::cout << "显存: " << prop.totalGlobalMem << " 字节" << std::endl;
        std::cout << "PCIe总线: " << prop.pciBusID << std::endl;
        std::cout << "PCIe设备: " << prop.pciDeviceID << std::endl;
        std::cout << "==================================================" << std::endl;
        
        if (config.verbose) {
            std::cout << "配置:" << std::endl;
            std::cout << "  测试大小: ";
            for (size_t i = 0; i < config.data_sizes.size(); ++i) {
                std::cout << config.data_sizes[i];
                if (i < config.data_sizes.size() - 1) std::cout << ", ";
            }
            std::cout << " 字节" << std::endl;
            std::cout << "  迭代次数: " << config.default_repeats << std::endl;
            std::cout << "  测试模式: ";
            if (config.test_async && config.test_sync) std::cout << "异步和同步";
            else if (config.test_async) std::cout << "仅异步";
            else std::cout << "仅同步";
            std::cout << std::endl;
            std::cout << "==================================================" << std::endl;
        }
    }
    
    // 对数据大小进行排序
    std::sort(config.data_sizes.begin(), config.data_sizes.end());
    
    // 输出CSV头部（仅在CSV模式下）
    if (config.csv_output) {
        std::cout << "data_size_bytes,mode,avg_latency_ms,bandwidth_gbs,min_latency_ms,max_latency_ms,peak_bandwidth_gbs" << std::endl;
    }
    
    // 收集所有结果
    std::vector<TestResult> all_results;
    
    // 运行测试
    for (size_t data_size : config.data_sizes) {
        // 根据数据大小调整迭代次数
        int n_repeat = config.default_repeats;
        if (data_size < 1024 * 1024) {  // < 1 MB
            n_repeat = std::max(n_repeat, 1000);
        } else if (data_size > 512 * 1024 * 1024) {  // > 512 MB
            n_repeat = std::min(n_repeat, 20);
        }
        
        if (config.test_async) {
            TestResult result = run_bandwidth_test(data_size, n_repeat, true, 
                                                   config.verbose, config.csv_output, prop);
            all_results.push_back(result);
        }
        
        if (config.test_sync) {
            TestResult result = run_bandwidth_test(data_size, n_repeat, false, 
                                                   config.verbose, config.csv_output, prop);
            all_results.push_back(result);
        }
        
        if (config.verbose && !config.csv_output) {
            std::cout << "--------------------------------------------------" << std::endl;
        }
    }
    
    // 输出结果（CSV模式输出所有结果）
    if (config.csv_output) {
        for (const auto& result : all_results) {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << result.data_size << ",";
            std::cout << result.mode << ",";
            std::cout << result.avg_latency_ms << ",";
            std::cout << std::setprecision(3) << result.bandwidth_gbs << ",";
            std::cout << std::setprecision(6) << result.min_latency_ms << ",";
            std::cout << result.max_latency_ms << ",";
            std::cout << std::setprecision(3) << result.peak_bandwidth_gbs << std::endl;
        }
    } else {
        // 显示总结（非CSV模式）
        std::cout << "\n==================================================" << std::endl;
        std::cout << "测试完成!" << std::endl;
        std::cout << "总共测试了 " << all_results.size() << " 个数据点" << std::endl;
        std::cout << "==================================================" << std::endl;
        
        // 输出简单表格（非CSV模式）
        std::cout << "\n测试结果汇总:" << std::endl;
        std::cout << std::left << std::setw(12) << "数据大小(字节)"
                  << std::setw(8) << "模式"
                  << std::setw(15) << "平均延迟(ms)"
                  << std::setw(12) << "带宽(GB/s)" << std::endl;
        std::cout << std::string(47, '-') << std::endl;
        
        for (const auto& result : all_results) {
            std::cout << std::left << std::setw(12) << result.data_size
                      << std::setw(8) << result.mode
                      << std::fixed << std::setprecision(6)
                      << std::setw(15) << result.avg_latency_ms
                      << std::setprecision(3)
                      << std::setw(12) << result.bandwidth_gbs << std::endl;
        }
    }
    
    // 重置设备
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}