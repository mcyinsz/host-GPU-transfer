import torch
import time

def run_benchmark():
    # 配置必须与 C++ 保持一致
    data_size_bytes = 100 * 1024 * 1024
    n_repeat = 100
    
    device = torch.device("cuda:0")
    
    print(f"[Python PyTorch] Settings: {data_size_bytes // (1024*1024)} MB, {n_repeat} iterations.")

    # 1. 准备数据 (Pinned Memory)
    # pin_memory() 对应 cudaMallocHost
    host_data = torch.randn(data_size_bytes // 4, dtype=torch.float32).pin_memory()
    
    # 2. 预热
    for _ in range(5):
        _ = host_data.to(device, non_blocking=True)
    torch.cuda.synchronize()

    # 3. 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_repeat):
        _ = host_data.to(device, non_blocking=True)
    end_event.record()
    
    end_event.synchronize()

    # 4. 计算
    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / n_repeat
    bandwidth_gb = (data_size_bytes * n_repeat) / (total_ms / 1000.0) / (1024**3)

    print(f"Avg Latency: {avg_ms:.4f} ms")
    print(f"Bandwidth:   {bandwidth_gb:.4f} GB/s")

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_benchmark()
    else:
        print("CUDA not available.")