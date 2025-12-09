import numpy as np

profile_data_size = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592]
profile_latency_ms = [0.00423, 0.004328, 0.004104, 0.004041, 0.004493, 0.006025, 0.007554, 0.010447, 0.019243, 0.037471, 0.074138, 0.147206, 0.29305, 0.584735, 1.168635, 2.336072, 4.670362, 9.339269, 18.678262, 37.355894, 74.731451, 149.501804, 299.221802, 597.798157]

def get_transfer_latency_ms(data_size_Byte: float) -> float:

    if data_size_Byte <= profile_data_size[0]:
        return profile_latency_ms[0]
    if data_size_Byte >= profile_data_size[-1]:
        return profile_latency_ms[-1]

    log_data_sizes = np.log10(profile_data_size)
    log_data_size = np.log10(data_size_Byte)

    idx = np.searchsorted(log_data_sizes, log_data_size) - 1

    if idx < 0:
        idx = 0
    if idx >= len(log_data_sizes) - 1:
        idx = len(log_data_sizes) - 2

    x1, x2 = log_data_sizes[idx], log_data_sizes[idx + 1]
    y1, y2 = profile_latency_ms[idx], profile_latency_ms[idx + 1]

    latency = y1 + (log_data_size - x1) * (y2 - y1) / (x2 - x1)
    
    return latency

if __name__ == "__main__":
    for transfer_data_size in [1000,10000,100000,1000000,10000000,8000000000]:
        print(get_transfer_latency_ms(transfer_data_size))