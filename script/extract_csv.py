import pandas as pd
import subprocess
import io
from env import *
import os
import datetime


if __name__ == "__main__":

    # profile params
    data_size = map(lambda x: str(x), [1<<(10+i) for i in range(24)])
    

    result = subprocess.run([os.path.join(BUILD_DIR, "benchmark_cpp"), '--csv', '--size', ",".join(data_size), '--mode', 'async'], 
                            capture_output=True, text=True)

    df = pd.read_csv(io.StringIO(result.stdout))
    print(df["data_size_bytes"])
    print(df["avg_latency_ms"])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(os.path.join(RESULT_DIR, f"benchmark_{timestamp}.csv"), index=False)