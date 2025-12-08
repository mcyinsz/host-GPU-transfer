import pandas as pd
import subprocess
import io
from env import *
import os
import datetime


if __name__ == "__main__":

    # profile params
    data_size = map(lambda x: str(x), [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10])
    

    result = subprocess.run([os.path.join(BUILD_DIR, "benchmark_cpp"), '--csv', '--size', ",".join(data_size), '--mode', 'async'], 
                            capture_output=True, text=True)

    df = pd.read_csv(io.StringIO(result.stdout))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(os.path.join(RESULT_DIR, f"benchmark_{timestamp}.csv"), index=False)