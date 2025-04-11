import os
import pandas as pd
import glob
import random


def load_dataset(data_dir):
    """
    Loads DSL model text files and corresponding response time measurements,
    calculates summary statistics, and returns train/val/test splits.

    Parameters:
        data_dir (str): Path to the data directory.

    Returns:
        tuple: (train_samples, val_samples, test_samples)
    """
    dsl_dir = os.path.join(data_dir, "dsl_models")
    csv_dir = os.path.join(data_dir, "measurements")

    dsl_files = glob.glob(os.path.join(dsl_dir, "*.tpcm"))
    samples = []

    for dsl_path in dsl_files:

        base_name = os.path.splitext(os.path.basename(dsl_path))[0]
        csv_path = os.path.join(csv_dir, base_name + ".csv")
        if not os.path.exists(csv_path):
            continue

        with open(dsl_path, "r") as f:
            tpcm_text = f.read()

        df = pd.read_csv(csv_path)
        if "Response Time[s]" not in df.columns:
            continue

        resp_times = df["Response Time[s]"].dropna()
        if len(resp_times) == 0:
            continue

        sample = {
            "tpcm_text": tpcm_text,
            "avg_resp_time": resp_times.mean(),
            "min_resp_time": resp_times.min(),
            "max_resp_time": resp_times.max(),
        }
        samples.append(sample)

    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.15)
    print("No of samples: ", len(samples))

    return samples[:train_end], samples[train_end:val_end], samples[val_end:]
