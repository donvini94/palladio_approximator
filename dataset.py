import os
import pandas as pd
import glob
import random

from sklearn.model_selection import train_test_split


def load_dataset(
    data_dir, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
):
    """
    Loads DSL model text files and corresponding response time measurements,
    calculates summary statistics, and returns train/val/test splits as DataFrames.

    Parameters:
        data_dir (str): Path to the data directory.
        train_size (float): Proportion for the training split.
        val_size (float): Proportion for the validation split.
        test_size (float): Proportion for the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    dsl_dir = os.path.join(data_dir, "dsl_models")
    csv_dir = os.path.join(data_dir, "measurements")
    resp_time_col = "Response Time[s]"

    samples = []

    for dsl_file in glob.glob(os.path.join(dsl_dir, "*.tpcm")):
        # get the first part of model_X.tpcm, aka model_X, (.tpcm is the second part)
        base_name = os.path.splitext(os.path.basename(dsl_file))[0]
        csv_file = os.path.join(csv_dir, f"{base_name}.csv")

        if not os.path.isfile(csv_file):
            continue

        with open(dsl_file, encoding="utf-8") as f:
            tpcm_text = f.read()

        df = pd.read_csv(csv_file)
        if resp_time_col not in df.columns:
            continue

        resp_times = df[resp_time_col].dropna()
        if resp_times.empty:
            continue

        samples.append(
            {
                "tpcm_text": tpcm_text,
                "avg_resp_time": resp_times.mean(),
                "min_resp_time": resp_times.min(),
                "max_resp_time": resp_times.max(),
            }
        )

    if not samples:
        raise ValueError(f"No valid samples found in {data_dir}.")

    all_samples_df = pd.DataFrame(samples)
    print(f"Number of samples: {len(all_samples_df)}")

    # First split into train and temp
    train_df, temp_df = train_test_split(
        all_samples_df, train_size=train_size, random_state=random_state
    )

    # Then split temp into validation and test
    temp_size = val_size + test_size
    val_ratio = val_size / temp_size
    val_df, test_df = train_test_split(
        temp_df, train_size=val_ratio, random_state=random_state
    )

    # Save all samples for inspection
    output_csv = os.path.join(data_dir, "all_samples.csv")
    all_samples_df.to_csv(output_csv, index=False)
    print(f"Saved all samples to: {output_csv}")

    return train_df, val_df, test_df


def build_time_series_dataset(data_dir):
    """
    Builds a dataset for time-series prediction.
    Each sample is (dsl_text, time_value) → response_time.

    Parameters:
        data_dir (str): Path to the dataset folder with 'dsl_models/' and 'measurements/'

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, texts_train, texts_val, texts_test)
    """
    dsl_dir = os.path.join(data_dir, "dsl_models")
    csv_dir = os.path.join(data_dir, "measurements")

    dsl_files = glob.glob(os.path.join(dsl_dir, "*.tpcm"))
    samples = []

    for dsl_path in dsl_files:
        base = os.path.splitext(os.path.basename(dsl_path))[0]
        csv_path = os.path.join(csv_dir, base + ".csv")
        if not os.path.exists(csv_path):
            continue

        with open(dsl_path, "r") as f:
            dsl_text = f.read()

        df = pd.read_csv(csv_path)
        if "Response Time[s]" not in df.columns or "Point in Time[s]" not in df.columns:
            continue

        for _, row in df.iterrows():
            if pd.isna(row["Response Time[s]"]) or pd.isna(row["Point in Time[s]"]):
                continue
            sample = {
                "tpcm_text": dsl_text,
                "time_value": row["Point in Time[s]"],
                "response_time": row["Response Time[s]"],
            }
            samples.append(sample)

    random.shuffle(samples)
    # Save all samples to a CSV file for inspection/debugging
    output_path = os.path.join(data_dir, "timeseries_samples.csv")
    pd.DataFrame(samples).to_csv(output_path, index=False)
    print(f"Saved all samples to: {output_path}")

    print("No of samples: ", len(samples))
    n = len(samples)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.15)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    def unpack(samples):
        X_raw = [(s["tpcm_text"], s["time_value"]) for s in samples]
        y = [s["response_time"] for s in samples]
        texts = [s["tpcm_text"] for s in samples]
        return X_raw, y, texts

    X_train, y_train, texts_train = unpack(train_samples)
    X_val, y_val, texts_val = unpack(val_samples)
    X_test, y_test, texts_test = unpack(test_samples)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        texts_train,
        texts_val,
        texts_test,
    )
