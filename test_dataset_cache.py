#!/usr/bin/env python3
"""
Test script to verify the dataset caching functionality.
This script loads the dataset with caching enabled, measures the time, then loads it again.
"""

import os
import time
import argparse
from dataset import load_dataset


def main():
    """Test dataset caching functionality"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    
    # Ensure cache directory exists
    cache_dir = os.path.join(args.data_dir, "cache")
    cache_path = os.path.join(cache_dir, "dataset_splits.pkl")
    
    # Check if cache already exists and report
    if os.path.exists(cache_path):
        print(f"Cache already exists at {cache_path}")
        cache_size = os.path.getsize(cache_path) / (1024 * 1024)  # Size in MB
        print(f"Cache size: {cache_size:.2f} MB")
    else:
        print(f"No cache found at {cache_path}")
    
    # First load: Should generate and save dataset
    print("\n=== First load (with cache generation) ===")
    start_time = time.time()
    train_df, val_df, test_df = load_dataset(
        args.data_dir, 
        save_dataset=True,
        load_dataset=False
    )
    first_load_time = time.time() - start_time
    print(f"First load time (no cache): {first_load_time:.2f} seconds")
    print(f"Dataset sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Second load: Should use cached dataset
    print("\n=== Second load (from cache) ===")
    start_time = time.time()
    train_df2, val_df2, test_df2 = load_dataset(
        args.data_dir, 
        save_dataset=False,
        load_dataset=True
    )
    second_load_time = time.time() - start_time
    print(f"Second load time (from cache): {second_load_time:.2f} seconds")
    
    # Verify datasets are the same
    print("\n=== Verification ===")
    train_match = len(train_df) == len(train_df2)
    val_match = len(val_df) == len(val_df2)
    test_match = len(test_df) == len(test_df2)
    
    if train_match and val_match and test_match:
        print("✓ Dataset sizes match")
    else:
        print("✗ Dataset sizes don't match")
        
    # Calculate speedup
    if first_load_time > 0:
        speedup = first_load_time / max(0.001, second_load_time)
        print(f"Cache speedup: {speedup:.1f}x faster")
    
    print(f"\nDataset cache is {'enabled' if os.path.exists(cache_path) else 'disabled'}")
    
    return 0


if __name__ == "__main__":
    exit(main())