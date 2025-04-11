#!/usr/bin/env python3

import os
import shutil


def collect_tpcm_and_csv(script_root):
    """
    Looks in 'PCMs/' (under script_root) for subdirectories named 'generated__XXXX.tpcm'.
    For each, finds a .tpcm file with the same name as the subdir and a Usage_Scenario*.csv.

    - If either doesn't exist, skip.
    - If the CSV only contains a header row (effectively empty), skip.
    - Otherwise, copy the .tpcm into 'tpcm/' and copy/rename the .csv into 'csv/' so they share the same base name.
    """

    # Directories
    input_dir = os.path.join(script_root, "PCMs")  # "PCMs/" is where subfolders live
    tpcm_output_dir = os.path.join(script_root, "data/dsl_models")  # output "tpcm/"
    csv_output_dir = os.path.join(script_root, "data/measurements")  # output "csv/"

    # Make sure output dirs exist.
    os.makedirs(tpcm_output_dir, exist_ok=True)
    os.makedirs(csv_output_dir, exist_ok=True)

    # If PCMs folder doesn't exist or is empty, just return.
    if not os.path.isdir(input_dir):
        print(f"ERROR: No 'PCMs/' directory found at {input_dir}")
        return

    # Iterate through items in PCMs/ looking for directories named like 'generated__XXXX.tpcm'
    for entry in os.listdir(input_dir):
        subfolder_path = os.path.join(input_dir, entry)

        # Check if it's a directory and follows the naming pattern
        if not (
            os.path.isdir(subfolder_path)
            and entry.startswith("generated__")
            and entry.endswith(".tpcm")
        ):
            continue  # skip anything else

        # The .tpcm file we want has the exact same name as the folder
        tpcm_filename = entry  # e.g. "generated__EAQHK.tpcm"
        tpcm_filepath = os.path.join(subfolder_path, tpcm_filename)

        # Check if .tpcm file exists
        if not os.path.isfile(tpcm_filepath):
            print(f"WARNING: No matching .tpcm file in {subfolder_path}, skipping.")
            continue

        # Search for usage scenario CSV (Usage_Scenario*.csv) anywhere in that folder tree
        usage_csv_path = None
        for root, dirs, files in os.walk(subfolder_path):
            for f in files:
                if f.startswith("Usage_Scenario") and f.endswith(".csv"):
                    usage_csv_path = os.path.join(root, f)
                    break
            if usage_csv_path:
                break

        if not usage_csv_path:
            print(
                f"WARNING: No usage scenario CSV found in {subfolder_path}, skipping."
            )
            continue

        # Check if the usage CSV is effectively empty (i.e., header-only)
        if is_header_only_csv(usage_csv_path):
            print(f"WARNING: CSV file in {subfolder_path} has no data rows, skipping.")
            continue

        # Copy the tpcm file into tpcm_output_dir (unchanged name)
        try:
            shutil.copy(tpcm_filepath, os.path.join(tpcm_output_dir, tpcm_filename))
        except Exception as e:
            print(f"ERROR copying .tpcm file from {subfolder_path}: {e}")
            continue

        # Rename the CSV to match the .tpcm file's base name, but with .csv
        base_name = os.path.splitext(tpcm_filename)[0]  # e.g. "generated__EAQHK"
        new_csv_filename = base_name + ".csv"  # "generated__EAQHK.csv"

        try:
            shutil.copy(usage_csv_path, os.path.join(csv_output_dir, new_csv_filename))
        except Exception as e:
            print(f"ERROR copying .csv file from {subfolder_path}: {e}")
            # Clean up the partially copied TPCM if CSV copy fails, to keep consistent
            dest_tpcm = os.path.join(tpcm_output_dir, tpcm_filename)
            if os.path.exists(dest_tpcm):
                os.remove(dest_tpcm)
            continue

        print(f"SUCCESS: Copied {tpcm_filename} and {new_csv_filename}")


def is_header_only_csv(csv_path):
    """
    Returns True if the CSV file at 'csv_path' has 0 or 1 non-header rows.
    In other words, if it doesn't have at least 2 lines total, we treat it as empty.
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # If there's 0 or 1 line, it means no real data
            return len(lines) < 2
    except Exception as e:
        print(f"ERROR reading CSV file {csv_path}: {e}")
        # If we can’t read it, treat it as empty
        return True


if __name__ == "__main__":
    # The script is meant to be placed in your repository root,
    # at the same level as the "PCMs/" folder.
    script_directory = os.path.abspath(os.path.dirname(__file__))
    collect_tpcm_and_csv(script_directory)
