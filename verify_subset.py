import pandas as pd
import re
import os
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_subset(log_file, label_df):
    block_ids = set(label_df['BlockId'])
    found_block_ids = set()

    logging.info(f"Starting verification for {len(block_ids)} block IDs in subset log...")
    try:
        verify_pattern = '|'.join([re.escape(bid) for bid in block_ids])
        compiled_verify_regex = re.compile(verify_pattern)
        logging.info("Verification regex compiled.")
    except Exception as e:
        logging.error(f"Error compiling verification regex: {e}")
        return

    total_verify_lines = 0
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f_count:
            for _ in f_count:
                total_verify_lines += 1
        logging.info(f"Total lines in subset log for verification: {total_verify_lines}")
    except Exception as e:
        logging.error(f"Error counting lines for verification: {e}")
        total_verify_lines = 0

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in tqdm(enumerate(f), total=total_verify_lines, desc="Verifying subset log"):
                if compiled_verify_regex.search(line):
                    matches = compiled_verify_regex.findall(line)
                    for match_bid in matches:
                        if match_bid in block_ids:
                            found_block_ids.add(match_bid)
        logging.info(f"Verification: {len(found_block_ids)}/{len(block_ids)} block IDs found in subset log")
        if block_ids - found_block_ids:
            logging.warning(f"Missing {len(block_ids - found_block_ids)} block IDs in subset log")
    except Exception as e:
        logging.error(f"Error verifying subset: {e}")

def main_verify():
    parser = argparse.ArgumentParser(description="Verify a subset of HDFS log dataset")
    parser.add_argument('--subset_log_file', default='data/subset/HDFS_subset.log', help='Path to the subset log file')
    parser.add_argument('--subset_label_file', default='data/subset/anomaly_label_subset.csv', help='Path to the subset label file')
    args = parser.parse_args()

    try:
        if not os.path.exists(args.subset_label_file):
            raise FileNotFoundError(f"'{args.subset_label_file}' not found. Ensure subset files are generated.")
        if not os.path.exists(args.subset_log_file):
            raise FileNotFoundError(f"'{args.subset_log_file}' not found. Ensure subset files are generated.")

        subset_df = pd.read_csv(args.subset_label_file)
        logging.info(f"Loaded existing subset labels from {args.subset_label_file}")
        logging.info(f"Subset contains {len(subset_df)} block IDs")
        logging.info(f"Normal/Anomaly distribution in loaded subset:\n{subset_df['Label'].value_counts()}")
    except Exception as e:
        logging.error(f"Error loading existing subset files for verification: {e}")
        return

    verify_subset(args.subset_log_file, subset_df)
    logging.info("Verification complete.")

if __name__ == "__main__":
    main_verify()
