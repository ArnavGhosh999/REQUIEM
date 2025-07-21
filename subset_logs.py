import pandas as pd
import re
import os
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f"Pandas version as seen by script: {pd.__version__}")

def validate_files(log_file, label_file):
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}. Ensure 'HDFS.log' is in the 'data/' directory.")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}. Ensure 'anomaly_label.csv' is in the 'data/' directory.")
    labels_df = pd.read_csv(label_file)
    required_columns = ['BlockId', 'Label']
    if not all(col in labels_df.columns for col in required_columns):
        raise ValueError(f"Label file must contain columns: {required_columns}")
    return labels_df

def main():
    parser = argparse.ArgumentParser(description="Create a subset of HDFS_1 log dataset")
    parser.add_argument('--log_file', default='data/HDFS.log', help='Path to HDFS.log')
    parser.add_argument('--label_file', default='data/anomaly_label.csv', help='Path to anomaly_label.csv')
    parser.add_argument('--output_dir', default='data/subset/', help='Output directory for subset files')
    parser.add_argument('--frac', type=float, default=0.1, help='Fraction of block IDs to sample (0.0–1.0)')
    args = parser.parse_args()

    try:
        labels_df = validate_files(args.log_file, args.label_file)
    except Exception as e:
        logging.error(f"Input validation failed: {e}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    subset_log_file = os.path.join(args.output_dir, 'HDFS_subset.log')
    subset_label_file = os.path.join(args.output_dir, 'anomaly_label_subset.csv')

    try:
        logging.info(f"Original label distribution:\n{labels_df['Label'].value_counts()}")

        normal_labels = labels_df[labels_df['Label'] == 'Normal']
        anomaly_labels = labels_df[labels_df['Label'] == 'Anomaly']

        num_normal_samples = max(1, round(len(normal_labels) * args.frac)) if len(normal_labels) > 0 else 0
        num_anomaly_samples = max(1, round(len(anomaly_labels) * args.frac)) if len(anomaly_labels) > 0 else 0

        num_normal_samples = min(num_normal_samples, len(normal_labels))
        num_anomaly_samples = min(num_anomaly_samples, len(anomaly_labels))

        logging.info(f"Sampling {num_normal_samples} Normal samples and {num_anomaly_samples} Anomaly samples.")

        sampled_normal_df = normal_labels.sample(n=num_normal_samples, random_state=42)
        sampled_anomaly_df = anomaly_labels.sample(n=num_anomaly_samples, random_state=42)

        subset_df = pd.concat([sampled_normal_df, sampled_anomaly_df])
        subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)

        subset_df.to_csv(subset_label_file, index=False)
        logging.info(f"Subset labels saved to {subset_label_file}")
        logging.info(f"Subset contains {len(subset_df)} block IDs")
        logging.info(f"Normal/Anomaly distribution in subset:\n{subset_df['Label'].value_counts()}")
    except Exception as e:
        logging.error(f"Error during manual label sampling: {e}")
        return

    block_ids = set(subset_df['BlockId'])

    logging.info(f"Compiling regex pattern for {len(block_ids)} block IDs...")
    try:
        block_id_pattern = '|'.join([re.escape(bid) for bid in block_ids])
        compiled_block_id_regex = re.compile(block_id_pattern)
        logging.info("Regex compilation complete.")
    except Exception as e:
        logging.error(f"Error compiling regex pattern: {e}")
        return

    try:
        total_lines = 0
        with open(args.log_file, 'r', encoding='utf-8', errors='ignore') as f_count:
            for _ in f_count:
                total_lines += 1
        logging.info(f"Total lines in log file: {total_lines}")
    except Exception as e:
        logging.error(f"Error counting lines in log file: {e}")
        total_lines = 0 

    try:
        matched_lines_count = 0
        with open(subset_log_file, 'w', encoding='utf-8') as outfile:
            with open(args.log_file, 'r', encoding='utf-8', errors='ignore') as infile:
                for line_num, line in tqdm(enumerate(infile), total=total_lines, desc="Processing log lines"):
                    if compiled_block_id_regex.search(line):
                        outfile.write(line)
                        matched_lines_count += 1
        logging.info(f"Subset logs saved to {subset_log_file} ({matched_lines_count} lines written)")
    except Exception as e:
        logging.error(f"Error processing log file: {e}")
        return

if __name__ == "__main__":
    main()
