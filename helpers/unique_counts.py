import re
from collections import defaultdict
from tqdm import tqdm

def count_lines(filename):
    """Quick function to count total lines in a file for tqdm."""
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)

def normalize_message(line):
    """
    Normalize a log line by removing variable fields (timestamp, node IDs, etc.),
    but preserve the log level, kernel info, message, and prefix.
    """
    prefix = '-' if line.startswith('-') else ''
    body = line[1:].lstrip() if prefix else line.lstrip()
    split_idx = body.find(":")
    if split_idx != -1:
        canonical_msg = body[split_idx+1:].strip()
    else:
        canonical_msg = body
    return f"{prefix} {canonical_msg}".strip()

def process_log_file(input_path, output_path, max_lines=None):
    msg_counter = defaultdict(int)
    total_lines = 0
    n_lines = count_lines(input_path)

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f, tqdm(total=min(n_lines, max_lines) if max_lines else n_lines, desc=f"Processing {input_path}") as pbar:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            normed = normalize_message(line.rstrip())
            msg_counter[normed] += 1
            total_lines += 1
            pbar.update(1)

    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(f"Total logs: {total_lines}\n")
        out.write(f"Unique normalized messages: {len(msg_counter)}\n\n")
        out.write("=== Unique Normalized Log Messages (prefix + message) ===\n")
        for msg, count in sorted(msg_counter.items(), key=lambda x: -x[1]):
            out.write(f"{count} :: {msg}\n")

process_log_file("data/BGL.log", "BGL_normalized_unique_log_summary.txt")
process_log_file("data/Thunderbird_subset.log", "Thunderbird_normalized_unique_log_summary.txt")