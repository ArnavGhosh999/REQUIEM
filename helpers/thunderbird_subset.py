import random
from tqdm import tqdm

def create_subset(input_file, output_file, total_lines=2_000_000, seed=42):
    random.seed(seed)

    total_log_lines = 211_212_192
    total_alerts = 3_248_239
    total_normals = 207_963_953

    alert_ratio = total_alerts / total_log_lines
    alert_target = int(total_lines * alert_ratio)
    normal_target = total_lines - alert_target

    alerts = []
    normals = []

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, total=total_log_lines, desc="Scanning logs"):
            if not line.strip():
                continue

            if line.startswith("- "):
                if len(normals) < normal_target:
                    if random.random() < (normal_target / total_normals):
                        normals.append(line)
            else:
                if len(alerts) < alert_target:
                    if random.random() < (alert_target / total_alerts):
                        alerts.append(line)

            if len(alerts) >= alert_target and len(normals) >= normal_target:
                break

    subset = alerts + normals
    random.shuffle(subset)

    with open(output_file, "w", encoding="utf-8") as out:
        out.writelines(subset)

    print(f"\nSubset created: {output_file}")
    print(f"Total lines: {len(subset):,} (Alerts: {len(alerts):,}, Normals: {len(normals):,})")
    print(f"Alert %: {(len(alerts) / len(subset)) * 100:.6f}%")

if __name__ == "__main__":
    create_subset("Thunderbird.log", "Thunderbird_subset.log")
