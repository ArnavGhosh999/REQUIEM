import random

def stratified_sample(input_path, output_path, total_samples=400_000, max_anomalies=None):
    # First pass: count normals and anomalies
    normal_count = 0
    anomaly_count = 0
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('-'):
                normal_count += 1
            else:
                anomaly_count += 1

    # Set how much of each class to sample
    if max_anomalies is not None and anomaly_count > max_anomalies:
        anom_samples = max_anomalies
        norm_samples = total_samples - anom_samples
    else:
        norm_frac = normal_count / (normal_count + anomaly_count)
        anom_frac = anomaly_count / (normal_count + anomaly_count)
        anom_samples = int(total_samples * anom_frac)
        norm_samples = total_samples - anom_samples

    print(f"Sampling {norm_samples} normals, {anom_samples} anomalies ({anom_samples/(anom_samples+norm_samples):.2%} anomalies)")

    # Reservoir-sample each class
    normals, anomalies = [], []
    cur_norm = 0
    cur_anom = 0
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('-'):
                if len(normals) < norm_samples:
                    normals.append(line)
                else:
                    m = random.randint(0, cur_norm)
                    if m < norm_samples:
                        normals[m] = line
                cur_norm += 1
            else:
                if len(anomalies) < anom_samples:
                    anomalies.append(line)
                else:
                    m = random.randint(0, cur_anom)
                    if m < anom_samples:
                        anomalies[m] = line
                cur_anom += 1

    with open(output_path, 'w', encoding='utf-8') as out:
        for l in normals + anomalies:
            out.write(l)

# Example usage (oversample all anomalies, cap at 40k, sample the rest as normals)
stratified_sample("data/BGL.log", "BGL_strat_sampled.log", total_samples=400_000, max_anomalies=40_000)
stratified_sample("data/Thunderbird_subset.log", "Thunderbird_strat_sampled.log", total_samples=400_000, max_anomalies=40_000)