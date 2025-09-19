import pandas as pd

def extract_cases(pkl_file, output_prefix, n_per_type=50):
    df = pd.read_pickle(pkl_file)
    # --- Normals in noise or anomaly-like clusters ---
    mixed_clusters = [-1, 6, 21, 27, 41, 48, 55]  # from your BGL analysis
    # Change as needed for Thunderbird, e.g., [-1]
    # Normals in noise or mixed clusters
    normals_noise_or_mixed = df[(df['label']==0) & (df['hdbscan_cluster'].isin(mixed_clusters))]
    # Anomalies in normal-dominated clusters:
    clusters = df.groupby('hdbscan_cluster')['label'].mean()
    normal_clusters = clusters[clusters < 0.2].index
    anomalies_in_norm = df[(df['label']==1) & (df['hdbscan_cluster'].isin(normal_clusters))]
    # Sample and save
    normals_noise_or_mixed.sample(min(n_per_type, len(normals_noise_or_mixed))).to_csv(f'{output_prefix}_normals.csv', index=False)
    anomalies_in_norm.sample(min(n_per_type, len(anomalies_in_norm))).to_csv(f'{output_prefix}_anomalies.csv', index=False)
    print(f"Saved hard samples for: {output_prefix}")

if __name__ == "__main__":
    extract_cases('BGL_clustered_centroids.pkl', 'BGL_hard_cases', n_per_type=50)
    extract_cases('Thunderbird_clustered_centroids.pkl', 'Thunderbird_hard_cases', n_per_type=50)
