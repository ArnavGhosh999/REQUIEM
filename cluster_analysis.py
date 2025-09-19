import pandas as pd
from collections import Counter

def analyze_clusters(pkl_file):
    df = pd.read_pickle(pkl_file)
    print(f"Total points: {len(df)}")
    cluster_labels = df['hdbscan_cluster'].unique()
    print(f"Total clusters: {len(cluster_labels) - (1 if -1 in cluster_labels else 0)} (excluding noise)")
    for c in sorted(cluster_labels):
        sub = df[df['hdbscan_cluster'] == c]
        label_counts = Counter(sub['label'])
        print(f"Cluster {c}: count={len(sub)} normal={label_counts[0]} anomaly={label_counts[1]}")
    # List mixed clusters
    mixed = df.groupby('hdbscan_cluster')['label'].nunique()
    mixed_clusters = mixed[mixed > 1].index.tolist()
    print(f"Mixed clusters (contain normal+anomaly): {mixed_clusters}")

if __name__ == "__main__":
    analyze_clusters('BGL_embeddings_hdbscan.pkl')
    analyze_clusters('Thunderbird_embeddings_hdbscan.pkl')
