import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def add_centroid_scores(pkl_file, output_pkl, use_pca=True):
    df = pd.read_pickle(pkl_file)
    # If clustering used PCA, you have to pass "embedding" as the pca-reduced features:
    features = np.vstack(df['embedding'].values)
    labels = df['label'].values

    # Calculate means
    normal_mean = features[labels == 0].mean(axis=0)
    anomaly_mean = features[labels == 1].mean(axis=0)

    print("Calculating distances to centroids...")
    df['dist_to_normal_centroid'] = cdist(features, [normal_mean]).flatten()
    df['dist_to_anomaly_centroid'] = cdist(features, [anomaly_mean]).flatten()
    df.to_pickle(output_pkl)
    print(f"Saved: {output_pkl}")

if __name__ == "__main__":
    add_centroid_scores('BGL_embeddings_hdbscan.pkl', 'BGL_clustered_centroids.pkl')
    add_centroid_scores('Thunderbird_embeddings_hdbscan.pkl', 'Thunderbird_clustered_centroids.pkl')