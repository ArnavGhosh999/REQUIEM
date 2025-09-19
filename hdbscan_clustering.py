import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import hdbscan
from tqdm import tqdm

# -- Recommended thread environment settings for best performance --
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

def fast_hdbscan_clustering(embedding_pkl, output_pkl,
                            n_components=50,
                            max_points=100_000,
                            min_cluster_size=75,
                            min_samples=50):
    print("Loading embeddings...")
    df = pd.read_pickle(embedding_pkl)
    X = np.vstack(list(tqdm(df['embedding'].values, desc="Embeddings")))
    
    print(f"Original shape: {X.shape}")
    print("Reducing dimensionality using PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA shape: {X_pca.shape}")

    if X_pca.shape[0] > max_points:
        print(f"Subsampling to {max_points} points for performance...")
        idx = np.random.choice(X_pca.shape[0], max_points, replace=False)
        X_pca = X_pca[idx]
        df = df.iloc[idx]
    
    print("Starting HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(X_pca)
    print("Clustering finished!")

    df['hdbscan_cluster'] = cluster_labels
    df.to_pickle(output_pkl)
    print(f"Saved clustered dataframe to: {output_pkl}")
    print(f"Clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
    print(f"Noise points: {(cluster_labels == -1).sum()}")
    return df

if __name__ == "__main__":
    fast_hdbscan_clustering('embeddings/BGL_embeddings.pkl', 'BGL_embeddings_hdbscan.pkl')
    fast_hdbscan_clustering('embeddings/Thunderbird_embeddings.pkl', 'Thunderbird_embeddings_hdbscan.pkl')
