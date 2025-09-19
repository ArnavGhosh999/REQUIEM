from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import numpy as np
import pandas as pd

def run_incremental_pca(pkl_file, out_pca_file, nchunks=100, n_components=50):
    df = pd.read_pickle(pkl_file)
    embeddings = np.vstack(df['embedding'].values)
    n = embeddings.shape[0]
    batch_size = n // nchunks + 1
    ipca = IncrementalPCA(n_components=n_components)
    # 1st pass: fit
    for i in tqdm(range(0, n, batch_size), desc="PCA Fit Batches"):
        ipca.partial_fit(embeddings[i:i+batch_size])
    # 2nd pass: transform
    pca_vectors = []
    for i in tqdm(range(0, n, batch_size), desc="PCA Transform Batches"):
        pca_vectors.append(ipca.transform(embeddings[i:i+batch_size]))
    pca_vectors = np.vstack(pca_vectors)
    np.save(out_pca_file, pca_vectors)
    return pca_vectors

run_incremental_pca("BGL_embeddings (1).pkl", "BGL_embeddings_pca.npy")
run_incremental_pca("Thunderbird_embeddings (1).pkl", "Thunderbird_embeddings_pca.npy")
