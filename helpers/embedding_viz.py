import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

def plot_embeddings(pkl_path, title):
    df = pd.read_pickle(pkl_path)
    embeddings = np.vstack(df['embedding'].values)
    labels = np.array(df['label'].values)
    # Subsample for faster viz if >100,000
    if len(embeddings) > 100_000:
        idx = np.random.choice(len(embeddings), 100_000, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10,8))
    plt.scatter(emb2d[labels==0,0], emb2d[labels==0,1], alpha=0.3, label='Normal', s=3)
    plt.scatter(emb2d[labels==1,0], emb2d[labels==1,1], alpha=0.5, label='Anomaly', s=10, c='r')
    plt.title(title)
    plt.legend()
    plt.show()

# Usage:
plot_embeddings("embeddings/BGL_embeddings.pkl", "BGL Log Embedding Space")
plot_embeddings("embeddings/Thunderbird_embeddings.pkl", "Thunderbird Log Embedding Space")
