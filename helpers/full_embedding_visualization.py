import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(pkl_file, out_img="embedding_space.png", n_points=10000):
    df = pd.read_pickle(pkl_file)
    n = min(n_points, len(df))
    idx = np.random.choice(len(df), n, replace=False)
    X = np.vstack(df.iloc[idx]['embedding'].values)
    y = df.iloc[idx]['label'].values
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X)
    plt.figure(figsize=(8, 7))
    plt.scatter(X_2d[y==0,0], X_2d[y==0,1], s=8, c="dodgerblue", label="Normal", alpha=0.4)
    plt.scatter(X_2d[y==1,0], X_2d[y==1,1], s=12, c="orangered", label="Anomaly", alpha=0.7)
    plt.legend()
    plt.title("Log Embedding Space")
    plt.tight_layout()
    plt.savefig(out_img, dpi=120)
    plt.close()
    print(f"Saved: {out_img}")

visualize_embeddings("BGL_embeddings (1).pkl", out_img="BGL_embedding_space.png", n_points=10000)
visualize_embeddings("Thunderbird_embeddings (1).pkl", out_img="Thunderbird_embedding_space.png", n_points=10000)