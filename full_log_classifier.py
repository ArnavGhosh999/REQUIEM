from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pandas as pd
import numpy as np

def fit_and_predict(pkl_file, pca_file, out_res_file, sample_size=100_000, batch_size=100_000):
    df = pd.read_pickle(pkl_file)
    pca = np.load(pca_file)
    y = df['label'].values
    idx = np.random.choice(len(df), min(sample_size, len(df)), replace=False)
    X_sample = pca[idx]
    y_sample = y[idx]
    mean_norm = X_sample[y_sample==0].mean(axis=0)
    mean_anom = X_sample[y_sample==1].mean(axis=0)
    d_norm = np.linalg.norm(pca - mean_norm, axis=1)
    d_anom = np.linalg.norm(pca - mean_anom, axis=1)
    X_train = np.hstack([X_sample, d_norm[idx][:,None], d_anom[idx][:,None]])
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_sample)
    # Predict in batches
    preds = np.zeros(len(df))
    probs = np.zeros(len(df))
    for i in tqdm(range(0, len(df), batch_size), desc="ML Predict Batches"):
        X_batch = np.hstack([
            pca[i:i+batch_size],
            d_norm[i:i+batch_size,None],
            d_anom[i:i+batch_size,None]
        ])
        preds[i:i+batch_size] = clf.predict(X_batch)
        probs[i:i+batch_size] = clf.predict_proba(X_batch)[:,1]
    df['dist_to_normal_centroid'] = d_norm
    df['dist_to_anomaly_centroid'] = d_anom
    df['anomaly_prob'] = probs
    df['anomaly_pred'] = preds
    df.to_pickle(out_res_file)
    print(f"Saved predictions to {out_res_file}")
    return df

fit_and_predict("BGL_embeddings (1).pkl", "BGL_embeddings_pca.npy", "BGL_full_results.pkl")
fit_and_predict("Thunderbird_embeddings (1).pkl", "Thunderbird_embeddings_pca.npy", "Thunderbird_full_results.pkl")