import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score

def cv_visualization_report(pkl_file, dataset_name, n_splits=10):
    df = pd.read_pickle(pkl_file)
    X_base = np.vstack(df['embedding'].values)
    y = df['label'].values

    f1s, aucs = [], []
    probas = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_base, y), 1):
        pca = PCA(n_components=50, random_state=42)
        X_train = pca.fit_transform(X_base[train_idx])
        X_test = pca.transform(X_base[test_idx])
        mean_norm = X_train[y[train_idx]==0].mean(axis=0)
        mean_anom = X_train[y[train_idx]==1].mean(axis=0)
        d_train_norm = np.linalg.norm(X_train-mean_norm, axis=1)[:,None]
        d_train_anom = np.linalg.norm(X_train-mean_anom, axis=1)[:,None]
        d_test_norm = np.linalg.norm(X_test-mean_norm, axis=1)[:,None]
        d_test_anom = np.linalg.norm(X_test-mean_anom, axis=1)[:,None]
        Xtr = np.hstack([X_train, d_train_norm, d_train_anom])
        Xts = np.hstack([X_test, d_test_norm, d_test_anom])
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(Xtr, y[train_idx])
        preds = clf.predict(Xts)
        fold_f1 = f1_score(y[test_idx], preds)
        fold_auc = roc_auc_score(y[test_idx], clf.decision_function(Xts))
        f1s.append(fold_f1)
        aucs.append(fold_auc)
        probas[test_idx] = clf.predict_proba(Xts)[:,1]

    # Boxplot
    plt.figure()
    plt.boxplot([f1s, aucs], labels=['F1 (Anomaly)', 'AUC'])
    plt.title(f"{dataset_name} Cross-Validation Scores")
    plt.ylabel("Score")
    plt.ylim(0.5, 1.01)
    plt.grid(True, axis='y')
    plt.savefig(f"{dataset_name}_cv_boxplot.png")
    plt.close()

    plt.figure()
    plt.violinplot([f1s, aucs], showmeans=True)
    plt.xticks([1,2], ['F1 (Anomaly)', 'AUC'])
    plt.ylabel("Score")
    plt.title(f"{dataset_name} Cross-Validation Violin Plot")
    plt.savefig(f"{dataset_name}_cv_violin.png")
    plt.close()

    plt.figure()
    plt.hist(probas[y == 0], bins=50, alpha=0.5, label="Normal")
    plt.hist(probas[y == 1], bins=50, alpha=0.5, label="Anomaly")
    plt.xlabel("Out-Of-Fold Predicted Probability (Anomaly)")
    plt.title(f"{dataset_name} OOF Probabilities")
    plt.legend()
    plt.savefig(f"{dataset_name}_cv_oof_probs.png")
    plt.close()

    # Save per-fold stats for the PDF
    np.savetxt(f"{dataset_name}_cv_f1s.txt", f1s, fmt="%.4f")
    np.savetxt(f"{dataset_name}_cv_aucs.txt", aucs, fmt="%.4f")

# Example usage:
cv_visualization_report("saved/BGL_clustered_centroids.pkl", "BGL", n_splits=10)
cv_visualization_report("saved/Thunderbird_clustered_centroids.pkl", "Thunderbird", n_splits=10)
