import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, roc_curve, confusion_matrix
from tqdm import tqdm

def full_cv_and_visuals(pkl_file, pca_file, dataset_name, n_splits=10, n_cv_points=100_000):
    df = pd.read_pickle(pkl_file)
    X_full = np.load(pca_file)
    y_full = df['label'].values

    # Stratified (balanced) subsample for time/memory
    idx = np.random.choice(len(df), min(n_cv_points, len(df)), replace=False)
    X = X_full[idx]
    y = y_full[idx]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1s, aucs, precisions, recalls = [], [], [], []
    all_probs = np.zeros(n_cv_points)
    all_preds = np.zeros(n_cv_points)

    for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc="CV Folds")):
        # Centroids per fold
        mean_norm = X[train_idx][y[train_idx]==0].mean(axis=0)
        mean_anom = X[train_idx][y[train_idx]==1].mean(axis=0)
        d_train_norm = np.linalg.norm(X[train_idx] - mean_norm, axis=1)
        d_train_anom = np.linalg.norm(X[train_idx] - mean_anom, axis=1)
        d_test_norm = np.linalg.norm(X[test_idx] - mean_norm, axis=1)
        d_test_anom = np.linalg.norm(X[test_idx] - mean_anom, axis=1)
        X_train = np.hstack([X[train_idx], d_train_norm[:, None], d_train_anom[:, None]])
        X_test = np.hstack([X[test_idx], d_test_norm[:, None], d_test_anom[:, None]])
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train, y[train_idx])
        probs = clf.predict_proba(X_test)[:,1]
        preds = clf.predict(X_test)
        f1s.append(f1_score(y[test_idx], preds))
        aucs.append(roc_auc_score(y[test_idx], probs))
        precisions.append(precision_score(y[test_idx], preds))
        recalls.append(recall_score(y[test_idx], preds))
        all_probs[test_idx] = probs
        all_preds[test_idx] = preds

    # Save metrics for report
    np.savetxt(f"{dataset_name}_cv_f1s.txt", f1s, fmt="%.4f")
    np.savetxt(f"{dataset_name}_cv_aucs.txt", aucs, fmt="%.4f")

    # F1/AUC boxplot
    plt.figure()
    plt.boxplot([f1s, aucs], labels=['F1 (anomaly)', 'AUC'])
    plt.title(f"{dataset_name} CV Scores per Fold")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_cv_boxplot.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label='CV Out-of-Fold')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{dataset_name} ROC (CV OOF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_cv_roc.png")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y, all_preds)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title(f"{dataset_name} Confusion Matrix (CV OOF)")
    plt.xlabel("Pred label")
    plt.ylabel("True label")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center', color='red')
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_cv_confmat.png")
    plt.close()

    # Out-of-fold probability hist
    plt.figure()
    plt.hist(all_probs[y==0], bins=50, alpha=0.5, label="Normal")
    plt.hist(all_probs[y==1], bins=50, alpha=0.5, label="Anomaly")
    plt.xlabel("CV OOF Predicted Probability (Anomaly)")
    plt.ylabel("Count")
    plt.title(f"{dataset_name} Out-of-Fold Probabilities")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_cv_oof_probs.png")
    plt.close()

    return f1s, aucs

# Example usage (for both):
bgl_f1s, bgl_aucs = full_cv_and_visuals("BGL_embeddings (1).pkl", "BGL_embeddings_pca.npy", "BGL_full")
thunder_f1s, thunder_aucs = full_cv_and_visuals("Thunderbird_embeddings (1).pkl", "Thunderbird_embeddings_pca.npy", "Thunderbird_full")