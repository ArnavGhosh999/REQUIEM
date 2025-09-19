import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def ml_benchmark_and_viz(pkl_file, dataset_name):
    df = pd.read_pickle(pkl_file)
    X = np.hstack([
        np.vstack(df['embedding'].values),
        df[['dist_to_normal_centroid','dist_to_anomaly_centroid']].values
    ])
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42, test_size=0.25)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    auc = roc_auc_score(y_test, clf.decision_function(X_test))
    print(f"[{dataset_name}] Logistic Regression:\n", classification_report(y_test, preds))
    print(f"AUC: {auc:.3f}")

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'])
    plt.title(f"{dataset_name} - Logistic Regression Confusion Matrix")
    plt.savefig(f"{dataset_name}_confusion_matrix.jpg")
    plt.close()

    # Centroid distance histogram (already generated, but here's the code)
    plt.figure(figsize=(8,6))
    plt.hist(df[df['label']==0]['dist_to_normal_centroid'], bins=50, alpha=0.5, label='Normal')
    plt.hist(df[df['label']==1]['dist_to_normal_centroid'], bins=50, alpha=0.5, label='Anomaly')
    plt.title(f"{dataset_name} - Distance to Normal Centroid")
    plt.legend()
    plt.savefig(f"{dataset_name}_distance_hist.jpg")
    plt.close()

    # Save hard cases for appendix
    hard_cases = df.sample(30, random_state=42)[['text','label','dist_to_normal_centroid','dist_to_anomaly_centroid']]  # adjust n as desired
    hard_cases.to_csv(f"{dataset_name}_hard_cases_sample.csv", index=False)

    # Save classification report to txt
    with open(f"{dataset_name}_classification_report.txt", "w") as f:
        f.write(str(classification_report(y_test, preds)))
        f.write(f"\nAUC: {auc:.3f}")

# Example usage
ml_benchmark_and_viz("BGL_clustered_centroids.pkl", "BGL")
ml_benchmark_and_viz("Thunderbird_clustered_centroids.pkl", "Thunderbird")