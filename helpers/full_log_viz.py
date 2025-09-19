import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def full_log_visualization(results_pkl, dataset_name):
    df = pd.read_pickle(results_pkl)
    y_true = df['label'].values
    y_pred = df['anomaly_pred'].values
    y_prob = df['anomaly_prob'].values

    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(f"{dataset_name}_full_classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title(f"{dataset_name} Full Log Confusion Matrix")
    plt.xlabel("Pred label")
    plt.ylabel("True label")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center', color='red')
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_full_confmat.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC: {auc(fpr, tpr):.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} Full Log ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_full_roc.png")
    plt.close()

    # Probability Histogram
    plt.figure()
    plt.hist(y_prob[y_true==0], bins=50, alpha=0.6, label='Normal')
    plt.hist(y_prob[y_true==1], bins=50, alpha=0.6, label='Anomaly')
    plt.xlabel("Predicted Probability (Anomaly)")
    plt.title(f"{dataset_name} Prediction Confidence Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_full_conf_hist.png")
    plt.close()
    print(f"Visuals saved for {dataset_name}")

# Example usage:
full_log_visualization("BGL_full_results.pkl", "BGL_full")
full_log_visualization("Thunderbird_full_results.pkl", "Thunderbird_full")
