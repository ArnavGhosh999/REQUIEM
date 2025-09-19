import matplotlib.pyplot as plt

def plot_full_pipeline_visuals(df, out_prefix):
    # 1. Confidence histogram
    plt.figure()
    plt.hist(df[df['label']==0]['anomaly_prob'], bins=50, alpha=0.6, label='Normal')
    plt.hist(df[df['label']==1]['anomaly_prob'], bins=50, alpha=0.6, label='Anomaly')
    plt.xlabel("Predicted Probability (Anomaly)")
    plt.title(f"Prediction Confidence Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_confidence_hist.jpg")
    plt.close()

    # 2. Distance to centroid hist
    plt.figure()
    plt.hist(df[df['label']==0]['dist_to_normal_centroid'], bins=50, alpha=0.6, label='Normal')
    plt.hist(df[df['label']==1]['dist_to_normal_centroid'], bins=50, alpha=0.6, label='Anomaly')
    plt.xlabel("Distance to Normal Centroid")
    plt.title(f"Distance to Normal Centroid")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_centroid_dist_hist.jpg")
    plt.close()

    # 3. ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(df['label'], df['anomaly_prob'])
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC: {auc(fpr, tpr):.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc_curve.jpg")
    plt.close()
    print(f"Visuals saved for {out_prefix}")

# Example:
plot_full_pipeline_visuals(bgl_results, "BGL_full")
plot_full_pipeline_visuals(thunder_results, "Thunderbird_full")
