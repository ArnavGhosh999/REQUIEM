# REQUIEM: SLM-embedding based Log Anomaly Detection Pipeline


## Introduction

**REQUIEM** is a modular, extensible machine learning pipeline for log-based anomaly detection. Designed for reproducible research and robust performance, REQUIEM offers end-to-end capabilities, including transformer log embedding, clustering, centroid scoring, classification, cross-validation, and publication-quality metrics visualization. It allows detailed benchmarking on real-world datasets and is ready for both academic and industrial research collaborations.



## Data Used

- **BGL (BlueGene/L) Logs:**  
  Labeled high-performance computing (HPC) system logs. Includes imbalanced classes for anomaly detection and large-scale benchmarking.

- **Thunderbird Logs:**  
  Labeled system log data with realistic distributions, supporting both anomaly detection and generalizability analysis.


## Directory Structure
```
REQUIEM/
├── helpers/          # Core pipeline utilities: reporting, sampling, viz, etc.
├── metrics/          # Saved metric reports
├── visualizations/   # Embedding space visualizations
├── .gitignore
├── README.md
├── requirements.txt
├── classifier.py
├── centroid_scoring.py
├── ... (other pipeline modules)
```

## Pipeline Overview

1. **Preprocessing & Embedding**
    - Structured log lines and labels are processed into transformer embeddings.
    - Principal Component Analysis (PCA) reduces feature dimensionality.

2. **Clustering & Centroid Scoring**
    - Anomaly and normal clusters are identified and scored.
    - Distances to centroids become interpretable features for classification.

3. **Classification**
    - Logistic Regression classifier (toggle other models with minimal refactoring).
    - Both full log and subset-based training/testing.
    - Stratified k-fold cross-validation for robust benchmarking.

4. **Analysis & Visualization**
    - Generate confusion matrices, ROC curves, boxplots, and centroid histograms.
    - Save sampled hard cases for error analysis and detailed report appendices.

5. **Automated Reports**
    - Compiles results into PDFs or HTML, including all figures, tables, and sampled hard cases.

## Visualization and Reports

For every experiment, REQUIEM generates:

- **Confusion matrix heatmaps**
- **ROC/AUC curves**
- **F1 score boxplots**
- **Average F1/AUC tables**
- **t-SNE/PCA embedding visualizations**
- **Distance-to-centroid histograms**
- **Sampled hard cases reports**  

## Contributing

We welcome contributions for new features, bug fixes, and improved documentation!
- Issue proposals, feature requests, and bug reports
- Pull requests are reviewed for reproducibility and code quality  

**Contact:**  
For any inquiries, feedback, or interest in collaboration, please reach out via the repository’s Issues tab or email the maintainers.

**License:**
Apache License
