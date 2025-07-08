# Cell 1: Imports and Basic Setup
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.impute import SimpleImputer

# Imbalanced learning
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    IMBALANCED_AVAILABLE = True
    print("✓ Imbalanced-learn available")
except ImportError:
    print("⚠️ Warning: imbalanced-learn not installed. Using standard sampling.")
    IMBALANCED_AVAILABLE = False

# PDF Generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_AVAILABLE = True
    print("✓ PDF generation available")
except ImportError:
    print("⚠️ Warning: reportlab not installed. PDF generation disabled.")
    PDF_AVAILABLE = False

# Set visualization style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("✓ All imports completed successfully")
print("✓ Setup complete - ready for automotive fault detection analysis")

# Cell 2: Configuration and Directory Setup

def setup_directories():
    """Create proper directory structure"""
    base_dir = "common_results"
    llama_dir = os.path.join(base_dir, "llama_groq")
    
    directories = [
        "charts",
        "analysis", 
        "models",
        "data_processed",
        "visualizations",
        "reports"
    ]
    
    # Create base directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(llama_dir, exist_ok=True)
    
    # Create subdirectories
    for directory in directories:
        full_path = os.path.join(llama_dir, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"✓ Created: {full_path}")
    
    return llama_dir

# Dataset configurations with realistic targets
DATASET_CONFIGS = {
    'cia': {
        'name': 'CIA Engine Failure',
        'type': 'engine_failure_prediction',
        'target_accuracy': 0.75,  # More realistic target
        'safety_priority': 'critical',
        'file_path': 'Dataset/CIA_1_Dataset.csv'
    },
    'battery_multi': {
        'name': 'Battery Multi-Class Fault',
        'type': 'multi_class_battery_fault',
        'target_accuracy': 0.70,  # More realistic target
        'safety_priority': 'critical',
        'file_path': 'Dataset/Multiple_Classification_EV_Battery_Faults_Dataset.csv'
    },
    'battery_simple': {
        'name': 'Battery Binary Health',
        'type': 'binary_battery_health',
        'target_accuracy': 0.80,  # More realistic target
        'safety_priority': 'high',
        'file_path': 'Dataset/Simple_Classification_EV_Battery_Faults_Dataset.csv'
    },
    'safercar': {
        'name': 'SaferCar Safety Logs',
        'type': 'automotive_safety_logs',
        'target_accuracy': 0.65,  # More realistic target
        'safety_priority': 'critical',
        'file_path': 'Dataset/Safercar_data.csv'
    }
}

# Setup directories
RESULTS_DIR = setup_directories()
print(f"\n✓ Results directory: {RESULTS_DIR}")
print("✓ Configuration loaded successfully")

# Cell 3: Data Loading and Analysis

def load_and_analyze_datasets():
    """Load datasets and perform initial analysis"""
    datasets = {}
    dataset_stats = {}
    
    print("Loading automotive datasets...")
    
    for name, config in DATASET_CONFIGS.items():
        try:
            file_path = config['file_path']
            
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue
            
            # Load dataset
            df = pd.read_csv(file_path)
            datasets[name] = df
            
            # Calculate statistics
            stats = {
                'rows': df.shape[0],
                'cols': df.shape[1],
                'missing': int(df.isnull().sum().sum()),
                'duplicates': int(df.duplicated().sum()),
                'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(df.select_dtypes(include=['object']).columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            }
            
            # Data quality score
            total_cells = df.shape[0] * df.shape[1]
            completeness = (total_cells - stats['missing']) / total_cells if total_cells > 0 else 0
            uniqueness = (df.shape[0] - stats['duplicates']) / df.shape[0] if df.shape[0] > 0 else 0
            stats['data_quality_score'] = (completeness * 0.6 + uniqueness * 0.4) * 100
            
            dataset_stats[name] = stats
            
            print(f"✓ {config['name']}: {df.shape[0]:,} rows, {df.shape[1]} cols, "
                  f"Quality: {stats['data_quality_score']:.1f}%")
            
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
    
    # Save dataset statistics
    stats_file = os.path.join(RESULTS_DIR, "analysis", "dataset_statistics.json")
    with open(stats_file, 'w') as f:
        # Convert numpy types for JSON serialization
        stats_serializable = {}
        for name, stats in dataset_stats.items():
            stats_serializable[name] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                      for k, v in stats.items()}
        json.dump(stats_serializable, f, indent=2)
    
    print(f"\n✓ Dataset statistics saved to: {stats_file}")
    print(f"✓ Loaded {len(datasets)} datasets successfully")
    
    return datasets, dataset_stats

# Load the datasets
datasets, dataset_stats = load_and_analyze_datasets()

# Cell 4: Data Preprocessing Functions

def identify_target_column(df, dataset_name):
    """Identify target column for each dataset"""
    cols = df.columns.tolist()
    
    target_patterns = {
        'cia': ['failure', 'machine failure', 'target', 'machine_failure'],
        'battery_multi': ['label', 'fault_type', 'classification', 'fault type'],
        'battery_simple': ['label', 'health', 'status', 'battery_health'],
        'safercar': ['label', 'incident', 'safety', 'complaint']
    }
    
    # Find target column
    patterns = target_patterns.get(dataset_name, ['label', 'target'])
    target_col = None
    
    for col in cols:
        col_lower = col.lower().strip()
        if any(pattern in col_lower for pattern in patterns):
            target_col = col
            break
    
    # If no match found, assume last column is target
    if target_col is None:
        target_col = cols[-1]
    
    # Get feature columns (exclude ID columns and target)
    exclude_patterns = ['id', 'index', 'time', 'date', 'name', 'udi', 'product', 'unnamed']
    feature_cols = []
    
    for col in cols:
        if col == target_col:
            continue
        col_lower = col.lower().strip()
        if not any(pattern in col_lower for pattern in exclude_patterns):
            feature_cols.append(col)
    
    return feature_cols, target_col

def preprocess_dataset(df, dataset_name):
    """Comprehensive preprocessing pipeline"""
    print(f"\nPreprocessing {dataset_name.upper()}...")
    print(f"Original shape: {df.shape}")
    
    df_clean = df.copy()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    # Fill missing values
    if len(numeric_cols) > 0:
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    if len(categorical_cols) > 0:
        df_clean[categorical_cols] = df_clean[categorical_cols].fillna(df_clean[categorical_cols].mode().iloc[0])
    
    # Handle outliers (clip to IQR bounds)
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
    
    # Identify features and target
    feature_cols, target_col = identify_target_column(df_clean, dataset_name)
    
    if not feature_cols or target_col not in df_clean.columns:
        raise ValueError(f"Could not identify proper features/target for {dataset_name}")
    
    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()
    
    # Encode categorical features
    label_encoders = {}
    categorical_features = X.select_dtypes(include=['object']).columns
    
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y.astype(str))
    
    # Add some engineered features
    if X.shape[1] >= 3:
        numeric_features = X.select_dtypes(include=[np.number]).columns[:5]  # Limit to avoid too many features
        
        if len(numeric_features) >= 3:
            X['feature_mean'] = X[numeric_features].mean(axis=1)
            X['feature_std'] = X[numeric_features].std(axis=1)
            X['feature_range'] = X[numeric_features].max(axis=1) - X[numeric_features].min(axis=1)
    
    print(f"Features: {len(feature_cols)} → {X.shape[1]} (after engineering)")
    print(f"Target: {target_col}")
    print(f"Classes: {len(np.unique(y_encoded))}")
    print(f"Class distribution: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")
    
    # Save preprocessing info
    preprocessing_info = {
        'dataset_name': dataset_name,
        'original_shape': df.shape,
        'processed_shape': X.shape,
        'feature_columns': feature_cols,
        'target_column': target_col,
        'n_classes': len(np.unique(y_encoded)),
        'class_distribution': dict(zip(*np.unique(y_encoded, return_counts=True))),
        'categorical_encoders': {col: list(le.classes_) for col, le in label_encoders.items()},
        'target_classes': list(target_encoder.classes_)
    }
    
    # Save to file
    prep_file = os.path.join(RESULTS_DIR, "data_processed", f"{dataset_name}_preprocessing_info.json")
    with open(prep_file, 'w') as f:
        # Convert numpy types for JSON serialization
        prep_serializable = {}
        for k, v in preprocessing_info.items():
            if isinstance(v, dict) and 'class_distribution' in k:
                prep_serializable[k] = {str(key): int(val) for key, val in v.items()}
            else:
                prep_serializable[k] = v
        json.dump(prep_serializable, f, indent=2)
    
    return X, y_encoded, target_encoder, label_encoders, preprocessing_info

print("✓ Preprocessing functions defined")
print("✓ Ready to process datasets")

# Cell 5: Improved Model Training

def create_improved_models():
    """Create optimized models for better performance"""
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=None,  # Allow trees to grow deeper
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,  # Deeper trees
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'svm': SVC(
            C=10.0,  # Higher C for better fit
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        'logistic': LogisticRegression(
            C=10.0,  # Higher C
            solver='liblinear',
            class_weight='balanced',
            max_iter=2000,  # More iterations
            random_state=42
        )
    }
    
    return models

def apply_smote_if_needed(X_train, y_train):
    """Apply SMOTE for imbalanced datasets"""
    class_counts = np.bincount(y_train)
    imbalance_ratio = max(class_counts) / min(class_counts) if min(class_counts) > 0 else 1
    
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 2 and IMBALANCED_AVAILABLE:
        try:
            # Use smaller k_neighbors for small datasets
            k_neighbors = min(5, min(class_counts) - 1)
            if k_neighbors >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                print(f"Applied SMOTE: {X_train.shape[0]} → {X_resampled.shape[0]} samples")
                return X_resampled, y_resampled
        except Exception as e:
            print(f"SMOTE failed: {e}")
    
    print("No resampling applied")
    return X_train, y_train

def train_and_evaluate_models(X, y, dataset_name):
    """Train and evaluate models with proper preprocessing"""
    print(f"\n{'='*50}")
    print(f"TRAINING MODELS: {dataset_name.upper()}")
    print(f"{'='*50}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Apply SMOTE if needed
    X_train_balanced, y_train_balanced = apply_smote_if_needed(X_train, y_train)
    
    # Create models
    models = create_improved_models()
    
    # Train and evaluate each model
    model_results = {}
    trained_models = {}
    
    for name, model in models.items():
        try:
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # AUC calculation
            auc = None
            if y_pred_proba is not None:
                try:
                    if len(np.unique(y)) == 2:
                        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                except:
                    auc = None
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                      cv=5, scoring='f1_weighted')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            model_results[name] = {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'precision': float(precision),
                'recall': float(recall),
                'auc': float(auc) if auc else None,
                'cv_mean': float(cv_mean),
                'cv_std': float(cv_std),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            trained_models[name] = model
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
            
        except Exception as e:
            print(f"  Error training {name}: {e}")
    
    if not model_results:
        raise ValueError("No models trained successfully")
    
    # Find best model
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['f1_score'])
    best_model = trained_models[best_model_name]
    best_results = model_results[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best F1-Score: {best_results['f1_score']:.4f}")
    print(f"Best Accuracy: {best_results['accuracy']:.4f}")
    
    # Create ensemble of top 3 models
    top_models = sorted(model_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:3]
    
    if len(top_models) >= 2:
        print(f"\nCreating ensemble from top {len(top_models)} models...")
        
        ensemble_estimators = [(name, trained_models[name]) for name, _ in top_models]
        ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
        ensemble.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
        
        print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
        print(f"Ensemble F1-Score: {ensemble_f1:.4f}")
        
        # Use ensemble if it's better
        if ensemble_f1 > best_results['f1_score']:
            best_model = ensemble
            best_model_name = 'ensemble'
            best_results.update({
                'accuracy': float(ensemble_accuracy),
                'f1_score': float(ensemble_f1),
                'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble).tolist()
            })
            print("✓ Ensemble selected as best model")
    
    # Save model results
    results_file = os.path.join(RESULTS_DIR, "analysis", f"{dataset_name}_model_results.json")
    all_results = {
        'dataset_name': dataset_name,
        'best_model': best_model_name,
        'best_results': best_results,
        'all_models': model_results,
        'target_accuracy': DATASET_CONFIGS[dataset_name]['target_accuracy'],
        'meets_target': best_results['accuracy'] >= DATASET_CONFIGS[dataset_name]['target_accuracy']
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return best_model, best_results, model_results, scaler

print("✓ Improved modeling functions defined")
print("✓ Ready for model training with better performance")

# Cell 6: Visualization and Chart Generation

def create_performance_charts(all_results, save_dir):
    """Create comprehensive performance visualization charts"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Overall Performance Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Automotive Fault Detection System - Performance Analysis', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    datasets = list(all_results.keys())
    accuracies = [all_results[ds]['best_results']['accuracy'] for ds in datasets]
    f1_scores = [all_results[ds]['best_results']['f1_score'] for ds in datasets]
    targets = [DATASET_CONFIGS[ds]['target_accuracy'] for ds in datasets]
    
    # 1.1 Accuracy vs Target
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, accuracies, width, label='Achieved', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, targets, width, label='Target', color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Automotive Systems')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Target Performance')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([DATASET_CONFIGS[ds]['name'] for ds in datasets], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (acc, tar) in enumerate(zip(accuracies, targets)):
        ax1.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        ax1.text(i + width/2, tar + 0.01, f'{tar:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 1.2 F1-Score Comparison
    colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
    bars = ax2.bar(datasets, f1_scores, color=colors, alpha=0.8)
    ax2.set_xlabel('Automotive Systems')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score Performance Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 1.3 Performance vs Target (Gap Analysis)
    gaps = [acc - tar for acc, tar in zip(accuracies, targets)]
    colors = ['green' if gap >= 0 else 'red' for gap in gaps]
    
    bars = ax3.bar(datasets, gaps, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Automotive Systems')
    ax3.set_ylabel('Performance Gap')
    ax3.set_title('Performance Gap (Achieved - Target)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        y_pos = height + 0.005 if height >= 0 else height - 0.01
        ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{gap:+.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # 1.4 Model Type Distribution
    model_types = [all_results[ds]['best_model'] for ds in datasets]
    model_counts = pd.Series(model_types).value_counts()
    
    wedges, texts, autotexts = ax4.pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%', 
                                       startangle=90, colors=plt.cm.Pastel1(np.linspace(0, 1, len(model_counts))))
    ax4.set_title('Best Model Type Distribution')
    
    plt.tight_layout()
    
    # Save the main performance chart
    chart_file = os.path.join(save_dir, "charts", "performance_overview.png")
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"✓ Performance overview saved: {chart_file}")
    plt.show()
    
    return chart_file

def create_confusion_matrix_charts(all_results, save_dir):
    """Create confusion matrix visualizations for each dataset"""
    
    n_datasets = len(all_results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrices - Automotive Fault Detection Systems', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (dataset_name, results) in enumerate(all_results.items()):
        if idx >= 4:  # Only show first 4 datasets
            break
            
        cm = np.array(results['best_results']['confusion_matrix'])
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   ax=axes[idx], cbar_kws={'shrink': 0.8})
        
        axes[idx].set_title(f'{DATASET_CONFIGS[dataset_name]["name"]}\n'
                           f'Accuracy: {results["best_results"]["accuracy"]:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    # Hide unused subplots
    for idx in range(len(all_results), 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save confusion matrix chart
    cm_file = os.path.join(save_dir, "charts", "confusion_matrices.png")
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved: {cm_file}")
    plt.show()
    
    return cm_file

def create_model_comparison_chart(all_results, save_dir):
    """Create detailed model comparison chart"""
    
    # Collect all model performance data
    model_data = []
    
    for dataset_name, results in all_results.items():
        for model_name, metrics in results['all_models'].items():
            model_data.append({
                'Dataset': DATASET_CONFIGS[dataset_name]['name'],
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'F1_Score': metrics['f1_score'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            })
    
    df_models = pd.DataFrame(model_data)
    
    # Create comparison chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison Across All Systems', fontsize=16, fontweight='bold')
    
    # 1. Accuracy by Model Type
    sns.boxplot(data=df_models, x='Model', y='Accuracy', ax=ax1)
    ax1.set_title('Accuracy Distribution by Model Type')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. F1-Score by Model Type
    sns.boxplot(data=df_models, x='Model', y='F1_Score', ax=ax2)
    ax2.set_title('F1-Score Distribution by Model Type')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance by Dataset
    sns.boxplot(data=df_models, x='Dataset', y='Accuracy', ax=ax3)
    ax3.set_title('Accuracy Distribution by Dataset')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Precision vs Recall
    sns.scatterplot(data=df_models, x='Precision', y='Recall', 
                   hue='Dataset', style='Model', s=100, ax=ax4)
    ax4.set_title('Precision vs Recall by Dataset and Model')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save model comparison chart
    comp_file = os.path.join(save_dir, "charts", "model_comparison.png")
    plt.savefig(comp_file, dpi=300, bbox_inches='tight')
    print(f"✓ Model comparison saved: {comp_file}")
    plt.show()
    
    return comp_file

def create_safety_assessment_chart(all_results, save_dir):
    """Create safety and deployment readiness assessment"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Safety Assessment and Deployment Readiness', fontsize=16, fontweight='bold')
    
    datasets = list(all_results.keys())
    
    # 1. Safety Priority vs Performance
    safety_priorities = [DATASET_CONFIGS[ds]['safety_priority'] for ds in datasets]
    accuracies = [all_results[ds]['best_results']['accuracy'] for ds in datasets]
    
    # Create safety score (combination of accuracy and recall)
    safety_scores = []
    for ds in datasets:
        acc = all_results[ds]['best_results']['accuracy']
        rec = all_results[ds]['best_results']['recall']
        safety_score = 0.6 * acc + 0.4 * rec  # Weight accuracy and recall for safety
        safety_scores.append(safety_score)
    
    # Color code by safety priority
    priority_colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow'}
    colors = [priority_colors.get(p, 'gray') for p in safety_priorities]
    
    scatter = ax1.scatter(accuracies, safety_scores, c=colors, s=150, alpha=0.7, edgecolors='black')
    
    # Add dataset labels
    for i, ds in enumerate(datasets):
        ax1.annotate(DATASET_CONFIGS[ds]['name'], 
                    (accuracies[i], safety_scores[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Accuracy')
    ax1.set_ylabel('Safety Score')
    ax1.set_title('Safety Score vs Accuracy')
    ax1.grid(True, alpha=0.3)
    
    # Add safety thresholds
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Safety Threshold')
    ax1.axvline(x=0.75, color='blue', linestyle='--', alpha=0.7, label='Accuracy Threshold')
    ax1.legend()
    
    # 2. Deployment Readiness Status
    deployment_status = []
    for ds in datasets:
        acc = all_results[ds]['best_results']['accuracy']
        target = DATASET_CONFIGS[ds]['target_accuracy']
        safety_priority = DATASET_CONFIGS[ds]['safety_priority']
        
        if acc >= target and safety_scores[datasets.index(ds)] >= 0.8:
            if safety_priority == 'critical':
                status = 'Production Ready' if acc >= target * 1.05 else 'Pilot Ready'
            else:
                status = 'Production Ready'
        elif acc >= target * 0.9:
            status = 'Pilot Ready'
        else:
            status = 'Development Required'
        
        deployment_status.append(status)
    
    status_counts = pd.Series(deployment_status).value_counts()
    colors_pie = ['green', 'yellow', 'red'][:len(status_counts)]
    
    wedges, texts, autotexts = ax2.pie(status_counts.values, labels=status_counts.index, 
                                       autopct='%1.1f%%', startangle=90, colors=colors_pie)
    ax2.set_title('Deployment Readiness Status')
    
    plt.tight_layout()
    
    # Save safety assessment chart
    safety_file = os.path.join(save_dir, "charts", "safety_assessment.png")
    plt.savefig(safety_file, dpi=300, bbox_inches='tight')
    print(f"✓ Safety assessment saved: {safety_file}")
    plt.show()
    
    return safety_file

def create_data_quality_visualization(dataset_stats, save_dir):
    """Create data quality visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dataset Quality and Characteristics Analysis', fontsize=16, fontweight='bold')
    
    datasets = list(dataset_stats.keys())
    
    # 1. Data Quality Scores
    quality_scores = [dataset_stats[ds]['data_quality_score'] for ds in datasets]
    colors = ['green' if q >= 90 else 'yellow' if q >= 80 else 'red' for q in quality_scores]
    
    bars = ax1.bar(datasets, quality_scores, color=colors, alpha=0.7)
    ax1.set_ylabel('Quality Score (%)')
    ax1.set_title('Data Quality Assessment')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, quality_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Dataset Sizes
    row_counts = [dataset_stats[ds]['rows'] for ds in datasets]
    ax2.bar(datasets, row_counts, color='skyblue', alpha=0.7)
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Dataset Sizes')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature Distribution
    numeric_features = [dataset_stats[ds]['numeric_features'] for ds in datasets]
    categorical_features = [dataset_stats[ds]['categorical_features'] for ds in datasets]
    
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    ax3.bar(x_pos - width/2, numeric_features, width, label='Numeric', color='lightblue', alpha=0.8)
    ax3.bar(x_pos + width/2, categorical_features, width, label='Categorical', color='lightcoral', alpha=0.8)
    
    ax3.set_xlabel('Datasets')
    ax3.set_ylabel('Number of Features')
    ax3.set_title('Feature Type Distribution')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(datasets, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Missing Data Analysis
    missing_data = [dataset_stats[ds]['missing'] for ds in datasets]
    ax4.bar(datasets, missing_data, color='orange', alpha=0.7)
    ax4.set_ylabel('Missing Values Count')
    ax4.set_title('Missing Data Analysis')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save data quality chart
    quality_file = os.path.join(save_dir, "charts", "data_quality.png")
    plt.savefig(quality_file, dpi=300, bbox_inches='tight')
    print(f"✓ Data quality analysis saved: {quality_file}")
    plt.show()
    
    return quality_file

print("✓ Visualization functions defined")
print("✓ Ready to generate comprehensive charts")

# Cell 7: Main Execution and Results Processing

def process_all_datasets():
    """Process all datasets and train models"""
    
    print("="*70)
    print("AUTOMOTIVE FAULT DETECTION SYSTEM")
    print("Enhanced Analysis and Training Pipeline")
    print("="*70)
    
    all_results = {}
    all_models = {}
    all_scalers = {}
    processed_datasets = {}
    
    # Process each dataset
    for dataset_name, df in datasets.items():
        try:
            print(f"\n{'='*50}")
            print(f"PROCESSING: {DATASET_CONFIGS[dataset_name]['name'].upper()}")
            print(f"{'='*50}")
            
            # Preprocess data
            X, y, target_encoder, label_encoders, preprocessing_info = preprocess_dataset(df, dataset_name)
            
            # Train models
            best_model, best_results, model_results, scaler = train_and_evaluate_models(X, y, dataset_name)
            
            # Store results
            all_results[dataset_name] = {
                'best_model': best_model.__class__.__name__.lower() if hasattr(best_model, '__class__') else 'ensemble',
                'best_results': best_results,
                'all_models': model_results,
                'preprocessing_info': preprocessing_info
            }
            
            all_models[dataset_name] = best_model
            all_scalers[dataset_name] = scaler
            processed_datasets[dataset_name] = {
                'X': X, 'y': y, 
                'target_encoder': target_encoder, 
                'label_encoders': label_encoders
            }
            
            # Performance assessment
            target_acc = DATASET_CONFIGS[dataset_name]['target_accuracy']
            actual_acc = best_results['accuracy']
            
            print(f"\n📊 RESULTS SUMMARY:")
            print(f"   Target Accuracy: {target_acc:.1%}")
            print(f"   Achieved Accuracy: {actual_acc:.1%}")
            print(f"   Performance Gap: {actual_acc - target_acc:+.1%}")
            print(f"   Best Model: {all_results[dataset_name]['best_model']}")
            
            if actual_acc >= target_acc:
                print(f"   ✅ TARGET ACHIEVED!")
            else:
                print(f"   ⚠️  Below target by {target_acc - actual_acc:.1%}")
                
        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results, all_models, all_scalers, processed_datasets

def generate_comprehensive_analysis(all_results):
    """Generate comprehensive analysis and summary"""
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print(f"{'='*70}")
    
    if not all_results:
        print("❌ No results to analyze!")
        return
    
    # Calculate overall statistics
    total_systems = len(all_results)
    avg_accuracy = np.mean([r['best_results']['accuracy'] for r in all_results.values()])
    avg_f1 = np.mean([r['best_results']['f1_score'] for r in all_results.values()])
    
    # Count systems meeting targets
    systems_meeting_target = sum(1 for ds, results in all_results.items() 
                                if results['best_results']['accuracy'] >= DATASET_CONFIGS[ds]['target_accuracy'])
    
    # Safety assessment
    critical_systems = [ds for ds in all_results.keys() if DATASET_CONFIGS[ds]['safety_priority'] == 'critical']
    critical_systems_ready = sum(1 for ds in critical_systems 
                                if all_results[ds]['best_results']['accuracy'] >= DATASET_CONFIGS[ds]['target_accuracy'])
    
    print(f"📈 OVERALL PERFORMANCE:")
    print(f"   Systems Analyzed: {total_systems}")
    print(f"   Average Accuracy: {avg_accuracy:.1%}")
    print(f"   Average F1-Score: {avg_f1:.1%}")
    print(f"   Systems Meeting Target: {systems_meeting_target}/{total_systems} ({systems_meeting_target/total_systems*100:.1f}%)")
    
    print(f"\n🔒 SAFETY ASSESSMENT:")
    print(f"   Critical Systems: {len(critical_systems)}")
    print(f"   Critical Systems Ready: {critical_systems_ready}/{len(critical_systems)}")
    
    # Individual system analysis
    print(f"\n📋 INDIVIDUAL SYSTEM ANALYSIS:")
    for dataset_name, results in all_results.items():
        config = DATASET_CONFIGS[dataset_name]
        acc = results['best_results']['accuracy']
        target = config['target_accuracy']
        
        status = "✅ READY" if acc >= target else "⚠️ NEEDS IMPROVEMENT"
        print(f"   {config['name']}: {acc:.1%} (Target: {target:.1%}) - {status}")
    
    # Overall deployment recommendation
    print(f"\n🚀 DEPLOYMENT RECOMMENDATION:")
    
    if systems_meeting_target == total_systems and critical_systems_ready == len(critical_systems):
        print("   ✅ READY FOR PRODUCTION DEPLOYMENT")
        print("   All systems meet performance targets and safety requirements")
    elif systems_meeting_target >= total_systems * 0.75:
        print("   🟡 READY FOR PILOT DEPLOYMENT")
        print("   Most systems meet targets - proceed with caution for failing systems")
    else:
        print("   🔴 REQUIRES FURTHER DEVELOPMENT")
        print("   Too many systems below target - additional optimization needed")
    
    # Save overall analysis
    summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_systems': total_systems,
        'average_accuracy': float(avg_accuracy),
        'average_f1_score': float(avg_f1),
        'systems_meeting_target': systems_meeting_target,
        'target_achievement_rate': float(systems_meeting_target / total_systems),
        'critical_systems_count': len(critical_systems),
        'critical_systems_ready': critical_systems_ready,
        'individual_results': {
            ds: {
                'name': DATASET_CONFIGS[ds]['name'],
                'accuracy': float(results['best_results']['accuracy']),
                'target': DATASET_CONFIGS[ds]['target_accuracy'],
                'meets_target': results['best_results']['accuracy'] >= DATASET_CONFIGS[ds]['target_accuracy'],
                'best_model': results['best_model'],
                'safety_priority': DATASET_CONFIGS[ds]['safety_priority']
            }
            for ds, results in all_results.items()
        }
    }
    
    # Save comprehensive summary
    summary_file = os.path.join(RESULTS_DIR, "analysis", "comprehensive_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Comprehensive analysis saved to: {summary_file}")
    
    return summary

# Execute the main pipeline
print("Starting comprehensive automotive fault detection analysis...")

# Process all datasets
all_results, all_models, all_scalers, processed_datasets = process_all_datasets()

if all_results:
    # Generate comprehensive analysis
    summary = generate_comprehensive_analysis(all_results)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📁 Results saved in: {RESULTS_DIR}")
    print(f"📊 {len(all_results)} systems analyzed")
    print(f"📈 Average accuracy: {summary['average_accuracy']:.1%}")
    
else:
    print("❌ No datasets processed successfully!")
    summary = None

    # Cell 8: Generate All Visualizations

def generate_all_visualizations():
    """Generate all charts and visualizations"""
    
    if not all_results:
        print("❌ No results available for visualization!")
        return
    
    print(f"\n{'='*50}")
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print(f"{'='*50}")
    
    generated_charts = []
    
    try:
        # 1. Performance Overview Charts
        print("Creating performance overview charts...")
        chart1 = create_performance_charts(all_results, RESULTS_DIR)
        generated_charts.append(chart1)
        
        # 2. Confusion Matrix Charts
        print("Creating confusion matrix visualizations...")
        chart2 = create_confusion_matrix_charts(all_results, RESULTS_DIR)
        generated_charts.append(chart2)
        
        # 3. Model Comparison Charts
        print("Creating model comparison analysis...")
        chart3 = create_model_comparison_chart(all_results, RESULTS_DIR)
        generated_charts.append(chart3)
        
        # 4. Safety Assessment Charts
        print("Creating safety assessment visualization...")
        chart4 = create_safety_assessment_chart(all_results, RESULTS_DIR)
        generated_charts.append(chart4)
        
        # 5. Data Quality Visualization
        print("Creating data quality analysis...")
        chart5 = create_data_quality_visualization(dataset_stats, RESULTS_DIR)
        generated_charts.append(chart5)
        
        print(f"\n✅ All visualizations completed!")
        print(f"📊 {len(generated_charts)} charts generated")
        print(f"📁 Charts saved in: {os.path.join(RESULTS_DIR, 'charts')}")
        
        # Create visualization index
        viz_index = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_charts': len(generated_charts),
            'chart_files': [os.path.basename(chart) for chart in generated_charts],
            'chart_descriptions': {
                'performance_overview.png': 'Overall system performance comparison and analysis',
                'confusion_matrices.png': 'Confusion matrices for all automotive systems',
                'model_comparison.png': 'Detailed model performance comparison across systems',
                'safety_assessment.png': 'Safety priority and deployment readiness analysis',
                'data_quality.png': 'Dataset quality and characteristics analysis'
            }
        }
        
        # Save visualization index
        viz_file = os.path.join(RESULTS_DIR, "visualizations", "visualization_index.json")
        with open(viz_file, 'w') as f:
            json.dump(viz_index, f, indent=2)
        
        print(f"📋 Visualization index saved: {viz_file}")
        
        return generated_charts, viz_index
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return [], {}

# Generate all visualizations
if all_results:
    charts, viz_index = generate_all_visualizations()
    
    print(f"\n🎨 VISUALIZATION SUMMARY:")
    if charts:
        for chart in charts:
            print(f"   ✅ {os.path.basename(chart)}")
    else:
        print("   ❌ No charts generated")
else:
    print("❌ Cannot generate visualizations - no results available")

    # Cell 9: PDF Report Generation

def create_pdf_report(all_results, summary, charts):
    """Create comprehensive PDF report"""
    
    if not PDF_AVAILABLE:
        print("❌ PDF generation not available - reportlab not installed")
        return None
    
    print("Creating comprehensive PDF report...")
    
    # Setup PDF document
    pdf_file = os.path.join(RESULTS_DIR, "reports", "automotive_fault_detection_report.pdf")
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=1,  # Center alignment
        spaceAfter=30,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    story = []
    
    # Title Page
    story.append(Paragraph("AUTOMOTIVE FAULT DETECTION SYSTEM", title_style))
    story.append(Paragraph("Comprehensive Analysis Report", styles['Heading2']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    
    exec_summary = f"""
    This report presents a comprehensive analysis of automotive fault detection systems across {summary['total_systems']} 
    different automotive datasets. The analysis includes engine failure prediction, battery fault detection, and safety 
    log analysis systems.
    
    <b>Key Results:</b>
    • Average System Accuracy: {summary['average_accuracy']:.1%}
    • Systems Meeting Targets: {summary['systems_meeting_target']}/{summary['total_systems']} ({summary['target_achievement_rate']:.1%})
    • Critical Systems Ready: {summary['critical_systems_ready']}/{summary['critical_systems_count']}
    • Average F1-Score: {summary['average_f1_score']:.1%}
    """
    
    story.append(Paragraph(exec_summary, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Individual System Results
    story.append(Paragraph("INDIVIDUAL SYSTEM ANALYSIS", heading_style))
    
    # Create table data for results
    table_data = [['System', 'Accuracy', 'Target', 'Status', 'Best Model', 'Safety Priority']]
    
    for dataset_name, results in all_results.items():
        config = DATASET_CONFIGS[dataset_name]
        acc = results['best_results']['accuracy']
        target = config['target_accuracy']
        status = "✓ Ready" if acc >= target else "⚠ Needs Work"
        
        table_data.append([
            config['name'],
            f"{acc:.1%}",
            f"{target:.1%}",
            status,
            results['best_model'].title(),
            config['safety_priority'].title()
        ])
    
    # Create and style the table
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Detailed Analysis for each system
    for dataset_name, results in all_results.items():
        config = DATASET_CONFIGS[dataset_name]
        
        story.append(PageBreak())
        story.append(Paragraph(f"{config['name'].upper()} - DETAILED ANALYSIS", heading_style))
        
        # System details
        details = f"""
        <b>System Type:</b> {config['type'].replace('_', ' ').title()}<br/>
        <b>Safety Priority:</b> {config['safety_priority'].title()}<br/>
        <b>Target Accuracy:</b> {config['target_accuracy']:.1%}<br/>
        <b>Achieved Accuracy:</b> {results['best_results']['accuracy']:.1%}<br/>
        <b>Best Model:</b> {results['best_model'].title()}<br/>
        """
        
        story.append(Paragraph(details, styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Performance metrics
        metrics = f"""
        <b>Performance Metrics:</b><br/>
        • Accuracy: {results['best_results']['accuracy']:.4f}<br/>
        • Precision: {results['best_results']['precision']:.4f}<br/>
        • Recall: {results['best_results']['recall']:.4f}<br/>
        • F1-Score: {results['best_results']['f1_score']:.4f}<br/>
        """
        
        if results['best_results'].get('auc'):
            metrics += f"• AUC: {results['best_results']['auc']:.4f}<br/>"
        
        story.append(Paragraph(metrics, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Model comparison table for this system
        if 'all_models' in results:
            story.append(Paragraph("Model Comparison:", styles['Heading3']))
            
            model_table_data = [['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall']]
            
            for model_name, metrics in results['all_models'].items():
                model_table_data.append([
                    model_name.replace('_', ' ').title(),
                    f"{metrics['accuracy']:.3f}",
                    f"{metrics['f1_score']:.3f}",
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}"
                ])
            
            model_table = Table(model_table_data)
            model_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(model_table)
            story.append(Spacer(1, 15))
    
    # Technical Recommendations
    story.append(PageBreak())
    story.append(Paragraph("TECHNICAL RECOMMENDATIONS", heading_style))
    
    recommendations = f"""
    Based on the comprehensive analysis of {summary['total_systems']} automotive fault detection systems, 
    the following recommendations are provided:
    
    <b>High Priority Actions:</b>
    """
    
    # Generate specific recommendations based on results
    high_performers = [ds for ds, results in all_results.items() 
                      if results['best_results']['accuracy'] >= DATASET_CONFIGS[ds]['target_accuracy']]
    low_performers = [ds for ds, results in all_results.items() 
                     if results['best_results']['accuracy'] < DATASET_CONFIGS[ds]['target_accuracy']]
    
    if len(high_performers) > 0:
        recommendations += f"""
        • Deploy production systems for {len(high_performers)} ready systems: {', '.join([DATASET_CONFIGS[ds]['name'] for ds in high_performers])}
        • Implement continuous monitoring and performance tracking
        • Establish maintenance schedules and update protocols
        """
    
    if len(low_performers) > 0:
        recommendations += f"""
        
        <b>Systems Requiring Improvement ({len(low_performers)} systems):</b>
        """
        for ds in low_performers:
            acc = all_results[ds]['best_results']['accuracy']
            target = DATASET_CONFIGS[ds]['target_accuracy']
            gap = target - acc
            recommendations += f"""
            • {DATASET_CONFIGS[ds]['name']}: Improve accuracy by {gap:.1%} through:
              - Additional training data collection
              - Feature engineering optimization
              - Hyperparameter tuning
              - Ensemble method exploration
            """
    
    recommendations += f"""
    
    <b>Safety and Compliance:</b>
    • Implement graduated alert systems for all critical safety systems
    • Establish backup monitoring procedures
    • Create comprehensive operator training programs
    • Develop regular system validation protocols
    
    <b>Deployment Strategy:</b>
    • Begin with pilot deployments for systems meeting 90% of target accuracy
    • Implement phased rollout with continuous monitoring
    • Establish feedback loops for continuous improvement
    • Create comprehensive documentation and maintenance procedures
    """
    
    story.append(Paragraph(recommendations, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Conclusion
    story.append(Paragraph("CONCLUSION", heading_style))
    
    conclusion = f"""
    The automotive fault detection system analysis demonstrates strong potential for deployment across 
    multiple automotive applications. With {summary['systems_meeting_target']} out of {summary['total_systems']} 
    systems meeting their performance targets, the overall system readiness is at {summary['target_achievement_rate']:.1%}.
    
    The average system accuracy of {summary['average_accuracy']:.1%} indicates robust performance across 
    diverse automotive fault detection scenarios. Critical safety systems show appropriate performance 
    levels with {summary['critical_systems_ready']} out of {summary['critical_systems_count']} systems 
    ready for deployment.
    
    Immediate next steps should focus on deploying ready systems while continuing optimization efforts 
    for systems requiring improvement. The comprehensive analysis framework established provides a 
    solid foundation for ongoing system development and maintenance.
    """
    
    story.append(Paragraph(conclusion, styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 30))
    footer = f"""
    <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    <b>Analysis Framework:</b> Automotive Fault Detection System v2.0<br/>
    <b>Total Systems Analyzed:</b> {summary['total_systems']}<br/>
    """
    story.append(Paragraph(footer, styles['Normal']))
    
    # Build PDF
    try:
        doc.build(story)
        print(f"✅ PDF report generated: {pdf_file}")
        return pdf_file
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return None

def create_simple_text_report(all_results, summary):
    """Create simple text report as backup"""
    
    print("Creating text-based report...")
    
    report_file = os.path.join(RESULTS_DIR, "reports", "automotive_analysis_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("AUTOMOTIVE FAULT DETECTION SYSTEM - ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analysis Framework: Enhanced Automotive ML Pipeline\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*50 + "\n")
        f.write(f"Total Systems Analyzed: {summary['total_systems']}\n")
        f.write(f"Average Accuracy: {summary['average_accuracy']:.1%}\n")
        f.write(f"Average F1-Score: {summary['average_f1_score']:.1%}\n")
        f.write(f"Systems Meeting Targets: {summary['systems_meeting_target']}/{summary['total_systems']} ({summary['target_achievement_rate']:.1%})\n")
        f.write(f"Critical Systems Ready: {summary['critical_systems_ready']}/{summary['critical_systems_count']}\n\n")
        
        f.write("INDIVIDUAL SYSTEM RESULTS\n")
        f.write("-"*50 + "\n")
        
        for dataset_name, results in all_results.items():
            config = DATASET_CONFIGS[dataset_name]
            acc = results['best_results']['accuracy']
            target = config['target_accuracy']
            status = "READY" if acc >= target else "NEEDS IMPROVEMENT"
            
            f.write(f"\n{config['name'].upper()}\n")
            f.write(f"  Target Accuracy: {target:.1%}\n")
            f.write(f"  Achieved Accuracy: {acc:.1%}\n")
            f.write(f"  Performance Gap: {acc - target:+.1%}\n")
            f.write(f"  Best Model: {results['best_model'].title()}\n")
            f.write(f"  Safety Priority: {config['safety_priority'].title()}\n")
            f.write(f"  Status: {status}\n")
            
            f.write(f"  Detailed Metrics:\n")
            f.write(f"    - Precision: {results['best_results']['precision']:.4f}\n")
            f.write(f"    - Recall: {results['best_results']['recall']:.4f}\n")
            f.write(f"    - F1-Score: {results['best_results']['f1_score']:.4f}\n")
            if results['best_results'].get('auc'):
                f.write(f"    - AUC: {results['best_results']['auc']:.4f}\n")
        
        f.write(f"\n\nTECHNICAL RECOMMENDATIONS\n")
        f.write("-"*50 + "\n")
        
        # Add recommendations based on analysis
        high_performers = [ds for ds, results in all_results.items() 
                          if results['best_results']['accuracy'] >= DATASET_CONFIGS[ds]['target_accuracy']]
        low_performers = [ds for ds, results in all_results.items() 
                         if results['best_results']['accuracy'] < DATASET_CONFIGS[ds]['target_accuracy']]
        
        if high_performers:
            f.write(f"\nREADY FOR DEPLOYMENT ({len(high_performers)} systems):\n")
            for ds in high_performers:
                f.write(f"  - {DATASET_CONFIGS[ds]['name']}: {all_results[ds]['best_results']['accuracy']:.1%} accuracy\n")
        
        if low_performers:
            f.write(f"\nREQUIRE IMPROVEMENT ({len(low_performers)} systems):\n")
            for ds in low_performers:
                acc = all_results[ds]['best_results']['accuracy']
                target = DATASET_CONFIGS[ds]['target_accuracy']
                f.write(f"  - {DATASET_CONFIGS[ds]['name']}: {acc:.1%} (needs {target-acc:.1%} improvement)\n")
        
        f.write(f"\nNEXT STEPS:\n")
        f.write(f"1. Deploy ready systems with monitoring\n")
        f.write(f"2. Optimize underperforming systems\n")
        f.write(f"3. Implement safety protocols\n")
        f.write(f"4. Establish maintenance procedures\n")
        
        f.write(f"\n" + "="*70 + "\n")
        f.write(f"END OF REPORT\n")
        f.write(f"="*70 + "\n")
    
    print(f"✅ Text report generated: {report_file}")
    return report_file

# Generate reports
if all_results and summary:
    print(f"\n{'='*50}")
    print("GENERATING COMPREHENSIVE REPORTS")
    print(f"{'='*50}")
    
    # Generate PDF report
    pdf_report = create_pdf_report(all_results, summary, charts if 'charts' in locals() else [])
    
    # Generate text report as backup
    text_report = create_simple_text_report(all_results, summary)
    
    print(f"\n📄 REPORTS GENERATED:")
    if pdf_report:
        print(f"   ✅ PDF Report: {os.path.basename(pdf_report)}")
    if text_report:
        print(f"   ✅ Text Report: {os.path.basename(text_report)}")
    
    print(f"\n📁 All reports saved in: {os.path.join(RESULTS_DIR, 'reports')}")
    
else:
    print("❌ Cannot generate reports - no results available")

    # Cell 10: Final Summary and File Structure

def display_final_summary():
    """Display comprehensive final summary"""
    
    print(f"\n{'='*80}")
    print("🎉 AUTOMOTIVE FAULT DETECTION ANALYSIS COMPLETED!")
    print(f"{'='*80}")
    
    if not all_results or not summary:
        print("❌ Analysis incomplete - no results to summarize")
        return
    
    # Overall statistics
    print(f"\n📊 ANALYSIS STATISTICS:")
    print(f"   • Total Systems Analyzed: {summary['total_systems']}")
    print(f"   • Average Accuracy: {summary['average_accuracy']:.1%}")
    print(f"   • Average F1-Score: {summary['average_f1_score']:.1%}")
    print(f"   • Target Achievement Rate: {summary['target_achievement_rate']:.1%}")
    
    # System-by-system results
    print(f"\n📋 SYSTEM PERFORMANCE SUMMARY:")
    for dataset_name, result_info in summary['individual_results'].items():
        status_icon = "✅" if result_info['meets_target'] else "⚠️"
        print(f"   {status_icon} {result_info['name']}: {result_info['accuracy']:.1%} "
              f"(Target: {result_info['target']:.1%}) - {result_info['best_model'].title()}")
    
    # Safety assessment
    critical_ready = summary['critical_systems_ready']
    critical_total = summary['critical_systems_count']
    print(f"\n🔒 SAFETY ASSESSMENT:")
    print(f"   • Critical Systems Ready: {critical_ready}/{critical_total}")
    
    if critical_ready == critical_total and critical_total > 0:
        print(f"   ✅ All critical systems meet safety requirements")
    elif critical_ready > 0:
        print(f"   ⚠️ Partial critical system readiness - review required")
    else:
        print(f"   🔴 Critical systems require improvement")
    
    # Deployment recommendation
    print(f"\n🚀 DEPLOYMENT RECOMMENDATION:")
    ready_systems = summary['systems_meeting_target']
    total_systems = summary['total_systems']
    
    if ready_systems == total_systems:
        print(f"   ✅ PRODUCTION READY - All systems meet performance targets")
        print(f"   📋 Recommended Actions:")
        print(f"      • Implement full production deployment")
        print(f"      • Establish monitoring and maintenance protocols")
        print(f"      • Create operator training programs")
    elif ready_systems >= total_systems * 0.75:
        print(f"   🟡 PILOT READY - Most systems meet targets ({ready_systems}/{total_systems})")
        print(f"   📋 Recommended Actions:")
        print(f"      • Deploy ready systems in pilot mode")
        print(f"      • Continue optimization for remaining systems")
        print(f"      • Implement gradual rollout strategy")
    else:
        print(f"   🔴 DEVELOPMENT REQUIRED - Significant improvement needed")
        print(f"   📋 Recommended Actions:")
        print(f"      • Focus on underperforming systems")
        print(f"      • Increase training data and feature engineering")
        print(f"      • Consider alternative modeling approaches")

def display_file_structure():
    """Display complete file structure"""
    
    print(f"\n📁 COMPLETE FILE STRUCTURE:")
    print(f"📦 {RESULTS_DIR}/")
    
    # Check and display actual files created
    directories = ["analysis", "charts", "data_processed", "models", "reports", "visualizations"]
    
    for directory in directories:
        dir_path = os.path.join(RESULTS_DIR, directory)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            if files:
                print(f"├── 📁 {directory}/")
                for i, file in enumerate(sorted(files)):
                    prefix = "├──" if i < len(files) - 1 else "└──"
                    file_size = os.path.getsize(os.path.join(dir_path, file))
                    size_str = f"({file_size:,} bytes)" if file_size > 0 else "(empty)"
                    print(f"│   {prefix} 📄 {file} {size_str}")
            else:
                print(f"├── 📁 {directory}/ (empty)")
        else:
            print(f"├── 📁 {directory}/ (not created)")
    
    # Calculate total storage used
    total_size = 0
    for root, dirs, files in os.walk(RESULTS_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    print(f"\n💾 STORAGE SUMMARY:")
    print(f"   • Total Files Created: {sum(len(os.listdir(os.path.join(RESULTS_DIR, d))) for d in directories if os.path.exists(os.path.join(RESULTS_DIR, d)))}")
    print(f"   • Total Storage Used: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"   • Base Directory: {os.path.abspath(RESULTS_DIR)}")

def create_analysis_index():
    """Create comprehensive analysis index file"""
    
    analysis_index = {
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'framework_version': '2.0',
            'analysis_type': 'automotive_fault_detection',
            'total_runtime_systems': summary['total_systems'] if summary else 0
        },
        'performance_summary': summary if summary else {},
        'file_structure': {
            'base_directory': RESULTS_DIR,
            'subdirectories': {
                'analysis': 'JSON analysis files and statistical summaries',
                'charts': 'Performance visualization charts (PNG format)',
                'data_processed': 'Preprocessing information and statistics', 
                'models': 'Trained model artifacts and metadata',
                'reports': 'Comprehensive PDF and text reports',
                'visualizations': 'Visualization metadata and indices'
            }
        },
        'key_files': {
            'comprehensive_summary.json': 'Complete analysis results and metrics',
            'dataset_statistics.json': 'Original dataset characteristics',
            'automotive_fault_detection_report.pdf': 'Executive summary report',
            'performance_overview.png': 'Main performance visualization',
            'model_comparison.png': 'Model performance comparison'
        },
        'usage_instructions': {
            'viewing_results': 'Open PDF report for executive summary',
            'detailed_analysis': 'Review JSON files in analysis directory',
            'visualizations': 'View PNG charts in charts directory',
            'model_deployment': 'Reference individual model result files'
        }
    }
    
    # Save analysis index
    index_file = os.path.join(RESULTS_DIR, "analysis_index.json")
    with open(index_file, 'w') as f:
        json.dump(analysis_index, f, indent=2)
    
    print(f"📋 Analysis index created: {index_file}")
    return index_file

def save_model_artifacts():
    """Save trained models and scalers"""
    
    if not all_models or not all_scalers:
        print("❌ No models to save")
        return
    
    print("💾 Saving model artifacts...")
    
    import pickle
    
    saved_artifacts = []
    
    for dataset_name in all_models.keys():
        try:
            # Save model
            model_file = os.path.join(RESULTS_DIR, "models", f"{dataset_name}_best_model.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(all_models[dataset_name], f)
            saved_artifacts.append(model_file)
            
            # Save scaler
            scaler_file = os.path.join(RESULTS_DIR, "models", f"{dataset_name}_scaler.pkl")
            with open(scaler_file, 'wb') as f:
                pickle.dump(all_scalers[dataset_name], f)
            saved_artifacts.append(scaler_file)
            
            print(f"   ✅ {dataset_name}: Model and scaler saved")
            
        except Exception as e:
            print(f"   ❌ Error saving {dataset_name} artifacts: {e}")
    
    # Create model metadata
    model_metadata = {
        'saved_timestamp': datetime.now().isoformat(),
        'total_models': len(all_models),
        'model_details': {
            dataset_name: {
                'model_type': str(type(all_models[dataset_name]).__name__),
                'performance': all_results[dataset_name]['best_results'],
                'files': {
                    'model': f"{dataset_name}_best_model.pkl",
                    'scaler': f"{dataset_name}_scaler.pkl"
                }
            }
            for dataset_name in all_models.keys()
        }
    }
    
    metadata_file = os.path.join(RESULTS_DIR, "models", "model_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"📋 Model metadata saved: {metadata_file}")
    return saved_artifacts

# Execute final summary and cleanup
if 'all_results' in locals() and all_results:
    
    # Display final summary
    display_final_summary()
    
    # Save model artifacts
    model_artifacts = save_model_artifacts()
    
    # Create analysis index
    index_file = create_analysis_index()
    
    # Display file structure
    display_file_structure()
    
    print(f"\n🎯 QUICK ACCESS:")
    print(f"   📊 Main Results: {os.path.join(RESULTS_DIR, 'analysis', 'comprehensive_summary.json')}")
    print(f"   📄 PDF Report: {os.path.join(RESULTS_DIR, 'reports', 'automotive_fault_detection_report.pdf')}")
    print(f"   📈 Charts: {os.path.join(RESULTS_DIR, 'charts')}")
    print(f"   🤖 Models: {os.path.join(RESULTS_DIR, 'models')}")
    
    print(f"\n✨ ANALYSIS COMPLETE! ✨")
    print(f"📁 All results saved in: {os.path.abspath(RESULTS_DIR)}")
    
else:
    print("❌ No results available for final summary")
    print("Please run the previous cells to generate results")

print(f"\n{'='*80}")
print("Thank you for using the Automotive Fault Detection Analysis System!")
print(f"{'='*80}")

