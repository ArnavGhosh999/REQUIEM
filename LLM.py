import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import requests
import json
import warnings
warnings.filterwarnings('ignore')

class QLORAFineTunedFaultDetector:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.models = {}
        self.feature_selectors = {}
        self.data_stats = {}
        self.ollama_url = "http://localhost:11434/api/generate"
        self.training_data = []
        
    def collect_training_data(self, context, response):
        """Collect data for QLora fine-tuning"""
        training_example = {
            "instruction": "Analyze fault detection results and provide technical insights",
            "input": context,
            "output": response
        }
        self.training_data.append(training_example)
    
    def query_dolphin_mistral_structured(self, prompt, expected_format="analysis"):
        """Enhanced query with structured prompts to reduce hallucination"""
        
        structured_prompt = f"""
You are a technical expert in industrial fault detection systems. Respond with ONLY factual, data-driven analysis.

CONSTRAINTS:
- Base responses ONLY on provided data
- Use specific numbers from the analysis
- Do not invent metrics or statistics
- Be concise and technical
- Focus on actionable recommendations

TASK: {prompt}

FORMAT: Provide analysis in {expected_format} format with specific data points.
"""
        
        payload = {
            "model": "dolphin-mistral",
            "prompt": structured_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Reduce creativity/hallucination
                "top_p": 0.8,
                "top_k": 20,
                "repeat_penalty": 1.2
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()['response']
                self.collect_training_data(prompt, result)
                return result
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Connection error: {str(e)}"
    
    def load_datasets(self):
        datasets = {}
        datasets['cia'] = pd.read_csv('Dataset/CIA_1_Dataset.csv')
        datasets['engine'] = pd.read_csv('Dataset/engine_fault_detection_dataset.csv')
        datasets['battery_multi'] = pd.read_csv('Dataset/Multiple_Classification_EV_Battery_Faults_Dataset.csv')
        datasets['battery_simple'] = pd.read_csv('Dataset/Simple_Classification_EV_Battery_Faults_Dataset.csv')
        
        print("Dataset Statistics:")
        for name, df in datasets.items():
            print(f"{name}: Shape={df.shape}, Columns={list(df.columns)}")
            self.data_stats[name] = {
                'rows': df.shape[0],
                'cols': df.shape[1],
                'missing': df.isnull().sum().sum(),
                'duplicates': df.duplicated().sum()
            }
        
        return datasets
    
    def advanced_preprocessing_small_dataset(self, df, dataset_type):
        """Specialized preprocessing for small datasets like CIA"""
        df_clean = df.copy()
        
        # Handle missing values more conservatively for small datasets
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isnull().sum() > 0:
                # Use median for small datasets to avoid outlier influence
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Conservative outlier handling for small datasets
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[col].quantile(0.15)  # More conservative quartiles
            Q3 = df_clean[col].quantile(0.85)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.0 * IQR  # Less aggressive outlier removal
            upper_bound = Q3 + 2.0 * IQR
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        
        return df_clean
    
    def create_features_conservative(self, X, dataset_type):
        """Conservative feature engineering for small datasets"""
        X_enhanced = X.copy()
        
        # Only create essential features for small datasets
        if X_enhanced.shape[1] >= 2 and X_enhanced.shape[0] > 100:
            # Create only the most important interaction features
            feature_importance_pairs = [(0, 1), (0, 2), (1, 2)] if X_enhanced.shape[1] >= 3 else [(0, 1)]
            
            for i, j in feature_importance_pairs[:3]:  # Limit to top 3 interactions
                if j < X_enhanced.shape[1]:
                    col_i = X_enhanced.columns[i] if hasattr(X_enhanced, 'columns') else f'feature_{i}'
                    col_j = X_enhanced.columns[j] if hasattr(X_enhanced, 'columns') else f'feature_{j}'
                    
                    # Only add ratio if denominator is not close to zero
                    denominator = X_enhanced.iloc[:, j] if hasattr(X_enhanced, 'iloc') else X_enhanced[:, j]
                    if np.abs(denominator).min() > 1e-6:
                        numerator = X_enhanced.iloc[:, i] if hasattr(X_enhanced, 'iloc') else X_enhanced[:, i]
                        X_enhanced[f'ratio_{i}_{j}'] = numerator / denominator
        
        return X_enhanced
    
    def preprocess_cia_data_optimized(self, df):
        """Optimized preprocessing specifically for CIA dataset"""
        print(f"CIA Dataset - Original shape: {df.shape}")
        
        df_clean = self.advanced_preprocessing_small_dataset(df, 'cia')
        
        # Identify feature and target columns
        available_cols = list(df_clean.columns)
        feature_cols = []
        for col in available_cols:
            if col.lower() not in ['udi', 'product id', 'type', 'failure type'] and 'id' not in col.lower():
                feature_cols.append(col)
        
        target_col = None
        for col in available_cols:
            if 'failure' in col.lower() and 'type' in col.lower():
                target_col = col
                break
        
        if target_col is None:
            target_col = available_cols[-1]
        
        print(f"CIA Features: {feature_cols}")
        print(f"CIA Target: {target_col}")
        
        X = df_clean[feature_cols].copy()
        y = df_clean[target_col].copy()
        
        # Check class distribution
        print(f"CIA Class distribution: {y.value_counts().to_dict()}")
        
        # Conservative feature engineering for small dataset
        X_enhanced = self.create_features_conservative(X, 'cia')
        print(f"CIA Enhanced shape: {X_enhanced.shape}")
        
        # Encode target
        if target_col not in self.encoders:
            self.encoders[target_col] = LabelEncoder()
            y_encoded = self.encoders[target_col].fit_transform(y)
        else:
            y_encoded = self.encoders[target_col].transform(y)
        
        # Feature selection - be more conservative with small datasets
        max_features = min(10, X_enhanced.shape[1], X_enhanced.shape[0] // 10)
        if max_features < X_enhanced.shape[1]:
            selector = SelectKBest(f_classif, k=max_features)
            X_selected = selector.fit_transform(X_enhanced, y_encoded)
            self.feature_selectors['cia'] = selector
        else:
            X_selected = X_enhanced.values
            self.feature_selectors['cia'] = None
        
        # Use PowerTransformer for better normalization
        if 'cia' not in self.scalers:
            self.scalers['cia'] = PowerTransformer(method='yeo-johnson', standardize=True)
            X_scaled = self.scalers['cia'].fit_transform(X_selected)
        else:
            X_scaled = self.scalers['cia'].transform(X_selected)
        
        print(f"CIA Final shape: {X_scaled.shape}")
        return X_scaled, y_encoded
    
    def preprocess_other_datasets(self, df, dataset_type):
        """Standard preprocessing for larger datasets"""
        df_clean = df.copy()
        
        # Handle missing values
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        # Identify columns
        available_cols = list(df_clean.columns)
        feature_cols = []
        target_col = None
        
        if dataset_type == 'engine':
            for col in available_cols:
                if 'condition' not in col.lower() and 'label' not in col.lower():
                    feature_cols.append(col)
            for col in available_cols:
                if 'condition' in col.lower() or 'label' in col.lower():
                    target_col = col
                    break
        else:  # battery datasets
            for col in available_cols:
                if 'label' not in col.lower() and 'target' not in col.lower():
                    feature_cols.append(col)
            for col in available_cols:
                if 'label' in col.lower() or 'target' in col.lower():
                    target_col = col
                    break
        
        if target_col is None:
            target_col = available_cols[-1]
        
        X = df_clean[feature_cols].copy()
        y = df_clean[target_col].copy()
        
        # Enhanced feature engineering for larger datasets
        X_enhanced = X.copy()
        if X.shape[0] > 1000:  # Only for larger datasets
            for i in range(min(3, X.shape[1])):
                for j in range(i+1, min(5, X.shape[1])):
                    if np.abs(X.iloc[:, j]).min() > 1e-6:
                        X_enhanced[f'ratio_{i}_{j}'] = X.iloc[:, i] / X.iloc[:, j]
        
        # Encode target
        if target_col not in self.encoders:
            self.encoders[target_col] = LabelEncoder()
            y_encoded = self.encoders[target_col].fit_transform(y)
        else:
            y_encoded = self.encoders[target_col].transform(y)
        
        # Feature selection
        max_features = min(20, X_enhanced.shape[1])
        selector = SelectKBest(f_classif, k=max_features)
        X_selected = selector.fit_transform(X_enhanced, y_encoded)
        self.feature_selectors[dataset_type] = selector
        
        # Scaling
        if dataset_type not in self.scalers:
            self.scalers[dataset_type] = RobustScaler()
            X_scaled = self.scalers[dataset_type].fit_transform(X_selected)
        else:
            X_scaled = self.scalers[dataset_type].transform(X_selected)
        
        return X_scaled, y_encoded
    
    def train_optimized_classifier(self, X, y, model_name):
        """Optimized classifier training with fixed LogisticRegression"""
        print(f"\nTraining {model_name} - Data shape: {X.shape}, Classes: {len(np.unique(y))}")
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance based on dataset size
        if len(y_train) < 1000:  # Small dataset like CIA
            # Use ADASYN for small datasets
            try:
                sampler = ADASYN(random_state=42, n_neighbors=min(5, len(y_train)//10))
                X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
                print(f"Applied ADASYN: {X_train.shape} -> {X_train_balanced.shape}")
            except:
                X_train_balanced, y_train_balanced = X_train, y_train
                print("No resampling applied (insufficient data)")
        else:
            # Use SMOTETomek for larger datasets
            try:
                sampler = SMOTETomek(random_state=42)
                X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
                print(f"Applied SMOTETomek: {X_train.shape} -> {X_train_balanced.shape}")
            except:
                X_train_balanced, y_train_balanced = X_train, y_train
        
        # Define models with fixed parameters
        models = {
            'rf': RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                random_state=42, 
                class_weight='balanced',
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=42,
                max_depth=6
            ),
            'et': ExtraTreesClassifier(
                n_estimators=150,
                random_state=42,
                class_weight='balanced',
                max_depth=8
            ),
            'lr': LogisticRegression(
                random_state=42, 
                class_weight='balanced', 
                max_iter=2000,  # Increased iterations
                solver='liblinear',  # Better solver for small datasets
                C=1.0
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=min(5, len(y_train_balanced)//10),
                weights='distance'
            ),
            'nb': GaussianNB(),
            'dt': DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced',
                max_depth=8,
                min_samples_split=5
            )
        }
        
        # Train individual models and get their performance
        trained_models = {}
        model_scores = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train_balanced, y_train_balanced)
                y_pred = model.predict(X_test)
                score = f1_score(y_test, y_pred, average='weighted')
                trained_models[name] = model
                model_scores[name] = score
                print(f"{name.upper()} F1-Score: {score:.4f}")
            except Exception as e:
                print(f"Failed to train {name}: {e}")
        
        # Select top 3 performing models for ensemble
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        ensemble_models = [(name, trained_models[name]) for name, score in top_models]
        
        print(f"Ensemble models: {[name for name, _ in ensemble_models]}")
        
        # Create ensemble
        ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble.fit(X_train_balanced, y_train_balanced)
        
        # Predictions
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test) if hasattr(ensemble, 'predict_proba') else None
        
        self.models[model_name] = ensemble
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=min(5, len(np.unique(y_train_balanced))), shuffle=True, random_state=42)
        cv_scores = cross_val_score(ensemble, X_train_balanced, y_train_balanced, cv=cv, scoring='f1_weighted')
        
        return {
            'model': ensemble,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'ensemble_models': [name for name, _ in ensemble_models],
            'individual_scores': model_scores
        }
    
    def analyze_with_factual_llm(self, dataset_name, metrics, data_stats):
        """Factual analysis with specific data points to prevent hallucination"""
        
        context = f"""
FACTUAL DATA for {dataset_name}:
- Dataset size: {data_stats['rows']} samples, {data_stats['cols']} features
- Model accuracy: {metrics['accuracy']:.4f}
- F1-score: {metrics['f1_score']:.4f}
- Cross-validation: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}
- Ensemble components: {metrics['ensemble_models']}
"""
        
        prompt = f"""
Based on the specific data provided above for {dataset_name}:

1. Assess if {metrics['accuracy']:.4f} accuracy is acceptable for industrial fault detection
2. Evaluate the reliability based on CV score {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}
3. Recommend specific improvements for a {data_stats['rows']}-sample dataset
4. Suggest monitoring frequency based on the current performance metrics

Use ONLY the provided numbers. Do not estimate or assume additional statistics.
"""
        
        return self.query_dolphin_mistral_structured(prompt, "technical_assessment")
    
    def generate_qlora_training_data(self):
        """Generate training data for QLora fine-tuning"""
        if len(self.training_data) > 0:
            qlora_data = {
                "training_examples": self.training_data,
                "total_examples": len(self.training_data),
                "domain": "industrial_fault_detection"
            }
            
            with open('qlora_training_data.json', 'w') as f:
                json.dump(qlora_data, f, indent=2)
            
            print(f"Generated {len(self.training_data)} training examples for QLora fine-tuning")
            print("Saved to: qlora_training_data.json")
        
        return self.training_data
    
    def generate_comprehensive_report(self, datasets):
        print("=== QLORA-OPTIMIZED FAULT DETECTION SYSTEM ===\n")
        
        all_results = {}
        
        for name, df in datasets.items():
            print(f"\nProcessing {name.upper()} Dataset...")
            
            if name == 'cia':
                X, y = self.preprocess_cia_data_optimized(df)
            else:
                X, y = self.preprocess_other_datasets(df, name)
            
            # Train classifier
            fault_results = self.train_optimized_classifier(X, y, name)
            
            all_results[name] = {
                'fault_classification': fault_results,
                'processed_data': (X, y),
                'data_stats': self.data_stats[name]
            }
            
            # Display results
            print(f"\n{name.upper()} Results:")
            print(f"  Accuracy: {fault_results['accuracy']:.4f}")
            print(f"  F1-Score: {fault_results['f1_score']:.4f}")
            print(f"  CV Score: {fault_results['cv_mean']:.4f} ± {fault_results['cv_std']:.4f}")
            print(f"  Ensemble: {fault_results['ensemble_models']}")
            
            # LLM Analysis
            print(f"\n--- FACTUAL ANALYSIS FOR {name.upper()} ---")
            llm_analysis = self.analyze_with_factual_llm(name, fault_results, self.data_stats[name])
            print(llm_analysis)
            print("-" * 80)
        
        return all_results
    
    def save_qlora_config(self):
        """Save QLora configuration for fine-tuning"""
        config = {
            "model_name": "dolphin-mistral",
            "task": "industrial_fault_detection",
            "training_params": {
                "learning_rate": 2e-4,
                "batch_size": 4,
                "num_epochs": 3,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1
            },
            "prompt_template": {
                "instruction": "You are a technical expert in industrial fault detection. Provide factual analysis based only on provided data.",
                "input": "{context}",
                "output": "{response}"
            }
        }
        
        with open('qlora_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("QLora configuration saved to: qlora_config.json")

def main():
    detector = QLORAFineTunedFaultDetector()
    
    print("Loading datasets for optimized analysis...")
    datasets = detector.load_datasets()
    
    print("\nRunning QLora-optimized fault detection...")
    results = detector.generate_comprehensive_report(datasets)
    
    print("\n=== PERFORMANCE SUMMARY ===")
    for dataset_name, result in results.items():
        metrics = result['fault_classification']
        stats = result['data_stats']
        
        print(f"\n{dataset_name.upper()}:")
        print(f"  Dataset: {stats['rows']} samples, {stats['cols']} features")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Reliability: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        print(f"  Ensemble: {', '.join(metrics['ensemble_models'])}")
    
    # Generate QLora training data
    training_data = detector.generate_qlora_training_data()
    detector.save_qlora_config()
    
    print(f"\n=== QLORA FINE-TUNING PREPARATION ===")
    print(f"Training examples collected: {len(training_data)}")
    print("Files generated:")
    print("  - qlora_training_data.json (training examples)")
    print("  - qlora_config.json (fine-tuning configuration)")
    
    return detector, results

if __name__ == "__main__":
    detector, results = main()