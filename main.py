"""
Fraud Detection System
Main script for training and evaluating fraud detection models
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FraudDetector:
    """Fraud Detection System using Machine Learning"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the fraud detector
        
        Args:
            model_type: Type of model to use ('random_forest' or 'isolation_forest')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, data_dir='dataset/data', sample_size=None):
        """
        Load transaction data from pickle files
        
        Args:
            data_dir: Directory containing pickle files
            sample_size: Number of files to load (None for all)
            
        Returns:
            Combined DataFrame
        """
        print("Loading data...")
        data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
        
        if sample_size:
            data_files = data_files[:sample_size]
        
        all_data = []
        for file in data_files:
            file_path = os.path.join(data_dir, file)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, pd.DataFrame):
                        all_data.append(data)
                    elif isinstance(data, dict):
                        # If data is a dictionary, try to convert to DataFrame
                        all_data.append(pd.DataFrame(data))
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data could be loaded. Please check the data format.")
        
        df = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(df)} transactions from {len(data_files)} files")
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the transaction data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        print("Preprocessing data...")
        df = df.copy()
        
        # Handle missing values
        df = df.fillna(0)
        
        # Convert date columns if they exist
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Extract features from datetime if available
        if date_columns:
            for col in date_columns:
                if df[col].dtype == 'datetime64[ns]':
                    df[f'{col}_hour'] = df[col].dt.hour
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
        
        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if it exists
        target_columns = [col for col in df.columns if 'fraud' in col.lower() or 'is_fraud' in col.lower() or 'label' in col.lower()]
        if target_columns:
            self.target_column = target_columns[0]
            numeric_columns = [col for col in numeric_columns if col != self.target_column]
        else:
            self.target_column = None
        
        # Select feature columns
        self.feature_columns = numeric_columns
        
        print(f"Selected {len(self.feature_columns)} features")
        return df
    
    def create_features(self, df):
        """
        Create additional features for fraud detection
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with additional features
        """
        print("Creating features...")
        df = df.copy()
        
        # Statistical features
        numeric_cols = [col for col in self.feature_columns if col in df.columns]
        
        if len(numeric_cols) > 0:
            # Transaction amount statistics (if amount column exists)
            amount_cols = [col for col in numeric_cols if 'amount' in col.lower() or 'value' in col.lower()]
            if amount_cols:
                df['amount_mean'] = df[amount_cols].mean(axis=1)
                df['amount_std'] = df[amount_cols].std(axis=1)
            
            # Count of non-zero values
            df['non_zero_count'] = (df[numeric_cols] != 0).sum(axis=1)
            
            # Sum of all numeric features
            df['feature_sum'] = df[numeric_cols].sum(axis=1)
            
            # Update feature columns
            self.feature_columns.extend(['amount_mean', 'amount_std', 'non_zero_count', 'feature_sum'])
            self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        return df
    
    def train(self, X, y=None):
        """
        Train the fraud detection model
        
        Args:
            X: Feature matrix
            y: Target labels (optional for Isolation Forest)
        """
        print(f"Training {self.model_type} model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.model_type == 'random_forest':
            if y is None:
                raise ValueError("Target labels required for Random Forest")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            self.model.fit(X_scaled, y)
            
        elif self.model_type == 'isolation_forest':
            self.model = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_scaled)
        
        print("Model training completed!")
    
    def predict(self, X):
        """
        Predict fraud for given transactions
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (1 for fraud, 0 for normal)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'random_forest':
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            return predictions, probabilities
        else:
            predictions = self.model.predict(X_scaled)
            # Convert Isolation Forest predictions (-1 for anomaly, 1 for normal)
            predictions = (predictions == -1).astype(int)
            scores = self.model.score_samples(X_scaled)
            # Convert scores to probabilities (lower scores = higher fraud probability)
            probabilities = 1 / (1 + np.exp(scores))
            return predictions, probabilities
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance
        
        Args:
            X: Feature matrix
            y: True labels
        """
        predictions, probabilities = self.predict(X)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        print("\nClassification Report:")
        print(classification_report(y, predictions, target_names=['Normal', 'Fraud']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y, predictions)
        print(cm)
        
        # Calculate metrics
        try:
            auc_score = roc_auc_score(y, probabilities)
            print(f"\nROC-AUC Score: {auc_score:.4f}")
        except:
            print("\nROC-AUC Score: Could not calculate")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Fraud'],
                    yticklabels=['Normal', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'confusion_matrix.png'")
        
        # Plot ROC curve
        try:
            fpr, tpr, _ = roc_curve(y, probabilities)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
            print("ROC curve saved as 'roc_curve.png'")
        except:
            pass
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': cm
        }


def main():
    """Main function to run fraud detection"""
    print("="*50)
    print("FRAUD DETECTION SYSTEM")
    print("="*50)
    
    # Initialize detector
    detector = FraudDetector(model_type='random_forest')
    
    # Load data
    try:
        df = detector.load_data(data_dir='fraud_detection/dataset/data', sample_size=10)  # Load first 10 files for testing
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating sample data for demonstration...")
        # Create sample data if loading fails
        np.random.seed(42)
        n_samples = 10000
        df = pd.DataFrame({
            'amount': np.random.exponential(100, n_samples),
            'transaction_id': range(n_samples),
            'merchant_id': np.random.randint(1, 100, n_samples),
            'user_id': np.random.randint(1, 1000, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day': np.random.randint(1, 31, n_samples),
            'is_fraud': np.random.binomial(1, 0.05, n_samples)  # 5% fraud rate
        })
        # Make fraud transactions have different characteristics
        fraud_mask = df['is_fraud'] == 1
        df.loc[fraud_mask, 'amount'] = df.loc[fraud_mask, 'amount'] * np.random.uniform(2, 5, fraud_mask.sum())
        df.loc[fraud_mask, 'hour'] = np.random.choice([2, 3, 4, 5], fraud_mask.sum())
    
    # Preprocess data
    df = detector.preprocess_data(df)
    df = detector.create_features(df)
    
    # Prepare features and target
    if detector.target_column and detector.target_column in df.columns:
        X = df[detector.feature_columns]
        y = df[detector.target_column]
    else:
        print("No target column found. Using Isolation Forest for unsupervised detection...")
        detector.model_type = 'isolation_forest'
        X = df[detector.feature_columns]
        y = None
    
    # Split data
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Fraud rate in training: {y_train.mean():.4f}")
        print(f"Fraud rate in test: {y_test.mean():.4f}")
    else:
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        y_train, y_test = None, None
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
    
    # Train model
    detector.train(X_train, y_train)
    
    # Evaluate model
    if y_test is not None:
        results = detector.evaluate(X_test, y_test)
        
        # Feature importance (for Random Forest)
        if detector.model_type == 'random_forest':
            feature_importance = pd.DataFrame({
                'feature': detector.feature_columns,
                'importance': detector.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n" + "="*50)
            print("TOP 10 MOST IMPORTANT FEATURES")
            print("="*50)
            print(feature_importance.head(10).to_string(index=False))
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            top_features = feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top 15 Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("\nFeature importance plot saved as 'feature_importance.png'")
    else:
        predictions, probabilities = detector.predict(X_test)
        print(f"\nPredicted {predictions.sum()} fraud cases out of {len(X_test)} transactions")
        print(f"Fraud rate: {predictions.mean():.4f}")
    
    print("\n" + "="*50)
    print("FRAUD DETECTION COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()


