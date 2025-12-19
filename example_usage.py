"""
Example usage of the Fraud Detection System
"""

from main import FraudDetector
import pandas as pd
import numpy as np

def example_supervised_learning():
    """Example using Random Forest with labeled data"""
    print("="*60)
    print("EXAMPLE 1: Supervised Learning (Random Forest)")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'transaction_id': range(n_samples),
        'amount': np.random.exponential(100, n_samples),
        'user_id': np.random.randint(1, 500, n_samples),
        'merchant_id': np.random.randint(1, 100, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'is_fraud': np.random.binomial(1, 0.05, n_samples)
    }
    
    # Make fraud transactions have different characteristics
    fraud_mask = data['is_fraud'] == 1
    data['amount'] = np.where(fraud_mask, 
                              data['amount'] * np.random.uniform(2, 5, n_samples),
                              data['amount'])
    data['hour'] = np.where(fraud_mask,
                            np.random.choice([2, 3, 4, 5], n_samples),
                            data['hour'])
    
    df = pd.DataFrame(data)
    
    # Initialize detector
    detector = FraudDetector(model_type='random_forest')
    
    # Preprocess
    df = detector.preprocess_data(df)
    df = detector.create_features(df)
    
    # Prepare data
    X = df[detector.feature_columns]
    y = df['is_fraud']
    
    # Split and train
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    detector.train(X_train, y_train)
    
    # Evaluate
    detector.evaluate(X_test, y_test)
    
    print("\n" + "="*60)


def example_unsupervised_learning():
    """Example using Isolation Forest without labels"""
    print("="*60)
    print("EXAMPLE 2: Unsupervised Learning (Isolation Forest)")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'transaction_id': range(n_samples),
        'amount': np.random.exponential(100, n_samples),
        'user_id': np.random.randint(1, 500, n_samples),
        'merchant_id': np.random.randint(1, 100, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
    }
    
    # Create some anomalies (fraud-like patterns)
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    data['amount'] = np.where(np.isin(range(n_samples), anomaly_indices),
                              data['amount'] * np.random.uniform(3, 6, n_samples),
                              data['amount'])
    
    df = pd.DataFrame(data)
    
    # Initialize detector
    detector = FraudDetector(model_type='isolation_forest')
    
    # Preprocess
    df = detector.preprocess_data(df)
    df = detector.create_features(df)
    
    # Prepare data
    X = df[detector.feature_columns]
    
    # Split and train
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Train
    detector.train(X_train)
    
    # Predict
    predictions, probabilities = detector.predict(X_test)
    
    print(f"\nPredicted {predictions.sum()} fraud cases out of {len(X_test)} transactions")
    print(f"Fraud rate: {predictions.mean():.4f}")
    print(f"Average fraud probability: {probabilities.mean():.4f}")
    
    # Show high-risk transactions
    high_risk_indices = np.argsort(probabilities)[-10:][::-1]
    print("\nTop 10 High-Risk Transactions:")
    print(f"Indices: {high_risk_indices}")
    print(f"Probabilities: {probabilities[high_risk_indices]}")
    
    print("\n" + "="*60)


def example_predict_single_transaction():
    """Example of predicting fraud for a single transaction"""
    print("="*60)
    print("EXAMPLE 3: Predicting Single Transaction")
    print("="*60)
    
    # Create and train a model first
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'amount': np.random.exponential(100, n_samples),
        'user_id': np.random.randint(1, 100, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'is_fraud': np.random.binomial(1, 0.1, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    detector = FraudDetector(model_type='random_forest')
    df = detector.preprocess_data(df)
    df = detector.create_features(df)
    
    X = df[detector.feature_columns]
    y = df['is_fraud']
    
    detector.train(X, y)
    
    # Create a new transaction
    new_transaction = pd.DataFrame({
        'amount': [500],  # High amount
        'user_id': [50],
        'hour': [3],  # Night time
    })
    
    # Preprocess new transaction
    new_transaction = detector.preprocess_data(new_transaction)
    new_transaction = detector.create_features(new_transaction)
    
    # Get features that exist in both
    available_features = [f for f in detector.feature_columns if f in new_transaction.columns]
    X_new = new_transaction[available_features]
    
    # Fill missing features with zeros
    for feature in detector.feature_columns:
        if feature not in X_new.columns:
            X_new[feature] = 0
    
    X_new = X_new[detector.feature_columns]
    
    # Predict
    prediction, probability = detector.predict(X_new)
    
    print(f"\nNew Transaction Details:")
    print(f"Amount: {new_transaction['amount'].iloc[0]}")
    print(f"Hour: {new_transaction['hour'].iloc[0]}")
    print(f"\nPrediction: {'FRAUD' if prediction[0] == 1 else 'NORMAL'}")
    print(f"Fraud Probability: {probability[0]:.4f}")
    print(f"Risk Level: {'HIGH' if probability[0] > 0.7 else 'MEDIUM' if probability[0] > 0.3 else 'LOW'}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run examples
    example_supervised_learning()
    print("\n")
    example_unsupervised_learning()
    print("\n")
    example_predict_single_transaction()

