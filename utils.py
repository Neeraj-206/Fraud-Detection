"""
Utility functions for fraud detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_transaction_features(df, user_col='user_id', amount_col='amount', time_col=None):
    """
    Calculate user-level transaction features
    
    Args:
        df: DataFrame with transaction data
        user_col: Column name for user identifier
        amount_col: Column name for transaction amount
        time_col: Column name for transaction time (optional)
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # User transaction statistics
    user_stats = df.groupby(user_col).agg({
        amount_col: ['count', 'mean', 'std', 'sum', 'min', 'max']
    }).reset_index()
    
    user_stats.columns = [user_col, 'user_txn_count', 'user_avg_amount', 
                         'user_std_amount', 'user_total_amount', 
                         'user_min_amount', 'user_max_amount']
    
    df = df.merge(user_stats, on=user_col, how='left')
    
    # Amount deviation from user average
    df['amount_deviation'] = (df[amount_col] - df['user_avg_amount']) / (df['user_std_amount'] + 1e-6)
    
    # Transaction frequency features (if time column available)
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Time since last transaction
        df = df.sort_values([user_col, time_col])
        df['time_since_last'] = df.groupby(user_col)[time_col].diff().dt.total_seconds() / 3600
        df['time_since_last'] = df['time_since_last'].fillna(24)  # Default to 24 hours
        
        # Hour of day
        df['hour'] = df[time_col].dt.hour
        df['is_night'] = (df['hour'] >= 22) | (df['hour'] <= 6)
        df['is_weekend'] = df[time_col].dt.dayofweek >= 5
    
    return df


def detect_outliers_iqr(df, columns, factor=1.5):
    """
    Detect outliers using IQR method
    
    Args:
        df: DataFrame
        columns: List of column names to check
        factor: IQR factor (default 1.5)
        
    Returns:
        Boolean Series indicating outliers
    """
    outlier_mask = pd.Series([False] * len(df))
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            outlier_mask |= (df[col] < lower_bound) | (df[col] > upper_bound)
    
    return outlier_mask


def calculate_risk_score(amount, user_avg_amount, user_std_amount, 
                        time_since_last=None, is_night=False, is_weekend=False):
    """
    Calculate a simple risk score for a transaction
    
    Args:
        amount: Transaction amount
        user_avg_amount: User's average transaction amount
        user_std_amount: User's transaction amount standard deviation
        time_since_last: Hours since last transaction
        is_night: Whether transaction is at night
        is_weekend: Whether transaction is on weekend
        
    Returns:
        Risk score (0-100)
    """
    risk_score = 0
    
    # Amount-based risk
    if user_std_amount > 0:
        z_score = abs((amount - user_avg_amount) / user_std_amount)
        risk_score += min(z_score * 10, 40)  # Max 40 points
    
    # Time-based risk
    if time_since_last is not None:
        if time_since_last < 1:  # Very recent transaction
            risk_score += 10
        elif time_since_last > 48:  # Long gap
            risk_score += 5
    
    # Temporal risk
    if is_night:
        risk_score += 15
    if is_weekend:
        risk_score += 5
    
    return min(risk_score, 100)


def balance_dataset(X, y, method='undersample'):
    """
    Balance the dataset to handle class imbalance
    
    Args:
        X: Feature matrix
        y: Target labels
        method: 'undersample' or 'oversample'
        
    Returns:
        Balanced X and y
    """
    from sklearn.utils import resample
    
    df = pd.DataFrame(X)
    df['target'] = y
    
    fraud = df[df['target'] == 1]
    normal = df[df['target'] == 0]
    
    if method == 'undersample':
        normal_downsampled = resample(normal,
                                     replace=False,
                                     n_samples=len(fraud) * 2,  # 2:1 ratio
                                     random_state=42)
        df_balanced = pd.concat([fraud, normal_downsampled])
    else:  # oversample
        fraud_upsampled = resample(fraud,
                                  replace=True,
                                  n_samples=len(normal) // 10,  # 10% fraud rate
                                  random_state=42)
        df_balanced = pd.concat([normal, fraud_upsampled])
    
    X_balanced = df_balanced.drop('target', axis=1).values
    y_balanced = df_balanced['target'].values
    
    return X_balanced, y_balanced


def plot_fraud_distribution(df, fraud_col='is_fraud', amount_col='amount'):
    """
    Plot distribution of fraud vs normal transactions
    
    Args:
        df: DataFrame with transaction data
        fraud_col: Column name for fraud indicator
        amount_col: Column name for transaction amount
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Amount distribution
    fraud_amounts = df[df[fraud_col] == 1][amount_col]
    normal_amounts = df[df[fraud_col] == 0][amount_col]
    
    axes[0].hist(normal_amounts, bins=50, alpha=0.7, label='Normal', color='blue')
    axes[0].hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud', color='red')
    axes[0].set_xlabel('Transaction Amount')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Transaction Amount Distribution')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # Fraud rate by amount bins
    df['amount_bin'] = pd.cut(df[amount_col], bins=20)
    fraud_rate = df.groupby('amount_bin')[fraud_col].mean()
    
    axes[1].bar(range(len(fraud_rate)), fraud_rate.values, color='red', alpha=0.7)
    axes[1].set_xlabel('Amount Bin')
    axes[1].set_ylabel('Fraud Rate')
    axes[1].set_title('Fraud Rate by Transaction Amount')
    axes[1].set_xticks(range(0, len(fraud_rate), 5))
    axes[1].set_xticklabels([str(fraud_rate.index[i]) for i in range(0, len(fraud_rate), 5)], 
                            rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('fraud_distribution.png', dpi=300, bbox_inches='tight')
    print("Fraud distribution plot saved as 'fraud_distribution.png'")

