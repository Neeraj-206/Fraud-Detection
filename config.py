"""
Configuration file for fraud detection system
"""

# Model Configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    },
    'isolation_forest': {
        'n_estimators': 100,
        'max_samples': 'auto',
        'contamination': 0.1,  # Expected proportion of anomalies
        'random_state': 42,
        'n_jobs': -1
    }
}

# Data Configuration
DATA_CONFIG = {
    'data_dir': 'fraud_detection/dataset/data',
    'sample_size': None,  # None for all files, or specify number
    'test_size': 0.2,
    'random_state': 42
}

# Feature Configuration
FEATURE_CONFIG = {
    'create_statistical_features': True,
    'create_temporal_features': True,
    'create_user_features': True,
    'outlier_detection': True
}

# Evaluation Configuration
EVAL_CONFIG = {
    'save_plots': True,
    'plot_dpi': 300,
    'show_plots': False
}

