# Fraud Detection System

A comprehensive machine learning-based fraud detection system for transaction data analysis.

## Features

- **Multiple Model Support**: Random Forest and Isolation Forest algorithms
- **Automatic Feature Engineering**: Creates statistical and temporal features
- **Comprehensive Evaluation**: Classification reports, confusion matrices, ROC curves
- **Feature Importance Analysis**: Identifies most important fraud indicators
- **Flexible Data Loading**: Handles pickle files with transaction data

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script to train and evaluate a fraud detection model:

```bash
python main.py
```

### Data Format

The system expects transaction data in pickle files located in `dataset/data/`. Each pickle file should contain a pandas DataFrame with transaction information.

Expected columns:
- Transaction amount/value
- Transaction IDs
- User/Merchant IDs
- Timestamps (optional)
- Fraud labels (optional, for supervised learning)

### Model Types

1. **Random Forest** (Supervised Learning)
   - Requires labeled data (fraud/non-fraud)
   - Better performance with labeled data
   - Provides feature importance

2. **Isolation Forest** (Unsupervised Learning)
   - Works without labels
   - Detects anomalies/outliers
   - Useful when fraud labels are unavailable

## Project Structure

```
fraud_detection/
├── main.py                 # Main fraud detection script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── dataset/
│   └── data/              # Transaction data files (.pkl)
└── utils/                 # Utility modules (optional)
```

## Output

The script generates:
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve for model evaluation
- `feature_importance.png`: Feature importance plot (Random Forest only)

## Model Performance Metrics

The system evaluates models using:
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix
- ROC-AUC Score
- Feature Importance Rankings

## Customization

### Changing Model Type

Edit `main.py` and modify:
```python
detector = FraudDetector(model_type='isolation_forest')  # or 'random_forest'
```

### Adjusting Model Parameters

Modify the model initialization in the `train()` method:
```python
self.model = RandomForestClassifier(
    n_estimators=200,  # Increase for better performance
    max_depth=25,
    # ... other parameters
)
```

## Example Output

```
==================================================
FRAUD DETECTION SYSTEM
==================================================
Loading data...
Loaded 50000 transactions from 10 files
Preprocessing data...
Selected 15 features
Creating features...
Training random_forest model...
Model training completed!

==================================================
MODEL EVALUATION
==================================================
Classification Report:
              precision    recall  f1-score   support

      Normal       0.99      1.00      0.99      9500
       Fraud       0.95      0.85      0.90       500

ROC-AUC Score: 0.9823
```

## Troubleshooting

### Data Loading Issues

If you encounter errors loading pickle files:
- Ensure files are valid pickle format
- Check that files contain pandas DataFrames
- Verify file paths are correct

### Memory Issues

For large datasets:
- Reduce `sample_size` parameter in `load_data()`
- Use data sampling techniques
- Process data in batches

## License

This project is open source and available for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

