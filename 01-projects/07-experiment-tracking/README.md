# Experiment Tracking Basics

## Goal

This project demonstrates how to track machine learning experiments using simple CSV logging.

## Dataset

Telco Customer Churn Dataset

## Experiments

- Logistic Regression (C=1.0)
- Logistic Regression (C=0.1)
- Random Forest (100 trees)
- Random Forest (300 trees)

Each run logs:

- Experiment ID
- Timestamp
- Model
- Parameters
- Accuracy
- F1 Score
- ROC-AUC

## Best Model

The best model is automatically saved to:

models/best_model.joblib

## Reproducibility

Run:

```bash
python src/train.py
```

to reproduce all experiments.

## Experiment Comparison

See:

- reports/experiment_comparison.png

- experiments/experiments.csvS