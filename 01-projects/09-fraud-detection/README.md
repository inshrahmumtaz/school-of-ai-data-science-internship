# Fraud Detection with Imbalanced Classification

## Business Problem

Financial fraud is rare but costly. A model must identify fraudulent transactions while minimizing missed fraud cases. Because fraud is highly imbalanced, accuracy alone is not an appropriate metric.

## Dataset

- Kaggle Credit Card Fraud Detection Dataset
- 284,807 transactions
- 492 fraud cases

## Models

- Logistic Regression (Baseline)
- Random Forest (Improved)

## Class Imbalance Strategy

- Used `class_weight="balanced"`
- Applied a decision threshold of 0.30

## Evaluation Metrics

- PR-AUC (Primary)
- ROC-AUC
- Recall

## Key Visualizations

- Fraud distribution
- Transaction amount distribution
- Confusion matrix
- Precision-Recall curve
- Feature importance

## Business Recommendation

A lower decision threshold improves fraud detection recall, reducing the number of missed fraudulent transactions. Although this increases false positives, it is often acceptable because missed fraud can result in significant financial loss.

## Run

```bash
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```