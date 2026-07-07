# Loan Default Risk Prediction

## Project Overview

This project predicts whether a loan applicant is likely to default using machine learning techniques. The objective is to build an explainable and fair classification model that can support loan risk assessment.

---

## Dataset

Loan Prediction Dataset

Target:
- Loan_Status
  - Y → 0 (No Default)
  - N → 1 (Default / High Risk)

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib

---

## Project Structure

```
04-loan-default-risk/
│
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── README.md
└── requirements.txt
```

---

## Modeling Approach

- Data Cleaning
- Missing Value Imputation
- One-Hot Encoding
- Standard Scaling
- Logistic Regression (Baseline)
- Random Forest Classifier (Improved)

---

## Evaluation Metrics

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1 Score

---

## Threshold Strategy

A threshold of **0.35** was selected to increase recall and identify more high-risk applicants while maintaining reasonable precision.

---

## Explainability

Feature importance analysis identifies the variables that contribute most to loan default prediction.

---

## Fairness Check

A fairness analysis compares:

- Positive prediction rate
- Recall
- False positive rate

across applicant groups (Gender).

Results are stored in:

```
reports/fairness_report.csv
```

---

## Leakage Prevention

Identifier columns such as **Loan_ID** were removed before model training to avoid leakage.

---

## Limitations

This is a learning project and should **not** be used for real-world credit approval decisions without additional validation, fairness testing, regulatory review, and domain expertise.

---

## Generated Figures

- Default Rate
- Missing Values
- Credit History
- Correlation Heatmap
- Confusion Matrix
- Precision-Recall Curve
- Feature Importance

---

## How to Run

```bash
pip install -r requirements.txt

python src/train.py

python src/evaluate.py

python src/fairness.py
```

---

## Author

**Inshrah Mumtaz**