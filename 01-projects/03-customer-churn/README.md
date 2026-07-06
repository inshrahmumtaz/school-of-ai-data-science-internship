# Customer Churn Prediction

## Project Overview

This project predicts whether a customer is likely to churn (leave a telecom company) using machine learning. The goal is to help businesses identify at-risk customers and take proactive retention actions.

---

## Dataset

- IBM Telco Customer Churn Dataset
- Target Variable: `Churn`

---

## Technologies Used

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
03-customer-churn/
│
├── data/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
├── README.md
└── requirements.txt
```

---

## Modeling Approach

- Data Cleaning
- Feature Engineering
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

## Threshold Tuning

Instead of using the default threshold (0.50), a threshold of **0.35** was selected to improve recall and identify more customers likely to churn.

---

## Business Impact

Customer churn prediction helps businesses:

- Identify customers at risk of leaving.
- Target retention campaigns effectively.
- Reduce customer acquisition costs.
- Improve long-term customer loyalty.

---

## Generated Figures

- Churn Rate
- Churn by Contract
- Churn by Tenure
- Monthly Charges Distribution
- Correlation Heatmap
- Precision-Recall Curve
- Confusion Matrix
- Top Feature Importance

---

## How to Run

```bash
pip install -r requirements.txt

python src/train.py

python src/evaluate.py
```

---

## Author

**Inshrah Mumtaz**