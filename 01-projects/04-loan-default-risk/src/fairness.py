from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import recall_score, confusion_matrix

from data import load_data

BASE_DIR = Path(__file__).resolve().parent.parent

# Load data
X_train, X_test, y_train, y_test = load_data(
    BASE_DIR / "data" / "raw" / "loan.csv"
)

# Load model
model = joblib.load(BASE_DIR / "models" / "model.joblib")

# Load threshold
with open(BASE_DIR / "models" / "threshold.txt") as f:
    threshold = float(f.read())

# Predictions
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# Fairness check by Gender
report = []

if "Gender" in X_test.columns:

    for group in X_test["Gender"].dropna().unique():

        mask = X_test["Gender"] == group

        yt = y_test[mask]
        yp = y_pred[mask]

        positive_rate = yp.mean()

        recall = recall_score(yt, yp, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(
            yt,
            yp,
            labels=[0,1]
        ).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        report.append({
            "Group": group,
            "PositivePredictionRate": positive_rate,
            "Recall": recall,
            "FalsePositiveRate": fpr
        })

report_df = pd.DataFrame(report)

reports_dir = BASE_DIR / "reports"
reports_dir.mkdir(exist_ok=True)

report_df.to_csv(
    reports_dir / "fairness_report.csv",
    index=False
)

print(report_df)
print("Fairness report saved!")