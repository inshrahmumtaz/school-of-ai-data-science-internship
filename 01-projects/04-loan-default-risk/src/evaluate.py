from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)

from data import load_data

# Project root
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

# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# Metrics
print("ROC-AUC :", roc_auc_score(y_test, y_prob))
print("PR-AUC  :", average_precision_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1       :", f1_score(y_test, y_pred))

# Figures folder
figures = BASE_DIR / "reports" / "figures"
figures.mkdir(parents=True, exist_ok=True)

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.savefig(figures / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig(figures / "pr_curve.png", dpi=200, bbox_inches="tight")
plt.close()

# Feature Importance
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = model.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = (
    importance_df
    .sort_values("Importance", ascending=False)
    .head(10)
)

plt.figure(figsize=(8,5))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importance")
plt.tight_layout()
plt.savefig(figures / "feature_importance.png", dpi=200)
plt.close()

print("Evaluation completed!")