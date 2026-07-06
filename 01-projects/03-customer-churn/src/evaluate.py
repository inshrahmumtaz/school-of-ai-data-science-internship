from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    BASE_DIR / "data" / "raw" / "churn.csv"
)

# Load model
model = joblib.load(BASE_DIR / "models" / "model.joblib")

# Load threshold
with open(BASE_DIR / "models" / "threshold.txt", "r") as f:
    threshold = float(f.read())

# Predictions
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# Metrics
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC:", average_precision_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

# Create figures directory
figures_dir = BASE_DIR / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")

plt.savefig(
    figures_dir / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# -------------------------
# Precision-Recall Curve
# -------------------------
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(8,6))

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.savefig(
    figures_dir / "pr_curve.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# -------------------------
# Feature Importance
# -------------------------
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

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Feature Importances")

plt.savefig(
    figures_dir / "top_features.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Evaluation completed!")