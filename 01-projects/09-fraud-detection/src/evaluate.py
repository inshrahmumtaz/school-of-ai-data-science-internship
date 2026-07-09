import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve
)

from data import load_data


X_train, X_test, y_train, y_test = load_data()

model = joblib.load("models/model.joblib")

prob = model.predict_proba(X_test)[:, 1]

threshold = 0.30

pred = (prob >= threshold).astype(int)

print("PR-AUC:", average_precision_score(y_test, prob))
print("ROC-AUC:", roc_auc_score(y_test, prob))
print("Recall:", recall_score(y_test, pred))

cm = confusion_matrix(y_test, pred)

ConfusionMatrixDisplay(cm).plot()

plt.savefig(
    "reports/figures/confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

precision, recall, _ = precision_recall_curve(y_test, prob)

plt.figure(figsize=(6,5))
plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.savefig(
    "reports/figures/pr_curve.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()