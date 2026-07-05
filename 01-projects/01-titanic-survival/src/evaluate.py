import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from data import load_data

# Load data
X, y = load_data("../data/raw/titanic.csv")

# Load trained model
model = joblib.load("../models/model.joblib")

# Predict
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Print metrics
print("Accuracy:", accuracy_score(y, y_pred))
print("F1 Score:", f1_score(y, y_pred))
print("ROC-AUC:", roc_auc_score(y, y_prob))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(
    "reports/figures/confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()