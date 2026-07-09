import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from data import load_data
from text_preprocess import clean_text

X_train, X_test, y_train, y_test = load_data()

X_test = X_test.apply(clean_text)

model = joblib.load("models/model.joblib")

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, pred))
print("F1:", f1_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))

cm = confusion_matrix(y_test, pred)

ConfusionMatrixDisplay(cm).plot()

plt.savefig(
    "reports/figures/confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

feature_names = model.named_steps["vectorizer"].get_feature_names_out()
coef = model.named_steps["model"].coef_[0]

top = pd.DataFrame({
    "word": feature_names,
    "coef": coef
}).sort_values("coef", ascending=False).head(20)

plt.figure(figsize=(8,6))
sns.barplot(data=top, x="coef", y="word")

plt.title("Top Positive Words")

plt.savefig(
    "reports/figures/top_words.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

errors = X_test[pred != y_test]

comparison = pd.DataFrame({
    "Review": errors,
    "Actual": y_test[pred != y_test],
    "Predicted": pred[pred != y_test]
})

print("\nMisclassified Examples:")
print(comparison.head(10))