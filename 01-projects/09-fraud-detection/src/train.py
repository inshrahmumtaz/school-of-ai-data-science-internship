import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from data import load_data
from features import build_pipeline


X_train, X_test, y_train, y_test = load_data()

# Logistic Regression
log_model = build_pipeline(
    LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
)

log_model.fit(X_train, y_train)

log_prob = log_model.predict_proba(X_test)[:, 1]

log_score = average_precision_score(y_test, log_prob)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_score = average_precision_score(y_test, rf_prob)

print("Logistic PR-AUC:", log_score)
print("Random Forest PR-AUC:", rf_score)

if rf_score > log_score:
    best_model = rf_model
    print("Best Model: Random Forest")
else:
    best_model = log_model
    print("Best Model: Logistic Regression")

joblib.dump(best_model, "models/model.joblib")

with open("models/threshold.txt", "w") as f:
    f.write("0.30")

print("Model saved.")