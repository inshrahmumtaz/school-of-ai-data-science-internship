import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data import load_data
from tracker import log_experiment


# Load data
X_train, X_test, y_train, y_test = load_data()

# Detect feature types
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Experiments
experiments = [
    ("LogisticRegression", LogisticRegression(C=1.0, max_iter=1000)),
    ("LogisticRegression", LogisticRegression(C=0.1, max_iter=1000)),
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=42))
]

best_auc = 0
best_pipeline = None

for i, (name, model) in enumerate(experiments, start=1):

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    log_experiment(
        experiment_id=i,
        model_name=name,
        parameters=model.get_params(),
        accuracy=accuracy,
        f1=f1,
        roc_auc=roc_auc
    )

    print(f"Experiment {i}")
    print("Model:", name)
    print("Accuracy:", round(accuracy, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("-" * 40)

    if roc_auc > best_auc:
        best_auc = roc_auc
        best_pipeline = pipeline

# Save best model
joblib.dump(best_pipeline, "models/best_model.joblib")

print("Best model saved!")