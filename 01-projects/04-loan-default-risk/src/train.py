from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from data import load_data
from features import build_preprocessor

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load data
X_train, X_test, y_train, y_test = load_data(
    BASE_DIR / "data" / "raw" / "loan.csv"
)

# Build preprocessing pipeline
preprocessor = build_preprocessor(X_train)

# Create model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Create models folder
models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

# Save model
joblib.dump(
    model,
    models_dir / "model.joblib"
)

# Save chosen threshold
with open(models_dir / "threshold.txt", "w") as f:
    f.write("0.35")

print("Model trained successfully!")
print("Model saved to models/model.joblib")
print("Threshold saved to models/threshold.txt")