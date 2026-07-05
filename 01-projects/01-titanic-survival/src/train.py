import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from data import load_data
from features import build_preprocessor


# Load data
X, y = load_data("../data/raw/titanic.csv")

# Build preprocessing pipeline
preprocessor = build_preprocessor()

# Build model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ]
)

# Train model
model.fit(X, y)

# Create models folder if it doesn't exist
os.makedirs("../models", exist_ok=True)

# Save model
joblib.dump(model, "../models/model.joblib")

print("Model trained and saved successfully!")