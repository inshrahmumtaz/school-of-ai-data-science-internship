import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from data import load_data, split_data
from features import build_preprocessor


# Load data
X, y = load_data("data/raw/house_prices.csv")

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Build preprocessor
preprocessor = build_preprocessor(X)

# Create pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.joblib")

print("Model trained successfully!")
print("Model saved to models/model.joblib")