import joblib

from sklearn.ensemble import RandomForestRegressor

from data import load_data
from features import create_features


df = load_data()

df = create_features(df)

split = int(len(df) * 0.8)

train = df.iloc[:split]

features = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14"
]

X_train = train[features]
y_train = train["AvgTemperature"]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "models/model.joblib")

print("Model saved.")