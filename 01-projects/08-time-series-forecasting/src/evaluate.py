import joblib

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from data import load_data
from features import create_features


df = load_data()
df = create_features(df)

split = int(len(df) * 0.8)

test = df.iloc[split:]

features = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14"
]

X_test = test[features]
y_test = test["AvgTemperature"]

model = joblib.load("models/model.joblib")

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)

rmse = np.sqrt(
    mean_squared_error(y_test, pred)
)

mape = mean_absolute_percentage_error(
    y_test,
    pred
) * 100

print("MAE :", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

plt.figure(figsize=(15, 5))

plt.plot(test["Date"], y_test, label="Actual")
plt.plot(test["Date"], pred, label="Predicted")

plt.legend()

plt.tight_layout()

plt.savefig("reports/figures/forecast_plot.png", dpi=200)

plt.show()