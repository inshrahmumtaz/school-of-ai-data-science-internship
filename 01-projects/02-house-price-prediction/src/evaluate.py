import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data import load_data, split_data


# Load data
X, y = load_data("data/raw/house_prices.csv")

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Load model
model = joblib.load("models/model.joblib")

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# Feature Importance
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

importances = model.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = (
    importance_df
    .sort_values("Importance", ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Feature Importances")

plt.savefig(
    "reports/figures/feature_importance.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()