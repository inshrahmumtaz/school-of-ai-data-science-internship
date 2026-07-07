from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    # Absolute path to the dataset
    data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "data.csv"

    df = pd.read_csv(data_path)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop missing values created by conversion
    df = df.dropna()

    # Convert target to 0/1
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # Drop customer ID
    df = df.drop(columns=["customerID"])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )