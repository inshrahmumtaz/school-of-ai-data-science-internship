import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)

    # Convert target
    df["Loan_Status"] = df["Loan_Status"].map({
        "Y": 0,
        "N": 1
    })

    # Drop identifier (not useful for prediction)
    df = df.drop(columns=["Loan_ID"])

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )