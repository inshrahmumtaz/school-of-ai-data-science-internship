import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():

    df = pd.read_csv("data/raw/fraud.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )