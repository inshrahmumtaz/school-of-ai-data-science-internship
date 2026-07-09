import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():

    df = pd.read_csv("data/raw/reviews.csv")

    df["sentiment"] = df["sentiment"].map({
        "negative": 0,
        "positive": 1
    })

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"]
    )

    return X_train, X_test, y_train, y_test