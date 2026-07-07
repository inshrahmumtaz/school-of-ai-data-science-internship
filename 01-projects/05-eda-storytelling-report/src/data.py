import pandas as pd


def load_data(path):
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.drop_duplicates()

    return df