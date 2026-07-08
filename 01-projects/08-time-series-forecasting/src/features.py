import pandas as pd


def create_features(df):

    df = df.copy()

    df["lag_1"] = df["AvgTemperature"].shift(1)
    df["lag_7"] = df["AvgTemperature"].shift(7)
    df["lag_14"] = df["AvgTemperature"].shift(14)

    df["rolling_mean_7"] = (
        df["AvgTemperature"].rolling(7).mean()
    )

    df["rolling_mean_14"] = (
        df["AvgTemperature"].rolling(14).mean()
    )

    df = df.dropna()

    return df