import pandas as pd


def load_data():
    df = pd.read_csv("data/raw/time_series.csv")

    df = df[df["City"] == "Washington"].copy()

    df["Date"] = pd.to_datetime(
        dict(
            year=df["Year"],
            month=df["Month"],
            day=df["Day"]
        ),
        errors="coerce"
    )

    df = df.dropna(subset=["Date"])

    df = df.sort_values("Date")

    df = df[["Date", "AvgTemperature"]]

    return df