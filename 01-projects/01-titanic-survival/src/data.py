import os
import pandas as pd

def load_data(path):
    """
    Load the Titanic dataset.
    """

    # Make the path relative to this file (data.py), not the current terminal
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, path)

    df = pd.read_csv(full_path)

    # Remove duplicate rows if any
    df = df.drop_duplicates()

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    return X, y