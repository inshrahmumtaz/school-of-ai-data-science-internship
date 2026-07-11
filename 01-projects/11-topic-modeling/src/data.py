import pandas as pd

def load_data():
    df = pd.read_csv(
        "data/raw/text_corpus.csv",
        sep="\t",
        engine="python"
    )

    return df