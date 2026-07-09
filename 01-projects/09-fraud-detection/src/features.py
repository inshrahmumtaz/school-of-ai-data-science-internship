from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(model):

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    return pipeline