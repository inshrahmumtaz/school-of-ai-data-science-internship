import pandas as pd
from datetime import datetime
import os


def log_experiment(
    experiment_id,
    model_name,
    parameters,
    accuracy,
    f1,
    roc_auc
):

    file_path = "experiments/experiments.csv"

    row = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(),
        "model_name": model_name,
        "parameters": str(parameters),
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc
    }

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(file_path, index=False)