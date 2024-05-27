import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("test-experiment")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved",
)
def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    mlflow.sklearn.autolog()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run():
        max_depth = 10
        min_samples_split = 2

        mlflow.log_param("maximum-depth", max_depth)
        mlflow.log_param("min-samples-split", max_depth)

        rf = RandomForestRegressor(
            max_depth=max_depth, random_state=0, min_samples_split=min_samples_split
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)


if __name__ == "__main__":
    run_train()
