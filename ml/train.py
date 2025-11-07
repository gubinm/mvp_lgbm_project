import argparse
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features import FeatureBuilder, ImputeAndTypes
from schema import validate_training_batch

LABEL = "target_unit_price_rub"


def load_and_validate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    valid = validate_training_batch(records)
    return pd.DataFrame(valid)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", ImputeAndTypes()),
            ("features", FeatureBuilder()),
            (
                "model",
                LGBMRegressor(
                    objective="regression",
                    metric="rmse",
                    num_leaves=64,
                    learning_rate=0.05,
                    n_estimators=3000,
                    min_data_in_leaf=50,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    bagging_freq=1,
                    lambda_l2=5.0,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main(args):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "mvp_lightgbm_price"))

    df = load_and_validate(args.csv)
    y = df[LABEL].astype(float)
    X = df.drop(columns=[LABEL])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline()
    with mlflow.start_run() as run:
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        mape = float(np.mean(np.abs((y_val - preds) / np.clip(y_val, 1e-6, None))) * 100)
        r2 = r2_score(y_val, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("r2", r2)

        # Log entire pipeline as a model
        input_example = X_train.head(3)
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=os.getenv("MLFLOW_REGISTER_MODEL", None),
        )
        print(f"RMSE: {rmse:.3f} | MAE: {mae:.3f} | MAPE: {mape:.2f}% | R2: {r2:.3f}")
        print("Run ID:", run.info.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to mvp_quotes.csv")
    args = parser.parse_args()
    main(args)
