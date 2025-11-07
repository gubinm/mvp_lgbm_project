import argparse
import os

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from ml.features import FeatureBuilder, ImputeAndTypes
from ml.schema import validate_training_batch

LABEL = "target_unit_price_rub"


def load_and_validate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    valid = validate_training_batch(records)
    return pd.DataFrame(valid)


def build_pipeline(
    num_leaves: int = 64,
    learning_rate: float = 0.05,
    n_estimators: int = 3000,
    min_data_in_leaf: int = 50,
    feature_fraction: float = 0.8,
    bagging_fraction: float = 0.8,
    bagging_freq: int = 1,
    lambda_l2: float = 5.0,
    lambda_l1: float = 0.0,
    min_gain_to_split: float = 0.0,
    max_depth: int = -1,
) -> Pipeline:
    return Pipeline(
        [
            ("impute", ImputeAndTypes()),
            ("features", FeatureBuilder()),
            (
                "model",
                LGBMRegressor(
                    objective="regression",
                    metric="rmse",
                    num_leaves=num_leaves,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    min_data_in_leaf=min_data_in_leaf,
                    feature_fraction=feature_fraction,
                    bagging_fraction=bagging_fraction,
                    bagging_freq=bagging_freq,
                    lambda_l2=lambda_l2,
                    lambda_l1=lambda_l1,
                    min_gain_to_split=min_gain_to_split,
                    max_depth=max_depth,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )


def objective(trial, X_train, y_train, cv_folds: int = 5):
    """Optuna objective function for hyperparameter tuning."""
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 500, 5000),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 20.0, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
    }

    pipe = build_pipeline(**params)

    # Use cross-validation for more robust evaluation
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipe, X_train, y_train, cv=kfold, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    rmse = -scores.mean()

    return rmse


def get_params_from_run(run_id: str) -> dict:
    """Get hyperparameters from a specific MLflow run."""
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id)
    
    params = {}
    for key, value in run_data.data.params.items():
        # Convert string values to appropriate types
        try:
            if '.' in value:
                params[key] = float(value)
            else:
                params[key] = int(value)
        except ValueError:
            params[key] = value
    
    return params


def main(args):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "mvp_lightgbm_price"))

    df = load_and_validate(args.csv)
    y = df[LABEL].astype(float)
    X = df.drop(columns=[LABEL])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get hyperparameters either from a specific run or via Optuna tuning
    if args.use_run_id:
        print(f"Using hyperparameters from run ID: {args.use_run_id}")
        try:
            best_params = get_params_from_run(args.use_run_id)
            print(f"Loaded parameters from run:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
            best_cv_rmse = None  # Not available when using existing run
        except Exception as e:
            print(f"Error loading parameters from run {args.use_run_id}: {e}")
            raise
    else:
        # Hyperparameter tuning with Optuna
        n_trials = args.trials if hasattr(args, "trials") else 50
        print(f"Starting hyperparameter tuning with {n_trials} trials...")
        study = optuna.create_study(direction="minimize", study_name="lgbm_hyperopt")
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, cv_folds=5),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        print(f"\nBest hyperparameters found:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"Best CV RMSE: {study.best_value:.3f}\n")
        
        best_params = study.best_params
        best_cv_rmse = study.best_value

    # Train final model with best hyperparameters
    best_pipe = build_pipeline(**best_params)
    with mlflow.start_run() as run:
        best_pipe.fit(X_train, y_train)

        preds = best_pipe.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        mape = float(np.mean(np.abs((y_val - preds) / np.clip(y_val, 1e-6, None))) * 100)
        r2 = r2_score(y_val, preds)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("r2", r2)
        if best_cv_rmse is not None:
            mlflow.log_metric("best_cv_rmse", best_cv_rmse)

        # Log hyperparameters
        for key, value in best_params.items():
            mlflow.log_param(key, value)

        # Log entire pipeline as a model
        input_example = X_train.head(3)
        mlflow.sklearn.log_model(
            sk_model=best_pipe,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=os.getenv("MLFLOW_REGISTER_MODEL", None),
        )
        print(f"Validation Metrics:")
        print(f"  RMSE: {rmse:.3f} | MAE: {mae:.3f} | MAPE: {mape:.2f}% | R2: {r2:.3f}")
        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to mvp_quotes.csv")
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of Optuna trials for hyperparameter tuning"
    )
    parser.add_argument(
        "--use-run-id", type=str, default=None,
        help="Use hyperparameters from a specific MLflow run ID instead of running Optuna"
    )
    args = parser.parse_args()
    main(args)
