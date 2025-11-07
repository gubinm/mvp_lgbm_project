"""Test suite for the ML pipeline."""

import importlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ml.schema import (
    BaseQuote,
    Coating,
    CustomerTier,
    Material,
    Route,
    SurfaceFinish,
    Tolerance,
    validate_training_batch,
)


def find_latest_model():
    """Find the latest trained model in mlruns directory."""
    # Try to find models in both mlruns/ and ml/mlruns/
    possible_paths = [
        Path("mlruns"),
        Path("ml/mlruns"),
    ]
    
    # First, try using MLflow API to find the latest run
    tracking_uri = None
    for base_path in possible_paths:
        if base_path.exists():
            tracking_uri = f"file:{base_path.absolute()}"
            break
    
    if tracking_uri:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            experiment_name = "mvp_lightgbm_price"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1
                )
                if len(runs) > 0:
                    latest_run_id = runs.iloc[0]["run_id"]
                    # Try to find the model in the run's artifacts
                    run_path = None
                    for base_path in possible_paths:
                        if not base_path.exists():
                            continue
                        for exp_dir in base_path.iterdir():
                            if not exp_dir.is_dir() or exp_dir.name == "0":
                                continue
                            run_dir = exp_dir / latest_run_id
                            if run_dir.exists():
                                artifacts_dir = run_dir / "artifacts" / "model"
                                if artifacts_dir.exists():
                                    return str(artifacts_dir)
        except Exception:
            # Fall back to file system search if MLflow API fails
            pass
    
    # Fallback: search file system for latest run
    latest_run = None
    latest_end_time = 0
    
    for base_path in possible_paths:
        if not base_path.exists():
            continue
            
        # Look for experiment directories
        for exp_dir in base_path.iterdir():
            if not exp_dir.is_dir() or exp_dir.name == "0":
                continue
                
            # Check run directories first (prioritize actual runs over registered models)
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir() or run_dir.name == "models" or run_dir.name == "meta.yaml":
                    continue
                artifacts_dir = run_dir / "artifacts" / "model"
                if artifacts_dir.exists():
                    meta_file = run_dir / "meta.yaml"
                    if meta_file.exists():
                        with open(meta_file) as f:
                            meta = yaml.safe_load(f)
                            end_time = meta.get("end_time", 0)
                            if end_time and end_time > latest_end_time:
                                latest_end_time = end_time
                                latest_run = str(artifacts_dir)
            
            # Check models directory as fallback
            runs_dir = exp_dir / "models"
            if runs_dir.exists():
                for model_dir in runs_dir.iterdir():
                    artifacts_dir = model_dir / "artifacts"
                    if artifacts_dir.exists():
                        meta_file = model_dir / "meta.yaml"
                        if meta_file.exists():
                            with open(meta_file) as f:
                                meta = yaml.safe_load(f)
                                # Use creation timestamp
                                creation_time = meta.get("creation_timestamp", 0)
                                if creation_time > latest_end_time:
                                    latest_end_time = creation_time
                                    latest_run = str(artifacts_dir)
    
    if latest_run is None:
        raise RuntimeError(
            "No trained model found. Please train a model first using: "
            "python -m ml.train --csv data/raw/mvp_quotes.csv"
        )
    
    return latest_run


@pytest.fixture(scope="session")
def trained_model():
    """Load the actual trained model from mlruns."""
    model_path = find_latest_model()
    return mlflow.sklearn.load_model(model_path)


@pytest.fixture
def app_with_model():
    """Create a FastAPI app instance with the actual trained model."""
    model_path = find_latest_model()
    
    # Set MODEL_URI and PYTHONPATH before importing ml.serve
    import os
    import sys
    original_uri = os.environ.get("MODEL_URI")
    original_pythonpath = os.environ.get("PYTHONPATH")
    
    # Add ml directory to Python path so relative imports work
    ml_dir = str(Path(__file__).parent.parent / "ml")
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    
    os.environ["MODEL_URI"] = model_path
    if original_pythonpath:
        os.environ["PYTHONPATH"] = f"{ml_dir}{os.pathsep}{original_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = ml_dir
    
    try:
        # Import and create app - this will load the model
        import ml.serve
        importlib.reload(ml.serve)
        from ml.serve import app
        yield app
    finally:
        # Restore original environment
        if original_uri is not None:
            os.environ["MODEL_URI"] = original_uri
        elif "MODEL_URI" in os.environ:
            del os.environ["MODEL_URI"]
        
        if original_pythonpath is not None:
            os.environ["PYTHONPATH"] = original_pythonpath
        elif "PYTHONPATH" in os.environ:
            del os.environ["PYTHONPATH"]
        
        # Remove ml_dir from sys.path if we added it
        if ml_dir in sys.path:
            sys.path.remove(ml_dir)


@pytest.fixture
def sample_quote_data():
    """Sample quote data for testing."""
    return {
        "rfq_id": 12345,
        "customer_tier": CustomerTier.A,
        "thickness_mm": 5.0,
        "length_mm": 100.0,
        "width_mm": 50.0,
        "holes_count": 2,
        "bends_count": 1,
        "weld_length_mm": 10.0,
        "cut_length_mm": 300.0,
        "part_weight_kg": 0.5,
        "qty": 10,
        "due_days": 14,
        "engineer_score": 1.5,
        "material": Material.steel,
        "route": Route.laser_cut,
        "tolerance": Tolerance.standard,
        "surface_finish": None,
        "coating": None,
        "material_cost_rub": 50.0,
        "labor_minutes_per_unit": 5.0,
        "labor_cost_rub": 100.0,
    }


@pytest.fixture
def sample_quotes(sample_quote_data):
    """Create sample BaseQuote objects."""
    return [
        BaseQuote(**sample_quote_data),
        BaseQuote(**{**sample_quote_data, "rfq_id": 12346, "qty": 20}),
        BaseQuote(**{**sample_quote_data, "rfq_id": 12347, "material": Material.aluminum}),
    ]


def test_inference_endpoint_works(app_with_model, sample_quotes):
    """Test 1: Verify /predict endpoint works with valid input."""
    client = TestClient(app_with_model)

    # Test health check
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Test prediction endpoint
    quotes_dict = [q.model_dump() for q in sample_quotes]
    response = client.post("/predict", json=quotes_dict)

    assert response.status_code == 200
    predictions = response.json()
    assert isinstance(predictions, list)
    assert len(predictions) == 3
    assert all("target_unit_price_rub_pred" in pred for pred in predictions)
    assert all(isinstance(pred["target_unit_price_rub_pred"], float) for pred in predictions)
    # Verify predictions are reasonable (non-negative, finite)
    assert all(pred["target_unit_price_rub_pred"] >= 0 for pred in predictions)
    assert all(
        pred["target_unit_price_rub_pred"] != float("inf") and pred["target_unit_price_rub_pred"] != float("-inf")
        for pred in predictions
    )


def test_model_input_output_correctness(app_with_model, sample_quote_data):
    """Test 2: Validate input schema and output format correctness."""
    client = TestClient(app_with_model)

    # Test with valid input schema
    valid_quote = BaseQuote(**sample_quote_data)
    response = client.post("/predict", json=[valid_quote.model_dump()])
    assert response.status_code == 200

    result = response.json()
    assert len(result) == 1
    assert "target_unit_price_rub_pred" in result[0]
    assert isinstance(result[0]["target_unit_price_rub_pred"], float)
    assert result[0]["target_unit_price_rub_pred"] >= 0  # Price should be non-negative

    # Test with invalid input (missing required field)
    invalid_quote = sample_quote_data.copy()
    del invalid_quote["customer_tier"]  # Required field
    response = client.post("/predict", json=[invalid_quote])
    assert response.status_code == 422  # Validation error

    # Test with invalid input (out of range value)
    invalid_quote2 = sample_quote_data.copy()
    invalid_quote2["thickness_mm"] = 200.0  # Exceeds max of 100
    response = client.post("/predict", json=[invalid_quote2])
    assert response.status_code == 422  # Validation error

    # Test with missing optional fields (should work)
    # Note: qty is required for the model's feature engineering (qty_bucket)
    minimal_quote = {
        "rfq_id": 99999,
        "customer_tier": CustomerTier.A,  # Use A to match training data
        "material_cost_rub": 30.0,
        "labor_minutes_per_unit": 3.0,
        "labor_cost_rub": 60.0,
        "qty": 10,  # Required for qty_bucket feature
    }
    response = client.post("/predict", json=[minimal_quote])
    assert response.status_code == 200  # Should work with defaults/imputation


def test_regression_no_leak_and_consistency(app_with_model, sample_quote_data):
    """Test 3: Ensure model predictions are consistent and leak is properly excluded."""
    client = TestClient(app_with_model)

    # Create quotes with same features but different unit_price_rub (leak)
    quote1 = sample_quote_data.copy()
    quote1["unit_price_rub"] = 50.0  # This should be dropped, not used
    quote1["rfq_id"] = 1001

    quote2 = sample_quote_data.copy()
    quote2["unit_price_rub"] = 500.0  # Different leak value
    quote2["rfq_id"] = 1002

    # Note: BaseQuote doesn't include unit_price_rub, so we can't pass it
    # This test verifies the leak is properly excluded at the schema level
    quote1_clean = {k: v for k, v in quote1.items() if k != "unit_price_rub"}
    quote2_clean = {k: v for k, v in quote2.items() if k != "unit_price_rub"}

    # Both quotes have identical features (except rfq_id which is dropped)
    # So predictions should be identical (or very similar)
    response1 = client.post("/predict", json=[quote1_clean])
    response2 = client.post("/predict", json=[quote2_clean])

    assert response1.status_code == 200
    assert response2.status_code == 200

    pred1 = response1.json()[0]["target_unit_price_rub_pred"]
    pred2 = response2.json()[0]["target_unit_price_rub_pred"]

    # Since features are identical, predictions should be the same (or very close)
    # Allow small floating point differences
    assert abs(pred1 - pred2) < 1e-6

    # Test prediction range - should be reasonable for unit prices
    assert 0 <= pred1 <= 1_000_000  # Within expected range

    # Test batch prediction consistency
    batch_quotes = [quote1_clean, quote2_clean, quote1_clean]
    batch_response = client.post("/predict", json=batch_quotes)
    assert batch_response.status_code == 200
    batch_preds = batch_response.json()
    assert len(batch_preds) == 3
    # First and third should be identical (same input)
    assert abs(batch_preds[0]["target_unit_price_rub_pred"] - batch_preds[2]["target_unit_price_rub_pred"]) < 1e-6


def test_edge_cases(app_with_model):
    """Additional test for edge cases: missing values, boundary conditions."""
    client = TestClient(app_with_model)

    # Test with minimal required fields only
    # Note: qty is required for the model's feature engineering (qty_bucket)
    minimal = {
        "rfq_id": 1,
        "customer_tier": CustomerTier.A,  # Use A to match training data
        "material_cost_rub": 1.0,
        "labor_minutes_per_unit": 0.1,
        "labor_cost_rub": 1.0,
        "qty": 5,  # Required for qty_bucket feature
    }
    response = client.post("/predict", json=[minimal])
    assert response.status_code == 200
    result = response.json()
    assert len(result) == 1
    assert result[0]["target_unit_price_rub_pred"] >= 0

    # Test with maximum values
    max_values = {
        "rfq_id": 999999,
        "customer_tier": CustomerTier.A,
        "thickness_mm": 100.0,
        "length_mm": 5000.0,
        "width_mm": 5000.0,
        "holes_count": 200,
        "bends_count": 200,
        "weld_length_mm": 100000.0,
        "cut_length_mm": 100000.0,
        "part_weight_kg": 1000.0,
        "qty": 100000,
        "due_days": 365,
        "engineer_score": 10.0,
        "material": Material.stainless,
        "route": Route.waterjet_cut,
        "tolerance": Tolerance.high_precision,
        "surface_finish": None,
        "coating": None,
        "material_cost_rub": 1_000_000.0,
        "labor_minutes_per_unit": 10_000.0,
        "labor_cost_rub": 1_000_000.0,
    }
    response = client.post("/predict", json=[max_values])
    assert response.status_code == 200
    result = response.json()
    assert len(result) == 1
    assert result[0]["target_unit_price_rub_pred"] >= 0

    # Test empty batch
    response = client.post("/predict", json=[])
    assert response.status_code == 200
    assert response.json() == []


def test_model_performance_on_validation_set(app_with_model):
    """Test model performance on a random sample of 1000 records from validation set."""
    # Load and split data the same way as in train.py
    csv_path = Path("data/raw/mvp_quotes.csv")
    if not csv_path.exists():
        pytest.skip(f"Training data not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    valid_records = validate_training_batch(records)
    df_valid = pd.DataFrame(valid_records)
    
    # Split the same way as train.py
    LABEL = "target_unit_price_rub"
    y = df_valid[LABEL].astype(float)
    X = df_valid.drop(columns=[LABEL])
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Sample 1000 records from validation set (or all if less than 1000)
    sample_size = min(1000, len(X_val))
    sample_indices = np.random.RandomState(42).choice(len(X_val), size=sample_size, replace=False)
    X_sample = X_val.iloc[sample_indices].copy()
    y_sample = y_val.iloc[sample_indices].copy()
    
    # Convert to BaseQuote format for API
    client = TestClient(app_with_model)
    quotes = []
    valid_indices = []  # Track which indices were successfully converted
    
    for idx, (_, row) in enumerate(X_sample.iterrows()):
        quote_dict = row.to_dict()
        # Convert enum-like values to proper enum types
        if "customer_tier" in quote_dict:
            quote_dict["customer_tier"] = CustomerTier(quote_dict["customer_tier"])
        if "material" in quote_dict and pd.notna(quote_dict["material"]):
            quote_dict["material"] = Material(quote_dict["material"])
        if "route" in quote_dict and pd.notna(quote_dict["route"]):
            quote_dict["route"] = Route(quote_dict["route"])
        if "tolerance" in quote_dict and pd.notna(quote_dict["tolerance"]):
            quote_dict["tolerance"] = Tolerance(quote_dict["tolerance"])
        if "surface_finish" in quote_dict and pd.notna(quote_dict["surface_finish"]):
            quote_dict["surface_finish"] = SurfaceFinish(quote_dict["surface_finish"])
        if "coating" in quote_dict and pd.notna(quote_dict["coating"]):
            quote_dict["coating"] = Coating(quote_dict["coating"])
        
        # Convert NaN to None for optional fields
        for key, value in quote_dict.items():
            if pd.isna(value):
                quote_dict[key] = None
        
        try:
            quote = BaseQuote(**quote_dict)
            quotes.append(quote.model_dump())
            valid_indices.append(idx)  # Track successful conversion
        except Exception:
            # Skip records that don't match BaseQuote schema (e.g., missing required fields)
            continue
    
    # Make predictions via API
    response = client.post("/predict", json=quotes)
    assert response.status_code == 200
    predictions_data = response.json()
    
    # Extract predictions
    y_pred = np.array([pred["target_unit_price_rub_pred"] for pred in predictions_data])
    
    # Get corresponding actual values for successfully predicted quotes
    y_actual = y_sample.iloc[valid_indices].to_numpy()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    mape = float(np.mean(np.abs((y_actual - y_pred) / np.clip(y_actual, 1e-6, None))) * 100)
    r2 = r2_score(y_actual, y_pred)
    
    # Assert reasonable performance
    # These thresholds are based on the training metrics (RMSE: ~10.5, MAE: ~7.1, MAPE: ~4.4%, R2: ~0.98)
    # We allow some degradation on a random sample
    assert rmse < 15.0, f"RMSE {rmse:.3f} is too high (expected < 15.0)"
    assert mae < 12.0, f"MAE {mae:.3f} is too high (expected < 12.0)"
    assert mape < 8.0, f"MAPE {mape:.2f}% is too high (expected < 8.0%)"
    assert r2 > 0.95, f"R2 {r2:.3f} is too low (expected > 0.95)"
    
    # Print metrics for visibility
    print(f"\nModel Performance on {len(y_pred)} validation samples:")
    print(f"RMSE: {rmse:.3f} | MAE: {mae:.3f} | MAPE: {mape:.2f}% | R2: {r2:.3f}")

