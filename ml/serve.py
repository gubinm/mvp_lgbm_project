import os
import sys
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ValidationError

# Note: FeatureBuilder and ImputeAndTypes are imported by the model pipeline
from .schema import BaseQuote

# Add ml directory to sys.path so that pickled models can find 'features' and 'schema' modules
# This is needed because models were trained with absolute imports (from features import ...)
ml_dir = str(Path(__file__).parent.absolute())
if ml_dir not in sys.path:
    sys.path.insert(0, ml_dir)

MODEL_URI = os.getenv("MODEL_URI", "")
if not MODEL_URI:
    msg = "Set MODEL_URI (e.g., models:/mvp-lightgbm-price/Production or ./mlruns/.../model)"
    raise RuntimeError(msg)


# Response models
class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status", example="ok")


class PredictionResponse(BaseModel):
    """Single prediction response model."""

    target_unit_price_rub_pred: float = Field(
        ...,
        description="Predicted target unit price in Russian Rubles",
        example=156.78,
        ge=0,
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error message", example="Validation error")


# Initialize FastAPI app with metadata
app = FastAPI(
    title="MVP LightGBM Price API",
    description="""
    API for predicting unit prices of metal fabrication quotes using a LightGBM regression model.

    ## Features

    * **Batch Predictions**: Submit multiple quotes for prediction in a single request
    * **Automatic Feature Engineering**: The API handles feature engineering and imputation automatically
    * **Health Monitoring**: Health check endpoint for service monitoring

    ## Model Information

    The model uses a LightGBM regressor trained on historical quote data with the following metrics:
    - RMSE: ~10.45
    - MAE: ~7.06
    - MAPE: ~4.42%
    - R²: ~0.979

    ## Input Data

    The API accepts quote data with various optional fields. Missing values are automatically imputed
    using median values (for numeric) or mode values (for categorical) from the training data.
    """,
    version="1.0.0",
    tags_metadata=[
        {
            "name": "health",
            "description": "Health check endpoints for monitoring service availability.",
        },
        {
            "name": "predictions",
            "description": "Price prediction endpoints. Submit quote data to get predicted unit prices.",
        },
    ],
)

# Load model (pipeline includes imputation and feature engineering)
model = mlflow.sklearn.load_model(MODEL_URI)


@app.get(
    "/healthz",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["health"],
    summary="Health check endpoint",
    description="Returns the health status of the API service.",
    response_description="Service is healthy and ready to accept requests",
)
def healthz() -> HealthResponse:
    """
    Health check endpoint.

    Use this endpoint to verify that the API service is running and ready to accept requests.
    Returns a simple status message indicating the service is operational.

    Returns:
        HealthResponse: Response containing service status
    """
    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=list[PredictionResponse],
    status_code=status.HTTP_200_OK,
    tags=["predictions"],
    summary="Predict unit prices for quotes",
    description="""
    Predicts the target unit price (in Russian Rubles) for one or more quotes.

    This endpoint accepts a list of quote objects and returns predictions for each.
    The model automatically handles:
    - Missing value imputation
    - Feature engineering (area, perimeter, volume, weight estimates, etc.)
    - Categorical encoding

    **Input Requirements:**
    - At minimum: `rfq_id`, `customer_tier`, `material_cost_rub`, `labor_minutes_per_unit`, `labor_cost_rub`
    - Other fields are optional and will be imputed if missing

    **Batch Processing:**
    - Submit multiple quotes in a single request for efficient batch processing
    - Empty batch returns an empty list
    """,
    response_description="List of predictions, one for each input quote",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": [
                        {"target_unit_price_rub_pred": 156.78},
                        {"target_unit_price_rub_pred": 234.56},
                    ]
                }
            },
        },
        422: {
            "description": "Validation error - invalid input data",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {"detail": "Validation error: field required"}
                }
            },
        },
    },
)
def predict(quotes: list[BaseQuote]) -> list[PredictionResponse]:
    """
    Predict target unit prices for a batch of quotes.

    Args:
        quotes: List of quote objects containing quote details. Each quote must include:
            - `rfq_id` (required): Request for quote identifier
            - `customer_tier` (required): Customer tier (A, B, or C)
            - `material_cost_rub` (required): Material cost in RUB
            - `labor_minutes_per_unit` (required): Labor minutes per unit
            - `labor_cost_rub` (required): Labor cost in RUB
            - Other fields are optional and will be imputed if missing

    Returns:
        List of prediction responses, each containing:
            - `target_unit_price_rub_pred`: Predicted unit price in Russian Rubles

    Raises:
        HTTPException: 422 if input validation fails

    Example:
        ```python
        quotes = [
            {
                "rfq_id": 1,
                "customer_tier": "A",
                "material": "steel",
                "thickness_mm": 5.0,
                "length_mm": 1000.0,
                "width_mm": 500.0,
                "material_cost_rub": 1000.0,
                "labor_minutes_per_unit": 10.0,
                "labor_cost_rub": 500.0
            }
        ]
        response = client.post("/predict", json=quotes)
        # Returns: [{"target_unit_price_rub_pred": 156.78}]
        ```
    """
    try:
        df = pd.DataFrame([q.model_dump() for q in quotes])
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        ) from e

    # Handle empty batch
    if len(df) == 0:
        return []

    # Use the model pipeline directly (it includes imputation and feature engineering)
    yhat = model.predict(df)
    return [
        PredictionResponse(target_unit_price_rub_pred=float(v)) for v in yhat
    ]
