import math
from enum import Enum

from pydantic import BaseModel, confloat, conint


class CustomerTier(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class Material(str, Enum):
    aluminum = "aluminum"
    stainless = "stainless"
    steel = "steel"


class Route(str, Enum):
    laser_cut = "laser_cut"
    plasma_cut = "plasma_cut"
    waterjet_cut = "waterjet_cut"


class Tolerance(str, Enum):
    standard = "standard"
    precise = "precise"
    high_precision = "high_precision"


class SurfaceFinish(str, Enum):
    none = "none"
    paint = "paint"
    anodized = "anodized"
    galvanized = "galvanized"


class Coating(str, Enum):
    none = "none"
    powder = "powder"
    zinc = "zinc"


class BaseQuote(BaseModel):
    rfq_id: conint(ge=1)
    # Fields that will be imputed - make Optional to allow missing values
    thickness_mm: confloat(gt=0, le=100) | None = None
    length_mm: confloat(gt=0, le=5000) | None = None
    width_mm: confloat(gt=0, le=5000) | None = None
    holes_count: conint(ge=0, le=200) | None = None
    bends_count: conint(ge=0, le=200) | None = None
    weld_length_mm: confloat(ge=0, le=100000) | None = None
    cut_length_mm: confloat(ge=0, le=100000) | None = None
    part_weight_kg: confloat(ge=0, le=1000) | None = None
    qty: conint(ge=1, le=100000) | None = None
    due_days: conint(ge=0, le=365) | None = None
    engineer_score: confloat(ge=-10, le=10) | None = None
    customer_tier: CustomerTier
    material: Material | None = None
    route: Route | None = None
    tolerance: Tolerance | None = None
    surface_finish: SurfaceFinish | None = None
    coating: Coating | None = None
    material_cost_rub: confloat(ge=0, le=1_000_000)
    labor_minutes_per_unit: confloat(ge=0, le=10_000)
    labor_cost_rub: confloat(ge=0, le=1_000_000)  # LEAK (excluded from features)


class TrainingQuote(BaseQuote):
    unit_price_rub: confloat(ge=0, le=1_000_000)
    target_labor_min: confloat(ge=0, le=10000) | None = None
    target_unit_price_rub: confloat(ge=0, le=1000000) | None = None


def _convert_nan_to_none(rec: dict) -> dict:
    """Convert NaN values to None for Optional fields, and handle NaN in required fields."""
    rec_clean = {}
    # Fields that are Optional in the schema
    optional_fields = {
        "engineer_score",
        "material",
        "route",
        "tolerance",
        "surface_finish",
        "coating",
        "target_labor_min",
        "target_unit_price_rub",
    }
    # Fields that will be imputed by ImputeAndTypes (can have NaN)
    imputable_fields = {
        "thickness_mm",
        "length_mm",
        "width_mm",
        "holes_count",
        "bends_count",
        "weld_length_mm",
        "cut_length_mm",
        "part_weight_kg",
        "qty",
        "due_days",
    }

    for key, value in rec.items():
        # Check if value is NaN
        if isinstance(value, float) and math.isnan(value):
            if key in optional_fields:
                # Convert NaN to None for Optional fields
                rec_clean[key] = None
            elif key in imputable_fields:
                # For imputable required fields, keep NaN (will be handled by imputation)
                # But we need to make it pass validation - use a sentinel or make field Optional
                # Actually, we should make these Optional or use a custom validator
                # For now, convert to None and make fields Optional
                rec_clean[key] = None
            else:
                # For other fields, keep NaN but it will likely fail validation
                rec_clean[key] = value
        else:
            rec_clean[key] = value
    return rec_clean


def validate_training_batch(records: list[dict]) -> list[dict]:
    out, errors = [], []
    for i, rec in enumerate(records):
        try:
            # Pre-process to convert NaN to None for Optional/imputable fields
            rec_clean = _convert_nan_to_none(rec)
            obj = TrainingQuote.model_validate(rec_clean)
            out.append(obj.model_dump())
        except Exception as e:
            errors.append((i, str(e)))
    if errors:
        detail = "\n".join([f"Row {i}: {msg}" for i, msg in errors[:20]])
        raise ValueError(f"Validation failed for {len(errors)} rows.\n" + detail)
    return out
