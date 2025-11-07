import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DENSITY = {"aluminum": 2.70e-6, "steel": 7.85e-6, "stainless": 8.00e-6}
CAT_COLS = ["customer_tier", "material", "route", "tolerance", "surface_finish", "coating"]
DROP_COLS = ["rfq_id", "target_labor_min", "unit_price_rub", "labor_cost_rub"]
LABEL = "target_unit_price_rub"


class ImputeAndTypes(BaseEstimator, TransformerMixin):
    def __init__(self, fill_unknown_category: bool = True):
        self.fill_unknown_category = fill_unknown_category
        self.num_medians_ = {}
        self.cat_modes_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        for c in X.select_dtypes(include="number").columns:
            if c == LABEL:
                continue
            self.num_medians_[c] = X[c].median()
        for c in CAT_COLS:
            if c in X.columns:
                mode = X[c].mode(dropna=True)
                self.cat_modes_[c] = mode.iloc[0] if not mode.empty else None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in DROP_COLS:
            if c in X.columns:
                X = X.drop(columns=c)
        for c, med in self.num_medians_.items():
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)
        for c in CAT_COLS:
            if c in X.columns:
                # Fill missing values BEFORE converting to categorical
                if self.fill_unknown_category and self.cat_modes_.get(c) is not None:
                    mode_value = self.cat_modes_[c]
                    # Check if column is already categorical
                    if isinstance(X[c].dtype, pd.CategoricalDtype):
                        # Add mode to categories if not present
                        current_cats = list(X[c].cat.categories)
                        if mode_value not in current_cats:
                            X[c] = X[c].cat.add_categories([mode_value])
                        X[c] = X[c].fillna(mode_value)
                    else:
                        # Fill missing values first, then convert to categorical
                        X[c] = X[c].fillna(mode_value)
                        # Convert to categorical, ensuring the mode is included in categories
                        # This handles the case where mode from training is not in current data
                        unique_vals = list(X[c].unique())
                        if mode_value not in unique_vals:
                            unique_vals.append(mode_value)
                        X[c] = pd.Categorical(X[c], categories=unique_vals)
                else:
                    X[c] = X[c].astype("category")
        return X


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["area_mm2"] = X["length_mm"] * X["width_mm"]
        X["perimeter_mm"] = 2 * (X["length_mm"] + X["width_mm"])
        X["volume_mm3"] = X["area_mm2"] * X["thickness_mm"]
        dens = X["material"].astype(str).map(DENSITY).fillna(DENSITY["steel"])
        X["weight_est_kg"] = X["volume_mm3"] * dens
        X["weight_gap"] = X["part_weight_kg"] - X["weight_est_kg"]
        X["features_per_area"] = (X["holes_count"] + X["bends_count"]) / X["area_mm2"].replace(
            0, np.nan
        )
        X["cut_ratio"] = X["cut_length_mm"] / X["perimeter_mm"].replace(0, np.nan)
        X["weld_per_area"] = X["weld_length_mm"] / X["area_mm2"].replace(0, np.nan)
        X["rush"] = (X["due_days"] <= 7).astype("int8")
        bins = [-np.inf, 1, 5, 20, 50, np.inf]
        labels = ["1", "2-5", "6-20", "21-50", "51+"]
        X["qty_bucket"] = pd.cut(X["qty"], bins=bins, labels=labels).astype("category")
        total_cost = X["material_cost_rub"] + 1e-6
        X["mat_share"] = X["material_cost_rub"] / total_cost
        X["labor_per_qty"] = X["labor_minutes_per_unit"] / X["qty"].replace(0, np.nan)
        X["cut_vs_area"] = X["cut_length_mm"] / X["area_mm2"].replace(0, np.nan)

        # Replace inf with nan, then fillna only for numeric columns
        # Categorical columns should already be handled and can't accept arbitrary fill values
        X = X.replace([np.inf, -np.inf], np.nan)
        # Fill numeric columns only (categorical columns are already handled)
        num_cols = X.select_dtypes(include=[np.number]).columns
        X[num_cols] = X[num_cols].fillna(0)
        # Ensure qty_bucket is categorical (in case it was created)
        if "qty_bucket" in X.columns:
            X["qty_bucket"] = X["qty_bucket"].astype("category")
        return X
