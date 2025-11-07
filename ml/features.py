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

        # Ensure numeric columns are properly typed (convert object dtypes with NaN to float)
        numeric_cols = ["length_mm", "width_mm", "thickness_mm", "holes_count", "bends_count",
                       "weld_length_mm", "cut_length_mm", "part_weight_kg", "qty", "due_days",
                       "engineer_score", "material_cost_rub", "labor_minutes_per_unit"]
        for col in numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        # Build all new features in a dictionary to avoid DataFrame fragmentation
        new_features = {}

        # Basic geometric features
        new_features["area_mm2"] = X["length_mm"] * X["width_mm"]
        new_features["perimeter_mm"] = 2 * (X["length_mm"] + X["width_mm"])
        new_features["volume_mm3"] = new_features["area_mm2"] * X["thickness_mm"]

        # Material density and weight features
        dens = X["material"].astype(str).map(DENSITY).fillna(DENSITY["steel"])
        new_features["weight_est_kg"] = new_features["volume_mm3"] * dens
        new_features["weight_gap"] = X["part_weight_kg"] - new_features["weight_est_kg"]
        new_features["weight_gap_pct"] = new_features["weight_gap"] / (new_features["weight_est_kg"] + 1e-6)

        # Aspect ratio and shape features
        new_features["aspect_ratio"] = X["length_mm"] / (X["width_mm"] + 1e-6)
        new_features["max_dim"] = X[["length_mm", "width_mm"]].max(axis=1)
        new_features["min_dim"] = X[["length_mm", "width_mm"]].min(axis=1)
        new_features["dim_ratio"] = new_features["max_dim"] / (new_features["min_dim"] + 1e-6)

        # Complexity features
        new_features["total_features"] = X["holes_count"] + X["bends_count"]
        new_features["features_per_area"] = new_features["total_features"] / new_features["area_mm2"].replace(0, np.nan)
        new_features["features_per_volume"] = new_features["total_features"] / new_features["volume_mm3"].replace(0, np.nan)
        new_features["complexity_score"] = (
            X["holes_count"] * 0.5 + X["bends_count"] * 1.0 + (X["weld_length_mm"] > 0).astype(int) * 2.0
        )

        # Cutting and welding features
        new_features["cut_ratio"] = X["cut_length_mm"] / new_features["perimeter_mm"].replace(0, np.nan)
        new_features["weld_per_area"] = X["weld_length_mm"] / new_features["area_mm2"].replace(0, np.nan)
        new_features["cut_vs_area"] = X["cut_length_mm"] / new_features["area_mm2"].replace(0, np.nan)
        new_features["cut_per_volume"] = X["cut_length_mm"] / new_features["volume_mm3"].replace(0, np.nan)
        new_features["weld_per_volume"] = X["weld_length_mm"] / new_features["volume_mm3"].replace(0, np.nan)
        new_features["has_welding"] = (X["weld_length_mm"] > 0).astype("int8")
        new_features["has_cutting"] = (X["cut_length_mm"] > 0).astype("int8")

        # Thickness features
        new_features["thickness_to_length"] = X["thickness_mm"] / (X["length_mm"] + 1e-6)
        new_features["thickness_to_width"] = X["thickness_mm"] / (X["width_mm"] + 1e-6)
        new_features["thickness_to_area"] = X["thickness_mm"] / new_features["area_mm2"].replace(0, np.nan)

        # Quantity and time features
        new_features["rush"] = (X["due_days"] <= 7).astype("int8")
        new_features["urgent"] = (X["due_days"] <= 3).astype("int8")
        new_features["normal_lead"] = (X["due_days"] > 14).astype("int8")
        bins = [-np.inf, 1, 5, 20, 50, np.inf]
        labels = ["1", "2-5", "6-20", "21-50", "51+"]
        new_features["qty_bucket"] = pd.cut(X["qty"], bins=bins, labels=labels).astype("category")

        # Cost and labor features
        total_cost = X["material_cost_rub"] + 1e-6
        new_features["mat_share"] = X["material_cost_rub"] / total_cost
        new_features["labor_per_qty"] = X["labor_minutes_per_unit"] / X["qty"].replace(0, np.nan)
        new_features["labor_per_area"] = X["labor_minutes_per_unit"] / new_features["area_mm2"].replace(0, np.nan)
        new_features["labor_per_volume"] = X["labor_minutes_per_unit"] / new_features["volume_mm3"].replace(0, np.nan)
        new_features["cost_per_area"] = X["material_cost_rub"] / new_features["area_mm2"].replace(0, np.nan)
        new_features["cost_per_volume"] = X["material_cost_rub"] / new_features["volume_mm3"].replace(0, np.nan)
        new_features["cost_per_weight"] = X["material_cost_rub"] / (X["part_weight_kg"] + 1e-6)

        # Interaction features
        new_features["qty_area"] = X["qty"] * new_features["area_mm2"]
        new_features["qty_volume"] = X["qty"] * new_features["volume_mm3"]
        new_features["qty_complexity"] = X["qty"] * new_features["complexity_score"]
        new_features["thickness_material_cost"] = X["thickness_mm"] * X["material_cost_rub"]
        new_features["area_material_cost"] = new_features["area_mm2"] * X["material_cost_rub"]
        new_features["volume_material_cost"] = new_features["volume_mm3"] * X["material_cost_rub"]
        new_features["labor_qty"] = X["labor_minutes_per_unit"] * X["qty"]
        new_features["due_qty"] = X["due_days"] * X["qty"]

        # Polynomial features for key dimensions
        new_features["area_squared"] = new_features["area_mm2"] ** 2
        new_features["volume_squared"] = new_features["volume_mm3"] ** 2
        new_features["thickness_squared"] = X["thickness_mm"] ** 2
        new_features["qty_squared"] = X["qty"] ** 2

        # Log transformations for skewed features (add small epsilon to avoid log(0))
        new_features["log_area"] = np.log1p(new_features["area_mm2"])
        new_features["log_volume"] = np.log1p(new_features["volume_mm3"])
        new_features["log_qty"] = np.log1p(X["qty"])
        new_features["log_material_cost"] = np.log1p(X["material_cost_rub"])
        new_features["log_labor_minutes"] = np.log1p(X["labor_minutes_per_unit"])
        new_features["log_cut_length"] = np.log1p(X["cut_length_mm"])
        new_features["log_weld_length"] = np.log1p(X["weld_length_mm"])

        # Engineer score features
        new_features["engineer_score_positive"] = (X["engineer_score"] > 0).astype("int8")
        new_features["engineer_score_negative"] = (X["engineer_score"] < 0).astype("int8")
        new_features["engineer_score_abs"] = np.abs(X["engineer_score"])
        new_features["engineer_score_high"] = (X["engineer_score"] > 1.0).astype("int8")
        new_features["engineer_score_low"] = (X["engineer_score"] < -1.0).astype("int8")

        # Additional binning features for continuous variables
        # Thickness bins
        thickness_bins = [-np.inf, 2, 5, 10, np.inf]
        thickness_labels = ["thin", "medium", "thick", "very_thick"]
        new_features["thickness_bucket"] = pd.cut(
            X["thickness_mm"], bins=thickness_bins, labels=thickness_labels
        ).astype("category")

        # Area bins
        area_bins = [-np.inf, 100000, 500000, 2000000, np.inf]
        area_labels = ["small", "medium", "large", "very_large"]
        new_features["area_bucket"] = pd.cut(
            new_features["area_mm2"], bins=area_bins, labels=area_labels
        ).astype("category")

        # Volume bins
        volume_bins = [-np.inf, 1000000, 5000000, 20000000, np.inf]
        volume_labels = ["small_vol", "medium_vol", "large_vol", "very_large_vol"]
        new_features["volume_bucket"] = pd.cut(
            new_features["volume_mm3"], bins=volume_bins, labels=volume_labels
        ).astype("category")

        # Due days bins
        due_bins = [-np.inf, 3, 7, 14, 30, np.inf]
        due_labels = ["urgent", "rush", "normal", "extended", "long"]
        new_features["due_bucket"] = pd.cut(
            X["due_days"], bins=due_bins, labels=due_labels
        ).astype("category")

        # Material-specific features
        new_features["is_aluminum"] = (X["material"].astype(str) == "aluminum").astype("int8")
        new_features["is_stainless"] = (X["material"].astype(str) == "stainless").astype("int8")
        new_features["is_steel"] = (X["material"].astype(str) == "steel").astype("int8")
        new_features["material_density"] = dens

        # Route-specific features
        new_features["is_laser_cut"] = (X["route"].astype(str) == "laser_cut").astype("int8")
        new_features["is_plasma_cut"] = (X["route"].astype(str) == "plasma_cut").astype("int8")
        new_features["is_waterjet_cut"] = (X["route"].astype(str) == "waterjet_cut").astype("int8")

        # Tolerance features
        new_features["is_high_precision"] = (
            X["tolerance"].astype(str) == "high_precision"
        ).astype("int8")
        new_features["is_precise"] = (X["tolerance"].astype(str) == "precise").astype("int8")
        new_features["is_standard"] = (X["tolerance"].astype(str) == "standard").astype("int8")

        # Surface finish features
        new_features["has_paint"] = (X["surface_finish"].astype(str) == "paint").astype("int8")
        new_features["has_anodized"] = (
            X["surface_finish"].astype(str) == "anodized"
        ).astype("int8")
        new_features["has_galvanized"] = (
            X["surface_finish"].astype(str) == "galvanized"
        ).astype("int8")
        new_features["has_surface_finish"] = (
            X["surface_finish"].astype(str) != "none"
        ).astype("int8")

        # Coating features
        new_features["has_powder"] = (X["coating"].astype(str) == "powder").astype("int8")
        new_features["has_zinc"] = (X["coating"].astype(str) == "zinc").astype("int8")
        new_features["has_coating"] = (X["coating"].astype(str) != "none").astype("int8")

        # Customer tier features
        new_features["is_tier_a"] = (X["customer_tier"].astype(str) == "A").astype("int8")
        new_features["is_tier_b"] = (X["customer_tier"].astype(str) == "B").astype("int8")
        new_features["is_tier_c"] = (X["customer_tier"].astype(str) == "C").astype("int8")

        # Advanced geometric features
        new_features["diagonal_mm"] = np.sqrt(X["length_mm"] ** 2 + X["width_mm"] ** 2)
        new_features["surface_area_mm2"] = 2 * (
            new_features["area_mm2"] + X["length_mm"] * X["thickness_mm"] + X["width_mm"] * X["thickness_mm"]
        )
        new_features["compactness"] = 4 * np.pi * new_features["area_mm2"] / (new_features["perimeter_mm"] ** 2 + 1e-6)
        new_features["rectangularity"] = new_features["area_mm2"] / (new_features["max_dim"] * new_features["min_dim"] + 1e-6)

        # Efficiency and productivity features
        new_features["labor_efficiency"] = new_features["area_mm2"] / (X["labor_minutes_per_unit"] + 1e-6)
        new_features["cut_efficiency"] = new_features["area_mm2"] / (X["cut_length_mm"] + 1e-6)
        new_features["weld_efficiency"] = new_features["area_mm2"] / (X["weld_length_mm"] + 1e-6)
        new_features["features_efficiency"] = new_features["total_features"] / (X["labor_minutes_per_unit"] + 1e-6)

        # Cost efficiency features
        new_features["cost_per_feature"] = X["material_cost_rub"] / (new_features["total_features"] + 1e-6)
        new_features["cost_per_hole"] = X["material_cost_rub"] / (X["holes_count"] + 1e-6)
        new_features["cost_per_bend"] = X["material_cost_rub"] / (X["bends_count"] + 1e-6)
        new_features["labor_cost_per_area"] = (
            X["labor_minutes_per_unit"] * 100
        ) / new_features["area_mm2"].replace(0, np.nan)  # Approximate labor cost proxy

        # Advanced interaction features
        new_features["qty_due_interaction"] = X["qty"] / (X["due_days"] + 1e-6)
        new_features["area_thickness_interaction"] = new_features["area_mm2"] * X["thickness_mm"]
        new_features["volume_qty_interaction"] = new_features["volume_mm3"] * X["qty"]
        new_features["complexity_qty_interaction"] = new_features["complexity_score"] * X["qty"]
        new_features["labor_complexity_interaction"] = (
            X["labor_minutes_per_unit"] * new_features["complexity_score"]
        )
        new_features["cost_complexity_interaction"] = (
            X["material_cost_rub"] * new_features["complexity_score"]
        )
        new_features["engineer_complexity_interaction"] = (
            X["engineer_score"] * new_features["complexity_score"]
        )
        new_features["due_complexity_interaction"] = X["due_days"] * new_features["complexity_score"]

        # Material-cost interactions
        new_features["aluminum_cost"] = new_features["is_aluminum"] * X["material_cost_rub"]
        new_features["stainless_cost"] = new_features["is_stainless"] * X["material_cost_rub"]
        new_features["steel_cost"] = new_features["is_steel"] * X["material_cost_rub"]

        # Route-labor interactions
        new_features["laser_labor"] = new_features["is_laser_cut"] * X["labor_minutes_per_unit"]
        new_features["plasma_labor"] = new_features["is_plasma_cut"] * X["labor_minutes_per_unit"]
        new_features["waterjet_labor"] = new_features["is_waterjet_cut"] * X["labor_minutes_per_unit"]

        # Tolerance-labor interactions
        new_features["high_precision_labor"] = new_features["is_high_precision"] * X["labor_minutes_per_unit"]
        new_features["precise_labor"] = new_features["is_precise"] * X["labor_minutes_per_unit"]

        # Surface finish-cost interactions
        new_features["paint_cost"] = new_features["has_paint"] * X["material_cost_rub"]
        new_features["anodized_cost"] = new_features["has_anodized"] * X["material_cost_rub"]
        new_features["galvanized_cost"] = new_features["has_galvanized"] * X["material_cost_rub"]

        # Coating-cost interactions
        new_features["powder_cost"] = new_features["has_powder"] * X["material_cost_rub"]
        new_features["zinc_cost"] = new_features["has_zinc"] * X["material_cost_rub"]

        # Customer tier interactions
        new_features["tier_a_qty"] = new_features["is_tier_a"] * X["qty"]
        new_features["tier_b_qty"] = new_features["is_tier_b"] * X["qty"]
        new_features["tier_c_qty"] = new_features["is_tier_c"] * X["qty"]

        # More polynomial features
        new_features["thickness_cubed"] = X["thickness_mm"] ** 3
        new_features["qty_cubed"] = X["qty"] ** 3
        new_features["area_cubed"] = new_features["area_mm2"] ** 1.5
        new_features["volume_cubed"] = new_features["volume_mm3"] ** 1.5

        # Cross-ratio features
        new_features["thickness_area_ratio"] = X["thickness_mm"] / new_features["area_mm2"].replace(0, np.nan)
        new_features["qty_area_ratio"] = X["qty"] / new_features["area_mm2"].replace(0, np.nan)
        new_features["qty_volume_ratio"] = X["qty"] / new_features["volume_mm3"].replace(0, np.nan)
        new_features["labor_area_ratio"] = X["labor_minutes_per_unit"] / new_features["area_mm2"].replace(0, np.nan)
        new_features["cost_thickness_ratio"] = X["material_cost_rub"] / (X["thickness_mm"] + 1e-6)
        new_features["cost_qty_ratio"] = X["material_cost_rub"] / (X["qty"] + 1e-6)

        # Weight-based features
        new_features["weight_per_area"] = X["part_weight_kg"] / new_features["area_mm2"].replace(0, np.nan)
        new_features["weight_per_volume"] = X["part_weight_kg"] / new_features["volume_mm3"].replace(0, np.nan)
        new_features["cost_per_kg"] = X["material_cost_rub"] / (X["part_weight_kg"] + 1e-6)
        new_features["labor_per_kg"] = X["labor_minutes_per_unit"] / (X["part_weight_kg"] + 1e-6)

        # Density-normalized features
        new_features["normalized_weight"] = X["part_weight_kg"] / (dens + 1e-6)
        new_features["normalized_cost"] = X["material_cost_rub"] / (dens + 1e-6)

        # Feature count ratios
        new_features["holes_to_bends_ratio"] = X["holes_count"] / (X["bends_count"] + 1e-6)
        new_features["bends_to_holes_ratio"] = X["bends_count"] / (X["holes_count"] + 1e-6)
        new_features["cut_to_weld_ratio"] = X["cut_length_mm"] / (X["weld_length_mm"] + 1e-6)
        new_features["weld_to_cut_ratio"] = X["weld_length_mm"] / (X["cut_length_mm"] + 1e-6)

        # More log transformations
        new_features["log_thickness"] = np.log1p(X["thickness_mm"])
        new_features["log_perimeter"] = np.log1p(new_features["perimeter_mm"])
        new_features["log_max_dim"] = np.log1p(new_features["max_dim"])
        new_features["log_min_dim"] = np.log1p(new_features["min_dim"])
        new_features["log_diagonal"] = np.log1p(new_features["diagonal_mm"])
        new_features["log_due_days"] = np.log1p(X["due_days"])
        new_features["log_total_features"] = np.log1p(new_features["total_features"])

        # Square root transformations for highly skewed features
        new_features["sqrt_area"] = np.sqrt(new_features["area_mm2"])
        new_features["sqrt_volume"] = np.sqrt(new_features["volume_mm3"])
        new_features["sqrt_qty"] = np.sqrt(X["qty"])
        new_features["sqrt_material_cost"] = np.sqrt(X["material_cost_rub"])

        # Convert dictionary to DataFrame and concatenate with original
        new_features_df = pd.DataFrame(new_features, index=X.index)

        # Replace inf with nan, then fillna only for numeric columns
        new_features_df = new_features_df.replace([np.inf, -np.inf], np.nan)
        num_cols = new_features_df.select_dtypes(include=[np.number]).columns
        new_features_df[num_cols] = new_features_df[num_cols].fillna(0)

        # Concatenate all new features at once to avoid fragmentation
        X = pd.concat([X, new_features_df], axis=1)

        # Ensure categorical buckets are properly typed
        for bucket_col in ["qty_bucket", "thickness_bucket", "area_bucket", "volume_bucket", "due_bucket"]:
            if bucket_col in X.columns:
                X[bucket_col] = X[bucket_col].astype("category")

        # Remove features with zero importance (identified from feature importance analysis)
        # These features don't contribute to model performance
        # Note: We keep original features like due_days and cut_length_mm even if they have zero importance
        # because they're used to create other important features (rush, log_due_days, cut_ratio, etc.)
        zero_importance_features = {
            # Derived features with zero importance (safe to remove)
            "has_cutting",
            "has_welding",
            "is_precise",
            "is_high_precision",
            "is_standard",
            "has_anodized",
            "is_laser_cut",
            "is_plasma_cut",
            "is_waterjet_cut",
            "engineer_score_positive",
            "thickness_bucket",
            "engineer_score_low",
            "material_density",
            "is_steel",
            "is_stainless",
            "is_aluminum",
            "area_bucket",
            "engineer_score_high",
            "qty_bucket",
            "normal_lead",
            "urgent",
            "volume_squared",
            "log_qty",
            "qty_squared",
            "log_volume",
            "is_tier_c",
            "has_coating",
            "is_tier_a",
            "is_tier_b",
            "has_zinc",
            "has_galvanized",
            "has_surface_finish",
            "has_powder",
            "has_paint",
            "plasma_labor",
            "laser_labor",
            "steel_cost",
            "stainless_cost",
            "aluminum_cost",
            "area_thickness_interaction",
            "tier_b_qty",
            "tier_a_qty",
            "zinc_cost",
            "powder_cost",
            "galvanized_cost",
            "anodized_cost",
            "paint_cost",
            "precise_labor",
            "high_precision_labor",
            "waterjet_labor",
            "volume_cubed",
            "tier_c_qty",
            "qty_cubed",
            "sqrt_area",
            "sqrt_volume",
            "sqrt_qty",
        }

        # Drop zero-importance features that exist in the DataFrame
        features_to_drop = [f for f in zero_importance_features if f in X.columns]
        if features_to_drop:
            X = X.drop(columns=features_to_drop)

        return X
