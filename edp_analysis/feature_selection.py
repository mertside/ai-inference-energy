"""
Feature Selection and Engineering Module for EDP Analysis

This module provides advanced feature selection and engineering capabilities
inspired by the FGCS 2023 methodology, focusing on GPU power modeling features
that are most relevant for energy-delay optimization.

Key Features:
- FGCS-inspired feature engineering (FP activity, DRAM activity, clock frequencies)
- Statistical feature selection methods
- Correlation-based feature filtering
- GPU-specific feature validation
- Feature importance analysis for EDP optimization
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_regression,
    mutual_info_regression,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)


class FGCSFeatureEngineering:
    """
    Feature engineering based on FGCS 2023 methodology.

    Implements the feature extraction and transformation approach from the
    "Energy-efficient DVFS scheduling for mixed-criticality systems" paper.
    """

    def __init__(self, gpu_type: str = "V100"):
        """
        Initialize FGCS feature engineering.

        Args:
            gpu_type: Target GPU type ('V100', 'A100', 'H100')
        """
        self.gpu_type = gpu_type
        self.feature_names = []
        self.scaler = StandardScaler()

        # FGCS model coefficients for validation
        self.fgcs_coefficients = {
            "intercept": -1.0318354343254663,
            "fp_coeff": 0.84864,
            "dram_coeff": 0.09749,
            "clock_coeff": 0.77006,
        }

        logger.info(f"FGCS feature engineering initialized for {gpu_type}")

    def extract_fgcs_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract core FGCS features from profiling data.

        Args:
            df: DataFrame with profiling data

        Returns:
            DataFrame with FGCS features
        """
        features_df = df.copy()

        # Core FGCS features
        if "fp_activity" not in features_df.columns:
            logger.warning("FP activity not found, estimating from available data")
            features_df["fp_activity"] = self._estimate_fp_activity(features_df)

        if "dram_activity" not in features_df.columns:
            logger.warning("DRAM activity not found, estimating from available data")
            features_df["dram_activity"] = self._estimate_dram_activity(features_df)

        # Clock frequency transformations
        if "sm_clock" in features_df.columns:
            features_df["n_sm_app_clock"] = np.log1p(features_df["sm_clock"])
        elif "frequency" in features_df.columns:
            features_df["sm_clock"] = features_df["frequency"]
            features_df["n_sm_app_clock"] = np.log1p(features_df["frequency"])

        # Power transformations (if available)
        if "power" in features_df.columns:
            features_df["n_power_usage"] = np.log1p(features_df["power"])

        # Runtime transformations (if available)
        if "execution_time" in features_df.columns:
            features_df["n_run_time"] = np.log1p(features_df["execution_time"])
        elif "runtime" in features_df.columns:
            features_df["execution_time"] = features_df["runtime"]
            features_df["n_run_time"] = np.log1p(features_df["runtime"])

        # Energy calculations
        if "power" in features_df.columns and "execution_time" in features_df.columns:
            features_df["energy"] = features_df["power"] * features_df["execution_time"]
            features_df["n_energy"] = np.log1p(features_df["energy"])

        # Store feature names
        self.feature_names = [
            col for col in features_df.columns if col.startswith(("fp_", "dram_", "n_"))
        ]

        logger.info(f"Extracted {len(self.feature_names)} FGCS features")
        return features_df

    def _estimate_fp_activity(self, df: pd.DataFrame) -> np.ndarray:
        """Estimate FP activity from available metrics."""
        if "gpu_utilization" in df.columns:
            # Use GPU utilization as proxy for FP activity
            return np.clip(df["gpu_utilization"] / 100.0, 0.1, 0.8)
        elif "sm_clock" in df.columns or "frequency" in df.columns:
            # Estimate based on frequency (higher freq -> likely higher FP activity)
            freq_col = "sm_clock" if "sm_clock" in df.columns else "frequency"
            normalized_freq = (df[freq_col] - df[freq_col].min()) / (
                df[freq_col].max() - df[freq_col].min()
            )
            return 0.2 + 0.4 * normalized_freq  # Range: 0.2 to 0.6
        else:
            # Default moderate FP activity
            return np.full(len(df), 0.3)

    def _estimate_dram_activity(self, df: pd.DataFrame) -> np.ndarray:
        """Estimate DRAM activity from available metrics."""
        if "memory_utilization" in df.columns:
            # Use memory utilization as proxy for DRAM activity
            return np.clip(df["memory_utilization"] / 100.0, 0.05, 0.4)
        elif "fp_activity" in df.columns:
            # DRAM activity typically correlates with FP activity but is lower
            return np.clip(df["fp_activity"] * 0.5, 0.05, 0.3)
        else:
            # Default low DRAM activity
            return np.full(len(df), 0.15)

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features based on FGCS methodology.

        Args:
            df: DataFrame with base features

        Returns:
            DataFrame with interaction features added
        """
        features_df = df.copy()

        # FGCS-specific interactions
        if (
            "fp_activity" in features_df.columns
            and "n_sm_app_clock" in features_df.columns
        ):
            features_df["fp_x_clock"] = (
                features_df["fp_activity"] * features_df["n_sm_app_clock"]
            )
            features_df["fp_x_clock_sq"] = features_df["fp_activity"] * (
                features_df["n_sm_app_clock"] ** 2
            )

        if (
            "dram_activity" in features_df.columns
            and "n_sm_app_clock" in features_df.columns
        ):
            features_df["dram_x_clock"] = (
                features_df["dram_activity"] * features_df["n_sm_app_clock"]
            )

        # Polynomial features for core activities
        if "fp_activity" in features_df.columns:
            features_df["fp_activity_sq"] = features_df["fp_activity"] ** 2
            features_df["fp_activity_cb"] = features_df["fp_activity"] ** 3

        if "dram_activity" in features_df.columns:
            features_df["dram_activity_sq"] = features_df["dram_activity"] ** 2

        # Clock frequency powers
        if "n_sm_app_clock" in features_df.columns:
            features_df["n_sm_app_clock_sq"] = features_df["n_sm_app_clock"] ** 2

        logger.info(
            f"Created {len(features_df.columns) - len(df.columns)} interaction features"
        )
        return features_df

    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate feature quality and FGCS compatibility.

        Args:
            df: DataFrame with features

        Returns:
            Validation results dictionary
        """
        validation_results = {
            "total_features": len(df.columns),
            "fgcs_core_features": [],
            "missing_fgcs_features": [],
            "feature_ranges": {},
            "data_quality": {},
            "recommendations": [],
        }

        # Check for core FGCS features
        core_features = ["fp_activity", "dram_activity", "n_sm_app_clock"]
        for feature in core_features:
            if feature in df.columns:
                validation_results["fgcs_core_features"].append(feature)

                # Check feature ranges
                if feature == "fp_activity":
                    feature_range = (df[feature].min(), df[feature].max())
                    validation_results["feature_ranges"][feature] = feature_range
                    if feature_range[0] < 0 or feature_range[1] > 1:
                        validation_results["recommendations"].append(
                            f"FP activity should be in range [0, 1], found {feature_range}"
                        )

                elif feature == "dram_activity":
                    feature_range = (df[feature].min(), df[feature].max())
                    validation_results["feature_ranges"][feature] = feature_range
                    if feature_range[0] < 0 or feature_range[1] > 1:
                        validation_results["recommendations"].append(
                            f"DRAM activity should be in range [0, 1], found {feature_range}"
                        )

                elif feature == "n_sm_app_clock":
                    feature_range = (df[feature].min(), df[feature].max())
                    validation_results["feature_ranges"][feature] = feature_range
                    # Check if log-transformed clock frequencies are reasonable
                    if (
                        feature_range[0] < 6 or feature_range[1] > 8
                    ):  # log(400) ~ 6, log(1500) ~ 7.3
                        validation_results["recommendations"].append(
                            f"Log clock frequency range unusual: {feature_range}, check frequency units"
                        )
            else:
                validation_results["missing_fgcs_features"].append(feature)

        # Data quality checks
        validation_results["data_quality"]["total_samples"] = len(df)
        validation_results["data_quality"]["missing_values"] = (
            df.isnull().sum().to_dict()
        )
        validation_results["data_quality"]["infinite_values"] = (
            np.isinf(df.select_dtypes(include=[np.number])).sum().to_dict()
        )

        # Recommendations
        if len(validation_results["missing_fgcs_features"]) > 0:
            validation_results["recommendations"].append(
                f"Missing core FGCS features: {validation_results['missing_fgcs_features']}"
            )

        if validation_results["data_quality"]["total_samples"] < 50:
            validation_results["recommendations"].append(
                "Sample size < 50 may be insufficient for reliable modeling"
            )

        logger.info(
            f"Feature validation complete: {len(validation_results['fgcs_core_features'])}/3 core features found"
        )
        return validation_results


class EDPFeatureSelector:
    """
    Feature selection specifically optimized for EDP analysis.

    Implements multiple feature selection strategies to identify the most
    important features for energy-delay product optimization.
    """

    def __init__(self, selection_method: str = "comprehensive"):
        """
        Initialize EDP feature selector.

        Args:
            selection_method: Method to use ('fgcs', 'statistical', 'model_based', 'comprehensive')
        """
        self.selection_method = selection_method
        self.selected_features = []
        self.feature_importance_scores = {}
        self.selection_results = {}

        logger.info(f"EDP feature selector initialized with method: {selection_method}")

    def select_features_for_edp(
        self, df: pd.DataFrame, target_col: str = "power", max_features: int = 10
    ) -> List[str]:
        """
        Select optimal features for EDP prediction.

        Args:
            df: DataFrame with features and target
            target_col: Target column name
            max_features: Maximum number of features to select

        Returns:
            List of selected feature names
        """
        if self.selection_method == "fgcs":
            return self._select_fgcs_features(df)
        elif self.selection_method == "statistical":
            return self._select_statistical_features(df, target_col, max_features)
        elif self.selection_method == "model_based":
            return self._select_model_based_features(df, target_col, max_features)
        elif self.selection_method == "comprehensive":
            return self._select_comprehensive_features(df, target_col, max_features)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

    def _select_fgcs_features(self, df: pd.DataFrame) -> List[str]:
        """Select features based on FGCS methodology."""
        fgcs_priority_features = [
            "fp_activity",
            "dram_activity",
            "n_sm_app_clock",
            "fp_x_clock",
            "fp_activity_sq",
            "n_sm_app_clock_sq",
        ]

        selected = [f for f in fgcs_priority_features if f in df.columns]

        # Add any additional numeric features that might be useful
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in selected and col not in ["power", "energy", "execution_time"]:
                if len(selected) < 8:  # Limit to reasonable number
                    selected.append(col)

        self.selected_features = selected
        logger.info(f"FGCS feature selection: {len(selected)} features selected")
        return selected

    def _select_statistical_features(
        self, df: pd.DataFrame, target_col: str, max_features: int
    ) -> List[str]:
        """Select features using statistical methods."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]

        if len(feature_cols) == 0:
            logger.warning("No numeric features found for statistical selection")
            return []

        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)

        # F-statistic based selection
        selector = SelectKBest(
            score_func=f_regression, k=min(max_features, len(feature_cols))
        )
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        selected = [feature_cols[i] for i in selected_indices]

        # Store scores for analysis
        self.feature_importance_scores["f_scores"] = dict(
            zip(feature_cols, selector.scores_)
        )

        self.selected_features = selected
        logger.info(f"Statistical feature selection: {len(selected)} features selected")
        return selected

    def _select_model_based_features(
        self, df: pd.DataFrame, target_col: str, max_features: int
    ) -> List[str]:
        """Select features using model-based importance."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]

        if len(feature_cols) == 0:
            logger.warning("No numeric features found for model-based selection")
            return []

        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)

        # Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Get feature importance scores
        importance_scores = dict(zip(feature_cols, rf.feature_importances_))

        # Select top features
        sorted_features = sorted(
            importance_scores.items(), key=lambda x: x[1], reverse=True
        )
        selected = [f[0] for f in sorted_features[:max_features]]

        self.feature_importance_scores["rf_importance"] = importance_scores
        self.selected_features = selected

        logger.info(f"Model-based feature selection: {len(selected)} features selected")
        return selected

    def _select_comprehensive_features(
        self, df: pd.DataFrame, target_col: str, max_features: int
    ) -> List[str]:
        """Comprehensive feature selection combining multiple methods."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]

        if len(feature_cols) == 0:
            logger.warning("No numeric features found for comprehensive selection")
            return []

        # Start with FGCS priority features
        fgcs_features = self._select_fgcs_features(df)
        selected_features = set(fgcs_features)

        # Add top statistical features
        if len(selected_features) < max_features:
            remaining_slots = max_features - len(selected_features)
            stat_features = self._select_statistical_features(
                df, target_col, remaining_slots * 2
            )
            for feature in stat_features:
                if (
                    feature not in selected_features
                    and len(selected_features) < max_features
                ):
                    selected_features.add(feature)

        # Add top model-based features
        if len(selected_features) < max_features:
            remaining_slots = max_features - len(selected_features)
            model_features = self._select_model_based_features(
                df, target_col, remaining_slots * 2
            )
            for feature in model_features:
                if (
                    feature not in selected_features
                    and len(selected_features) < max_features
                ):
                    selected_features.add(feature)

        selected = list(selected_features)
        self.selected_features = selected

        logger.info(
            f"Comprehensive feature selection: {len(selected)} features selected"
        )
        return selected

    def get_feature_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of selected features.

        Returns:
            Dictionary with feature analysis results
        """
        analysis = {
            "selected_features": self.selected_features,
            "num_selected": len(self.selected_features),
            "selection_method": self.selection_method,
            "importance_scores": self.feature_importance_scores,
            "selection_results": self.selection_results,
        }

        return analysis


def create_optimized_feature_set(
    df: pd.DataFrame,
    gpu_type: str = "V100",
    target_col: str = "power",
    max_features: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create an optimized feature set for EDP analysis.

    Args:
        df: Input DataFrame with profiling data
        gpu_type: Target GPU type
        target_col: Target variable for optimization
        max_features: Maximum number of features to select

    Returns:
        Tuple of (processed DataFrame, analysis results)
    """
    logger.info("Creating optimized feature set for EDP analysis")

    # Step 1: FGCS feature engineering
    feature_engineer = FGCSFeatureEngineering(gpu_type=gpu_type)
    df_features = feature_engineer.extract_fgcs_features(df)
    df_interactions = feature_engineer.create_interaction_features(df_features)

    # Step 2: Feature validation
    validation_results = feature_engineer.validate_features(df_interactions)

    # Step 3: Feature selection
    selector = EDPFeatureSelector(selection_method="comprehensive")
    selected_features = selector.select_features_for_edp(
        df_interactions, target_col, max_features
    )

    # Step 4: Create final feature set
    final_columns = selected_features + [target_col]
    if "energy" in df_interactions.columns and "energy" not in final_columns:
        final_columns.append("energy")
    if (
        "execution_time" in df_interactions.columns
        and "execution_time" not in final_columns
    ):
        final_columns.append("execution_time")

    # Keep only columns that exist in the DataFrame
    final_columns = [col for col in final_columns if col in df_interactions.columns]
    df_final = df_interactions[final_columns].copy()

    # Step 5: Compile analysis results
    analysis_results = {
        "feature_engineering": {
            "total_features_created": len(df_interactions.columns),
            "fgcs_features": feature_engineer.feature_names,
        },
        "validation": validation_results,
        "selection": selector.get_feature_analysis(),
        "final_feature_set": {
            "features": selected_features,
            "total_columns": len(final_columns),
            "target_column": target_col,
        },
    }

    logger.info(
        f"Feature optimization complete: {len(selected_features)} features selected"
    )
    return df_final, analysis_results
