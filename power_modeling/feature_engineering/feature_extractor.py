"""
Feature Extractor for GPU Power Modeling

Extracts relevant features from GPU profiling data for power prediction models.
Based on the feature selection methodology from the FGCS 2023 paper.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

logger = logging.getLogger(__name__)


class GPUFeatureExtractor:
    """Extract and select features from GPU profiling data."""

    def __init__(self, correlation_threshold: float = 0.8):
        """
        Initialize feature extractor.

        Args:
            correlation_threshold: Threshold for removing highly correlated features
        """
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.correlation_matrix = None
        self.feature_importance = None

    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic GPU features from profiling data.

        Expected columns from DCGMI data:
        - POWER: GPU power draw (W)
        - SMACT: SM active percentage
        - SMOCC: SM occupancy percentage
        - FP32A: FP32 active percentage
        - FP64A: FP64 active percentage
        - DRAMA: DRAM active percentage
        - TEMP: GPU temperature (C)
        - MEMCLK: Memory clock (MHz)
        - GRCLK: Graphics clock (MHz)

        Args:
            df: DataFrame with raw GPU profiling data

        Returns:
            DataFrame with extracted features
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        # Clean column names and handle variations
        df = df.copy()

        # Map common column name variations
        column_mapping = {
            "power.draw [W]": "POWER",
            "utilization.gpu [%]": "SMACT",
            "utilization.memory [%]": "DRAMA",
            "clocks.current.sm [MHz]": "GRCLK",
            "clocks.current.memory [MHz]": "MEMCLK",
            "temperature.gpu": "TEMP",
            "memory.used [MiB]": "MEM_USED",
            "memory.free [MiB]": "MEM_FREE",
        }

        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Remove metadata columns that are not features
        metadata_columns = ["Entity", "#", "timestamp", "app", "application"]
        for col in metadata_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Remove rows that are headers (common in DCGMI output)
        for col in df.columns:
            if col in df.values:
                df = df[df[col] != col]

        # Convert to numeric, coercing errors to NaN
        numeric_columns = [
            "POWER",
            "SMACT",
            "SMOCC",
            "FP32A",
            "FP64A",
            "DRAMA",
            "TEMP",
            "MEMCLK",
            "GRCLK",
            "MEM_USED",
            "MEM_FREE",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows with all NaN values
        df = df.dropna(how="all")

        logger.info(
            f"Extracted basic features from {len(df)} samples with {len(df.columns)} features"
        )

        return df

    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features based on FGCS paper methodology.

        Args:
            df: DataFrame with basic features

        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()

        # Combined floating point activity (key feature from paper)
        if "FP32A" in df.columns and "FP64A" in df.columns:
            # Handle NaN values in FP calculations
            fp32_clean = df["FP32A"].fillna(0)
            fp64_clean = df["FP64A"].fillna(0)

            # Calculate combined FP activity as in the paper
            df["fp_active"] = fp32_clean / 2 + fp64_clean
            logger.info("Calculated combined FP activity feature")

        # Memory utilization ratio
        if "MEM_USED" in df.columns and "MEM_FREE" in df.columns:
            total_memory = df["MEM_USED"] + df["MEM_FREE"]
            df["memory_utilization"] = df["MEM_USED"] / total_memory
            df["memory_utilization"] = df["memory_utilization"].fillna(0)

        # Clock ratio features
        if "GRCLK" in df.columns:
            # Normalized graphics clock (log transform as in paper)
            df["sm_app_clock"] = df["GRCLK"]
            df["n_sm_app_clock"] = np.log1p(df["GRCLK"])

        if "MEMCLK" in df.columns:
            df["n_mem_clock"] = np.log1p(df["MEMCLK"])

        # Compute utilization efficiency metrics
        if "SMACT" in df.columns and "DRAMA" in df.columns:
            df["compute_memory_ratio"] = df["SMACT"] / (
                df["DRAMA"] + 1e-6
            )  # Avoid division by zero
            df["compute_memory_ratio"] = df["compute_memory_ratio"].replace(
                [np.inf, -np.inf], 0
            )

        # Power efficiency metrics
        if "POWER" in df.columns and "SMACT" in df.columns:
            df["power_per_utilization"] = df["POWER"] / (df["SMACT"] + 1e-6)
            df["power_per_utilization"] = df["power_per_utilization"].replace(
                [np.inf, -np.inf], 0
            )

        logger.info(f"Calculated derived features, total features: {len(df.columns)}")

        return df

    def remove_highly_correlated_features(
        self, df: pd.DataFrame, target_col: str = None
    ) -> pd.DataFrame:
        """
        Remove highly correlated features to reduce multicollinearity.

        Args:
            df: DataFrame with features
            target_col: Target column to exclude from correlation analysis

        Returns:
            DataFrame with reduced feature set
        """
        if df.empty:
            return df

        # Exclude target column from feature correlation analysis
        feature_cols = [col for col in df.columns if col != target_col]

        if len(feature_cols) < 2:
            return df

        # Calculate correlation matrix for features only
        feature_df = df[feature_cols].select_dtypes(include=[np.number])

        if feature_df.empty:
            logger.warning("No numeric features found for correlation analysis")
            return df

        corr_matrix = feature_df.corr().abs()
        self.correlation_matrix = corr_matrix

        # Find pairs of highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > self.correlation_threshold)
        ]

        # Keep features that are more correlated with target if available
        if target_col and target_col in df.columns:
            target_correlations = (
                df[feature_cols + [target_col]].corr()[target_col].abs()
            )

            # For each pair of highly correlated features, keep the one more correlated with target
            refined_drop = []
            for col in to_drop:
                # Find what it's correlated with
                correlated_features = upper_triangle.index[
                    upper_triangle[col] > self.correlation_threshold
                ].tolist()

                if correlated_features:
                    # Compare target correlations and keep the better one
                    col_target_corr = target_correlations.get(col, 0)
                    keep_current = True

                    for corr_feature in correlated_features:
                        other_target_corr = target_correlations.get(corr_feature, 0)
                        if other_target_corr > col_target_corr:
                            keep_current = False
                            break

                    if not keep_current:
                        refined_drop.append(col)
                else:
                    refined_drop.append(col)

            to_drop = refined_drop

        # Drop the highly correlated features
        df_reduced = df.drop(columns=to_drop)

        logger.info(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
        logger.info(f"Remaining features: {len(df_reduced.columns)}")

        return df_reduced

    def select_top_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        n_features: int = 10,
        method: str = "f_regression",
    ) -> pd.DataFrame:
        """
        Select top N features using statistical methods.

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            n_features: Number of features to select
            method: Selection method ('f_regression', 'mutual_info')

        Returns:
            DataFrame with selected features and target
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]

        # Remove rows with NaN in target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Fill NaN in features with median
        X = X.fillna(X.median())

        if X.empty or len(X.columns) == 0:
            logger.warning("No valid numeric features found")
            return df[[target_col]]

        # Limit n_features to available features
        n_features = min(n_features, len(X.columns))

        # Select features based on method
        if method == "f_regression":
            selector = SelectKBest(score_func=f_regression, k=n_features)
        elif method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        X_selected = selector.fit_transform(X, y)
        selected_feature_names = X.columns[selector.get_support()].tolist()

        # Store feature importance scores
        self.feature_importance = dict(zip(X.columns, selector.scores_))
        self.selected_features = selected_feature_names

        # Create result DataFrame
        result_df = pd.DataFrame(
            X_selected, columns=selected_feature_names, index=X.index
        )
        result_df[target_col] = y

        logger.info(
            f"Selected top {len(selected_feature_names)} features using {method}"
        )
        logger.info(f"Selected features: {selected_feature_names}")

        return result_df

    def calculate_feature_correlations(
        self, df: pd.DataFrame, target_col: str
    ) -> Dict[str, float]:
        """
        Calculate correlations between features and target variable.

        Args:
            df: DataFrame with features and target
            target_col: Name of target column

        Returns:
            Dictionary of feature -> correlation with target
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        correlations = {}
        target_values = df[target_col].dropna()

        for col in df.columns:
            if col != target_col and df[col].dtype in [np.float64, np.int64]:
                # Get overlapping non-null values
                mask = ~(df[col].isna() | df[target_col].isna())
                if mask.sum() > 1:  # Need at least 2 points for correlation
                    corr, p_value = pearsonr(
                        df.loc[mask, col], df.loc[mask, target_col]
                    )
                    correlations[col] = corr

        # Sort by absolute correlation
        correlations = dict(
            sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        logger.info(f"Calculated correlations for {len(correlations)} features")

        return correlations

    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for all features.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with feature summary statistics
        """
        numeric_df = df.select_dtypes(include=[np.number])

        summary = {
            "feature": numeric_df.columns,
            "count": numeric_df.count(),
            "mean": numeric_df.mean(),
            "std": numeric_df.std(),
            "min": numeric_df.min(),
            "max": numeric_df.max(),
            "missing_pct": (numeric_df.isna().sum() / len(numeric_df)) * 100,
        }

        summary_df = pd.DataFrame(summary)

        if self.feature_importance:
            importance_scores = [
                self.feature_importance.get(feat, 0) for feat in summary_df["feature"]
            ]
            summary_df["importance_score"] = importance_scores

        return (
            summary_df.sort_values("importance_score", ascending=False)
            if "importance_score" in summary_df.columns
            else summary_df
        )


def process_dcgmi_data(
    df: pd.DataFrame,
    target_col: str = "POWER",
    n_features: int = 10,
    remove_outliers: bool = True,
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for DCGMI data.

    Args:
        df: Raw DCGMI DataFrame
        target_col: Target variable column name
        n_features: Number of features to select
        remove_outliers: Whether to remove outlier data points

    Returns:
        Dictionary with processed data and metadata
    """
    extractor = GPUFeatureExtractor()

    # Extract basic features
    df_basic = extractor.extract_basic_features(df)

    # Calculate derived features
    df_derived = extractor.calculate_derived_features(df_basic)

    # Remove highly correlated features
    df_reduced = extractor.remove_highly_correlated_features(df_derived, target_col)

    # Remove outliers if requested
    if remove_outliers and target_col in df_reduced.columns:
        # Use IQR method to remove outliers
        Q1 = df_reduced[target_col].quantile(0.25)
        Q3 = df_reduced[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (df_reduced[target_col] >= lower_bound) & (
            df_reduced[target_col] <= upper_bound
        )
        df_reduced = df_reduced[outlier_mask]

        logger.info(f"Removed {(~outlier_mask).sum()} outliers")

    # Select top features
    df_final = extractor.select_top_features(df_reduced, target_col, n_features)

    # Calculate feature correlations
    correlations = extractor.calculate_feature_correlations(df_final, target_col)

    # Get feature summary
    feature_summary = extractor.get_feature_summary(df_final)

    result = {
        "processed_data": df_final,
        "feature_correlations": correlations,
        "feature_summary": feature_summary,
        "selected_features": extractor.selected_features,
        "correlation_matrix": extractor.correlation_matrix,
        "extractor": extractor,
    }

    logger.info(
        f"DCGMI data processing complete: {len(df_final)} samples, {len(df_final.columns)-1} features"
    )

    return result
