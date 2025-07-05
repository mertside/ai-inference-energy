"""
Power Modeling Validation Utilities

This module provides comprehensive validation metrics and testing utilities
for the power modeling framework, including model validation, statistical
analysis, and performance benchmarking.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score

logger = logging.getLogger(__name__)


class ModelValidationMetrics:
    """Comprehensive model validation metrics calculator."""

    @staticmethod
    def calculate_basic_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate basic regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "max_error": np.max(np.abs(y_true - y_pred)),
            "std_error": np.std(y_true - y_pred),
        }
        return metrics

    @staticmethod
    def calculate_relative_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate relative accuracy metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of relative metrics
        """
        mean_true = np.mean(y_true)
        relative_metrics = {
            "relative_mae": mean_absolute_error(y_true, y_pred) / mean_true * 100,
            "relative_rmse": np.sqrt(mean_squared_error(y_true, y_pred))
            / mean_true
            * 100,
            "accuracy_within_5pct": np.mean(np.abs(y_true - y_pred) / y_true <= 0.05)
            * 100,
            "accuracy_within_10pct": np.mean(np.abs(y_true - y_pred) / y_true <= 0.10)
            * 100,
            "accuracy_within_15pct": np.mean(np.abs(y_true - y_pred) / y_true <= 0.15)
            * 100,
        }
        return relative_metrics

    @staticmethod
    def calculate_energy_specific_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate energy-specific validation metrics.

        Args:
            y_true: True power values
            y_pred: Predicted power values

        Returns:
            Dictionary of energy-specific metrics
        """
        # Energy estimation errors (assuming 1-second intervals)
        energy_true = y_true
        energy_pred = y_pred

        energy_metrics = {
            "energy_mae": mean_absolute_error(energy_true, energy_pred),
            "energy_mape": mean_absolute_percentage_error(energy_true, energy_pred)
            * 100,
            "total_energy_error": np.abs(np.sum(energy_true) - np.sum(energy_pred)),
            "relative_total_energy_error": np.abs(
                np.sum(energy_true) - np.sum(energy_pred)
            )
            / np.sum(energy_true)
            * 100,
        }
        return energy_metrics


class CrossValidationAnalyzer:
    """Cross-validation analysis for model validation."""

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize cross-validation analyzer.

        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def validate_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform cross-validation analysis.

        Args:
            model: Trained model with predict method
            X: Feature matrix
            y: Target values

        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {self.cv_folds}-fold cross-validation")

        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=self.kfold, scoring="r2", n_jobs=-1)

        # Calculate additional metrics for each fold
        mae_scores = []
        mse_scores = []
        mape_scores = []

        for train_idx, test_idx in self.kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model on fold
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            mse_scores.append(mean_squared_error(y_test, y_pred))
            mape_scores.append(mean_absolute_percentage_error(y_test, y_pred) * 100)

        results = {
            "r2_scores": cv_scores,
            "mae_scores": np.array(mae_scores),
            "mse_scores": np.array(mse_scores),
            "mape_scores": np.array(mape_scores),
            "mean_r2": np.mean(cv_scores),
            "std_r2": np.std(cv_scores),
            "mean_mae": np.mean(mae_scores),
            "std_mae": np.std(mae_scores),
            "mean_mse": np.mean(mse_scores),
            "std_mse": np.std(mse_scores),
            "mean_mape": np.mean(mape_scores),
            "std_mape": np.std(mape_scores),
            "confidence_interval_r2": (
                np.mean(cv_scores) - 1.96 * np.std(cv_scores) / np.sqrt(len(cv_scores)),
                np.mean(cv_scores) + 1.96 * np.std(cv_scores) / np.sqrt(len(cv_scores)),
            ),
        }

        return results


class PowerModelValidator:
    """Comprehensive power model validation suite."""

    def __init__(self):
        """Initialize the power model validator."""
        self.metrics_calculator = ModelValidationMetrics()
        self.cv_analyzer = CrossValidationAnalyzer()

    def validate_power_model(
        self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of a power model.

        Args:
            model: Trained power model
            X_test: Test features
            y_test: Test power values
            model_name: Name of the model

        Returns:
            Complete validation results
        """
        logger.info(f"Validating {model_name}")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate all metrics
        basic_metrics = self.metrics_calculator.calculate_basic_metrics(y_test, y_pred)
        relative_metrics = self.metrics_calculator.calculate_relative_metrics(
            y_test, y_pred
        )
        energy_metrics = self.metrics_calculator.calculate_energy_specific_metrics(
            y_test, y_pred
        )

        # Combine all metrics
        validation_results = {
            "model_name": model_name,
            "basic_metrics": basic_metrics,
            "relative_metrics": relative_metrics,
            "energy_metrics": energy_metrics,
            "predictions": {
                "y_true": y_test,
                "y_pred": y_pred,
                "residuals": y_test - y_pred,
            },
        }

        return validation_results

    def compare_models(
        self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare multiple models using validation metrics.

        Args:
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test power values

        Returns:
            Model comparison results
        """
        logger.info(f"Comparing {len(models)} models")

        comparison_results = {}

        for model_name, model in models.items():
            comparison_results[model_name] = self.validate_power_model(
                model, X_test, y_test, model_name
            )

        # Create comparison summary
        summary = self._create_comparison_summary(comparison_results)

        return {
            "individual_results": comparison_results,
            "summary": summary,
            "best_model": summary["best_model"],
        }

    def _create_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of model comparison results."""
        summary_data = []

        for model_name, result in results.items():
            summary_data.append(
                {
                    "model": model_name,
                    "r2": result["basic_metrics"]["r2"],
                    "mae": result["basic_metrics"]["mae"],
                    "rmse": result["basic_metrics"]["rmse"],
                    "mape": result["basic_metrics"]["mape"],
                    "relative_mae": result["relative_metrics"]["relative_mae"],
                    "accuracy_within_10pct": result["relative_metrics"][
                        "accuracy_within_10pct"
                    ],
                }
            )

        summary_df = pd.DataFrame(summary_data)

        # Find best model (highest R²)
        best_idx = summary_df["r2"].idxmax()
        best_model = summary_df.loc[best_idx, "model"]

        return {
            "summary_table": summary_df,
            "best_model": best_model,
            "best_metrics": summary_df.loc[best_idx].to_dict(),
        }

    def validate_frequency_predictions(
        self, model, fp_activity: float, dram_activity: float, frequencies: List[int]
    ) -> Dict[str, Any]:
        """
        Validate model predictions across frequency range.

        Args:
            model: Trained power model
            fp_activity: FP activity level
            dram_activity: DRAM activity level
            frequencies: List of frequencies to test

        Returns:
            Frequency prediction validation results
        """
        logger.info("Validating frequency predictions")

        predictions = []

        for freq in frequencies:
            # Create feature vector
            features = np.array([[fp_activity, dram_activity, freq]])
            power_pred = model.predict(features)[0]
            predictions.append({"frequency": freq, "power": power_pred})

        pred_df = pd.DataFrame(predictions)

        # Validate prediction characteristics
        validation_results = {
            "predictions": pred_df,
            "characteristics": {
                "monotonic_increasing": pred_df["power"].is_monotonic_increasing,
                "power_range": (pred_df["power"].min(), pred_df["power"].max()),
                "power_gradient": np.gradient(pred_df["power"].values),
                "reasonable_range": (pred_df["power"].min() > 0)
                and (pred_df["power"].max() < 1000),
            },
        }

        return validation_results


def generate_validation_report(
    validation_results: Dict[str, Any], output_file: Optional[str] = None
) -> str:
    """
    Generate a comprehensive validation report.

    Args:
        validation_results: Results from model validation
        output_file: Optional output file path

    Returns:
        Formatted validation report as string
    """
    report_lines = []

    # Header
    report_lines.append("Power Model Validation Report")
    report_lines.append("=" * 50)
    report_lines.append("")

    # Model comparison summary
    if "summary" in validation_results:
        report_lines.append("Model Comparison Summary:")
        report_lines.append("-" * 30)

        summary_df = validation_results["summary"]["summary_table"]
        report_lines.append(summary_df.to_string(index=False, float_format="%.4f"))
        report_lines.append("")

        best_model = validation_results["summary"]["best_model"]
        report_lines.append(f"Best Model: {best_model}")
        report_lines.append("")

    # Detailed metrics for each model
    if "individual_results" in validation_results:
        report_lines.append("Detailed Metrics:")
        report_lines.append("-" * 20)

        for model_name, result in validation_results["individual_results"].items():
            report_lines.append(f"\n{model_name}:")
            report_lines.append(f"  R² Score: {result['basic_metrics']['r2']:.4f}")
            report_lines.append(f"  MAE: {result['basic_metrics']['mae']:.2f} W")
            report_lines.append(f"  RMSE: {result['basic_metrics']['rmse']:.2f} W")
            report_lines.append(f"  MAPE: {result['basic_metrics']['mape']:.2f}%")
            report_lines.append(
                f"  Accuracy within 10%: {result['relative_metrics']['accuracy_within_10pct']:.1f}%"
            )

    # Generate report string
    report = "\n".join(report_lines)

    # Save to file if specified
    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        logger.info(f"Validation report saved to {output_file}")

    return report


# Export key classes
__all__ = [
    "ModelValidationMetrics",
    "CrossValidationAnalyzer",
    "PowerModelValidator",
    "generate_validation_report",
]
