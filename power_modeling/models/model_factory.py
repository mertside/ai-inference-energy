"""
Power Model Factory

Creates and manages different types of power prediction models based on the
methodologies from the FGCS 2023 paper. Supports multiple regression algorithms
including Random Forest, XGBoost, SVR, Linear Regression, and the exact FGCS models.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

from .ensemble_models import (
    EnhancedRandomForestModel,
    ModelEvaluator,
    XGBoostPowerModel,
)

# Import FGCS models
from .fgcs_models import (
    FGCSPowerModel,
    PerformanceMetricsCalculator,
    PolynomialPowerModel,
)

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None

logger = logging.getLogger(__name__)


class PowerModel(ABC):
    """Abstract base class for power prediction models."""

    def __init__(self, **kwargs):
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.hyperparameters = kwargs

    @abstractmethod
    def create_model(self, **kwargs):
        """Create the underlying model instance."""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None):
        """Train the model."""
        if self.model is None:
            self.create_model()

        self.model.fit(X, y)
        self.is_trained = True
        self.feature_names = feature_names
        logger.info(
            f"{self.__class__.__name__} trained on {X.shape[0]} samples with {X.shape[1]} features"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_)
        return None


class RandomForestPowerModel(PowerModel):
    """Random Forest power prediction model."""

    def create_model(self, **kwargs):
        """Create Random Forest model with optimized parameters from FGCS paper."""
        default_params = {
            "n_estimators": 1400,
            "max_depth": 100,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "auto",
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(self.hyperparameters)
        default_params.update(kwargs)

        self.model = RandomForestRegressor(**default_params)
        logger.info(f"Created Random Forest model with parameters: {default_params}")


class XGBoostPowerModel(PowerModel):
    """XGBoost power prediction model."""

    def create_model(self, **kwargs):
        """Create XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not available. Install with: pip install xgboost"
            )

        default_params = {
            "n_estimators": 800,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(self.hyperparameters)
        default_params.update(kwargs)

        self.model = XGBRegressor(**default_params)
        logger.info(f"Created XGBoost model with parameters: {default_params}")


class SVRPowerModel(PowerModel):
    """Support Vector Regression power prediction model."""

    def create_model(self, **kwargs):
        """Create SVR model."""
        default_params = {"kernel": "rbf", "C": 1.0, "gamma": "scale", "epsilon": 0.1}
        default_params.update(self.hyperparameters)
        default_params.update(kwargs)

        self.model = SVR(**default_params)
        logger.info(f"Created SVR model with parameters: {default_params}")


class LinearPowerModel(PowerModel):
    """Linear regression power prediction model."""

    def create_model(self, **kwargs):
        """Create linear model."""
        model_type = kwargs.get("model_type", "linear")

        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "ridge":
            alpha = kwargs.get("alpha", 1.0)
            self.model = Ridge(alpha=alpha)
        elif model_type == "lasso":
            alpha = kwargs.get("alpha", 1.0)
            self.model = Lasso(alpha=alpha)
        else:
            raise ValueError(f"Unknown linear model type: {model_type}")

        logger.info(f"Created {model_type} regression model")


class PolynomialPowerModel(PowerModel):
    """Polynomial regression power prediction model."""

    def create_model(self, **kwargs):
        """Create polynomial regression model."""
        degree = kwargs.get("degree", 2)
        include_bias = kwargs.get("include_bias", True)
        interaction_only = kwargs.get("interaction_only", False)

        polynomial_features = PolynomialFeatures(
            degree=degree, include_bias=include_bias, interaction_only=interaction_only
        )

        linear_regression = LinearRegression()

        self.model = Pipeline(
            [("poly", polynomial_features), ("linear", linear_regression)]
        )

        logger.info(f"Created polynomial regression model (degree={degree})")


class PowerModelFactory:
    """Factory for creating power prediction models."""

    _model_registry = {
        "random_forest": RandomForestPowerModel,
        "rf": RandomForestPowerModel,
        "xgboost": XGBoostPowerModel,
        "xgb": XGBoostPowerModel,
        "svr": SVRPowerModel,
        "svm": SVRPowerModel,
        "linear": LinearPowerModel,
        "ridge": LinearPowerModel,
        "lasso": LinearPowerModel,
        "polynomial": PolynomialPowerModel,
        "poly": PolynomialPowerModel,
    }

    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> PowerModel:
        """
        Create a power prediction model.

        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'svr', 'linear', 'polynomial')
            **kwargs: Model-specific parameters

        Returns:
            PowerModel instance
        """
        model_type = model_type.lower()

        if model_type not in cls._model_registry:
            available_models = list(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. Available: {available_models}"
            )

        model_class = cls._model_registry[model_type]

        # Handle special cases for linear model variants
        if model_type in ["ridge", "lasso"]:
            kwargs["model_type"] = model_type

        return model_class(**kwargs)

    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model types."""
        return list(cls._model_registry.keys())

    @classmethod
    def get_recommended_params(
        cls, model_type: str, dataset_size: str = "medium"
    ) -> Dict[str, Any]:
        """
        Get recommended hyperparameters based on FGCS paper results.

        Args:
            model_type: Type of model
            dataset_size: Size category ('small', 'medium', 'large')

        Returns:
            Dictionary of recommended parameters
        """
        model_type = model_type.lower()

        # Parameters optimized from FGCS paper experiments
        recommendations = {
            "random_forest": {
                "small": {"n_estimators": 800, "max_depth": 50},
                "medium": {"n_estimators": 1400, "max_depth": 100},
                "large": {"n_estimators": 2000, "max_depth": None},
            },
            "xgboost": {
                "small": {"n_estimators": 500, "max_depth": 4},
                "medium": {"n_estimators": 800, "max_depth": 6},
                "large": {"n_estimators": 1200, "max_depth": 8},
            },
            "svr": {
                "small": {"C": 1.0, "gamma": "scale"},
                "medium": {"C": 10.0, "gamma": "scale"},
                "large": {"C": 100.0, "gamma": "auto"},
            },
        }

        if model_type in recommendations:
            return recommendations[model_type].get(
                dataset_size, recommendations[model_type]["medium"]
            )

        return {}


class ModelEvaluator:
    """Evaluate and compare power prediction models."""

    @staticmethod
    def evaluate_model(
        model: PowerModel, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.

        Args:
            model: Trained PowerModel instance
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Accuracy (from FGCS paper definition)
        accuracy = 100 - mape

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "accuracy": accuracy,
        }

        logger.info(
            f"Model evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Accuracy: {accuracy:.2f}%"
        )

        return metrics

    @staticmethod
    def compare_models(
        models: Dict[str, PowerModel], X_test: np.ndarray, y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same test set.

        Args:
            models: Dictionary of model_name -> PowerModel
            X_test: Test features
            y_test: Test targets

        Returns:
            DataFrame with comparison results
        """
        results = []

        for name, model in models.items():
            if not model.is_trained:
                logger.warning(f"Model {name} is not trained, skipping evaluation")
                continue

            metrics = ModelEvaluator.evaluate_model(model, X_test, y_test)
            metrics["model"] = name
            results.append(metrics)

        comparison_df = pd.DataFrame(results)

        if not comparison_df.empty:
            # Sort by accuracy (descending)
            comparison_df = comparison_df.sort_values("accuracy", ascending=False)
            logger.info(f"Model comparison completed for {len(comparison_df)} models")

        return comparison_df


def hyperparameter_tuning(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_type: str = "random",
    cv_folds: int = 5,
    n_iter: int = 100,
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for power models.

    Args:
        model_type: Type of model to tune
        X_train: Training features
        y_train: Training targets
        search_type: 'random' or 'grid' search
        cv_folds: Number of cross-validation folds
        n_iter: Number of iterations for random search

    Returns:
        Dictionary with best parameters and scores
    """
    # Define parameter grids based on FGCS paper
    param_grids = {
        "random_forest": {
            "n_estimators": [200, 400, 800, 1200, 1400, 1600, 2000],
            "max_depth": [10, 50, 100, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt"],
        },
        "xgboost": {
            "n_estimators": [100, 300, 500, 800, 1000],
            "max_depth": [3, 4, 6, 8, 10],
            "learning_rate": [0.01, 0.1, 0.2, 0.3],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        },
        "svr": {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            "epsilon": [0.01, 0.1, 0.2, 0.5],
        },
    }

    if model_type not in param_grids:
        raise ValueError(f"Hyperparameter tuning not supported for {model_type}")

    # Create base model
    base_model = PowerModelFactory.create_model(model_type)
    base_model.create_model()

    param_grid = param_grids[model_type]

    # Perform search
    if search_type == "random":
        search = RandomizedSearchCV(
            base_model.model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv_folds,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
    elif search_type == "grid":
        search = GridSearchCV(
            base_model.model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown search type: {search_type}")

    search.fit(X_train, y_train)

    result = {
        "best_params": search.best_params_,
        "best_score": -search.best_score_,  # Convert back from negative MAE
        "cv_results": search.cv_results_,
    }

    logger.info(f"Hyperparameter tuning completed for {model_type}")
    logger.info(f"Best parameters: {result['best_params']}")
    logger.info(f"Best score (MAE): {result['best_score']:.4f}")

    return result


# ==============================================================================
# FGCS 2023 Model Factory Integration
# ==============================================================================


class FGCSModelFactory:
    """
    Factory for creating models using FGCS 2023 methodology.
    Provides access to both the exact FGCS models and enhanced versions.
    """

    @staticmethod
    def create_fgcs_power_model() -> FGCSPowerModel:
        """
        Create the exact power model from FGCS 2023 paper.
        Uses hardcoded coefficients from the published work.

        Returns:
            FGCSPowerModel instance
        """
        logger.info("Creating FGCS 2023 power model with paper coefficients")
        return FGCSPowerModel()

    @staticmethod
    def create_polynomial_model(
        degree: int = 2, include_bias: bool = True
    ) -> PolynomialPowerModel:
        """
        Create polynomial power model based on FGCS methodology.

        Args:
            degree: Polynomial degree
            include_bias: Whether to include bias term

        Returns:
            PolynomialPowerModel instance
        """
        logger.info(f"Creating polynomial power model (degree={degree})")
        return PolynomialPowerModel(degree=degree, include_bias=include_bias)

    @staticmethod
    def create_enhanced_random_forest(
        optimization_method: str = "random", n_iter: int = 100
    ) -> EnhancedRandomForestModel:
        """
        Create enhanced Random Forest model with FGCS optimizations.

        Args:
            optimization_method: 'random' or 'grid' search
            n_iter: Number of iterations for random search

        Returns:
            EnhancedRandomForestModel instance
        """
        logger.info(
            f"Creating enhanced Random Forest model with {optimization_method} search"
        )
        return EnhancedRandomForestModel(
            optimization_method=optimization_method, n_iter=n_iter
        )

    @staticmethod
    def create_xgboost_model() -> XGBoostPowerModel:
        """
        Create XGBoost model optimized for power prediction.

        Returns:
            XGBoostPowerModel instance
        """
        logger.info("Creating XGBoost power model")
        return XGBoostPowerModel()

    @staticmethod
    def create_model_suite() -> Dict[str, Any]:
        """
        Create a complete suite of models for comparison.

        Returns:
            Dictionary of model_name -> model_instance
        """
        logger.info("Creating complete FGCS model suite")

        models = {
            "fgcs_original": FGCSModelFactory.create_fgcs_power_model(),
            "polynomial_deg2": FGCSModelFactory.create_polynomial_model(degree=2),
            "polynomial_deg3": FGCSModelFactory.create_polynomial_model(degree=3),
            "random_forest_enhanced": FGCSModelFactory.create_enhanced_random_forest(),
            "random_forest_basic": PowerModelFactory.create_model("random_forest"),
            "linear_regression": PowerModelFactory.create_model("linear"),
            "ridge_regression": PowerModelFactory.create_model("ridge"),
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models["xgboost_enhanced"] = FGCSModelFactory.create_xgboost_model()
            models["xgboost_basic"] = PowerModelFactory.create_model("xgboost")

        logger.info(f"Created {len(models)} models in suite")
        return models

    @staticmethod
    def create_model_pipeline(
        model_types: Optional[List[str]] = None,
    ) -> "ModelPipeline":
        """
        Create a ModelPipeline instance for end-to-end modeling workflows.

        Args:
            model_types: List of model types to include in pipeline.
                        If None, uses default FGCS model selection.

        Returns:
            ModelPipeline instance ready for training and evaluation
        """
        logger.info("Creating FGCS model pipeline")

        # Use FGCS-optimized model selection if not specified
        if model_types is None:
            model_types = [
                "fgcs_original",
                "polynomial_deg2",
                "random_forest_enhanced",
                "linear",
            ]

        return ModelPipeline(model_types=model_types)


class ModelPipeline:
    """
    Complete modeling pipeline integrating FGCS methodology.
    """

    def __init__(self, model_types: Optional[List[str]] = None):
        """
        Initialize modeling pipeline.

        Args:
            model_types: List of model types to include in pipeline
        """
        self.model_types = model_types or [
            "fgcs_original",
            "polynomial_deg2",
            "random_forest_enhanced",
            "linear",
        ]
        self.models = {}
        self.evaluation_results = {}
        self.best_model = None

    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Train all models and evaluate performance.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with training and evaluation results
        """
        logger.info(f"Training {len(self.model_types)} models")

        # Create all models
        all_models = FGCSModelFactory.create_model_suite()

        trained_models = {}
        evaluation_results = {}

        for model_name in self.model_types:
            if model_name not in all_models:
                logger.warning(f"Model {model_name} not available, skipping")
                continue

            logger.info(f"Training {model_name}")
            model = all_models[model_name]

            try:
                # Handle different model interfaces
                if hasattr(model, "fit") and not isinstance(model, FGCSPowerModel):
                    if hasattr(model, "optimize_hyperparameters"):
                        # Enhanced models with optimization
                        model.fit(X_train, y_train, optimize=True)
                    else:
                        # Standard sklearn interface
                        model.fit(X_train, y_train)

                    # Evaluate model
                    from ..models.ensemble_models import (
                        ModelEvaluator as EnsembleEvaluator,
                    )

                    metrics = EnsembleEvaluator.evaluate_model(
                        model, X_test, y_test, model_name
                    )
                    evaluation_results[model_name] = metrics
                    trained_models[model_name] = model

                elif isinstance(model, FGCSPowerModel):
                    # FGCS model doesn't need training - it uses fixed coefficients
                    logger.info(
                        f"{model_name} uses fixed coefficients, no training needed"
                    )
                    trained_models[model_name] = model
                    # Evaluation would need to be done separately for FGCS model

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue

        self.models = trained_models
        self.evaluation_results = evaluation_results

        # Find best model
        best_model_name = None
        if evaluation_results:
            best_model_name = max(
                evaluation_results.keys(), key=lambda x: evaluation_results[x]["r2"]
            )
            self.best_model = trained_models[best_model_name]
            logger.info(
                f"Best model: {best_model_name} (R² = {evaluation_results[best_model_name]['r2']:.4f})"
            )

        return {
            "models": trained_models,
            "evaluations": evaluation_results,
            "best_model": self.best_model,
            "best_model_name": best_model_name,
        }

    def predict_power_across_frequencies(
        self,
        fp_activity: float,
        dram_activity: float,
        frequencies: List[int],
        model_name: str = None,
    ) -> pd.DataFrame:
        """
        Predict power consumption across frequency range using specified model.

        Args:
            fp_activity: FP operations activity
            dram_activity: DRAM activity
            frequencies: List of frequencies to evaluate
            model_name: Model to use (default: best model)

        Returns:
            DataFrame with predictions across frequencies
        """
        if model_name is None and self.best_model is not None:
            model = self.best_model  # best_model is now the actual model, not a tuple
        elif model_name is None:
            raise ValueError("No trained models available")
        elif model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in trained models")
        else:
            model = self.models[model_name]

        if isinstance(model, FGCSPowerModel):
            # Use FGCS model's built-in frequency prediction
            return model.predict_power(fp_activity, dram_activity, frequencies)
        else:
            # For other models, create feature matrix and predict
            features = pd.DataFrame(
                {
                    "fp_activity": [fp_activity] * len(frequencies),
                    "dram_activity": [dram_activity] * len(frequencies),
                    "sm_clock": frequencies,
                }
            )

            predictions = model.predict(features.values)

            result_df = pd.DataFrame({"frequency": frequencies, "power": predictions})

            return result_df


# Convenience functions for backward compatibility
create_model = PowerModelFactory.create_model
