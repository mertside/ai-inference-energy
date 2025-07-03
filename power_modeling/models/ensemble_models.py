"""
Enhanced Random Forest and ensemble models extracted from FGCS 2023 paper.
Provides optimized hyperparameter tuning and model evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None

logger = logging.getLogger(__name__)


class EnhancedRandomForestModel:
    """
    Enhanced Random Forest model with hyperparameter optimization from FGCS 2023.
    """
    
    def __init__(self, optimization_method: str = 'random', n_iter: int = 100, 
                 cv_folds: int = 10, random_state: int = 42):
        """
        Initialize Enhanced Random Forest model.
        
        Args:
            optimization_method: 'random' or 'grid' search
            n_iter: Number of iterations for random search
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.optimization_method = optimization_method
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.is_trained = False
        
    def get_hyperparameter_grid(self) -> Dict[str, List]:
        """
        Get hyperparameter grid based on FGCS 2023 methodology.
        Uses conservative parameters to avoid invalid combinations.
        
        Returns:
            Dictionary of hyperparameter ranges
        """
        return {
            'n_estimators': [200, 500, 800, 1000],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [10, 30, 50, 80, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['squared_error', 'absolute_error'],
            # Only use bootstrap=True to avoid max_samples conflicts
            'bootstrap': [True],
            'max_samples': [0.8, 0.9, 1.0]
        }
        
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Optimize hyperparameters using the specified method.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Best hyperparameters found
        """
        logger.info(f"Optimizing Random Forest hyperparameters using {self.optimization_method} search")
        
        # Create base model with safe defaults
        base_rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        param_grid = self.get_hyperparameter_grid()
        
        # Validate that criterion values are supported
        try:
            test_rf = RandomForestRegressor(criterion='squared_error', n_estimators=10)
            test_rf.fit(X_train[:10], y_train[:10])  # Quick test
        except Exception as e:
            logger.warning(f"squared_error criterion not supported, falling back to mse: {e}")
            # Update param grid to use older criterion names
            param_grid['criterion'] = ['mse', 'mae']
        
        if self.optimization_method == 'random':
            search = RandomizedSearchCV(
                estimator=base_rf,
                param_distributions=param_grid,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                verbose=2,
                random_state=self.random_state,
                n_jobs=-1,
                scoring='neg_mean_squared_error'
            )
        elif self.optimization_method == 'grid':
            # Reduced grid for grid search to prevent excessive computation
            reduced_grid = {
                'n_estimators': [200, 500],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [10, 50, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True],  # Only bootstrap=True to avoid max_samples conflicts
                'max_samples': [0.8, 1.0]
            }
            search = GridSearchCV(
                estimator=base_rf,
                param_grid=reduced_grid,
                cv=self.cv_folds,
                verbose=2,
                n_jobs=-1,
                scoring='neg_mean_squared_error'
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
            
        search.fit(X_train, y_train)
        
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        
        logger.info(f"Best hyperparameters: {self.best_params}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return self.best_params
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            optimize: bool = True) -> 'EnhancedRandomForestModel':
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            optimize: Whether to optimize hyperparameters
            
        Returns:
            Self for method chaining
        """
        if optimize:
            self.optimize_hyperparameters(X_train, y_train)
        else:
            # Use default parameters
            self.model = RandomForestRegressor(
                n_estimators=1000,
                max_features='auto',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
        self.is_trained = True
        return self
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X_test)
        
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before accessing feature importance")
        return self.model.feature_importances_


class XGBoostPowerModel:
    """
    XGBoost model for power prediction with FGCS 2023 optimizations.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
            
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.is_trained = False
        
    def get_hyperparameter_grid(self) -> Dict[str, List]:
        """Get XGBoost hyperparameter grid."""
        return {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            optimize: bool = True) -> 'XGBoostPowerModel':
        """Train XGBoost model with optional hyperparameter optimization."""
        if optimize:
            logger.info("Optimizing XGBoost hyperparameters")
            
            base_xgb = XGBRegressor(random_state=self.random_state, n_jobs=-1)
            param_grid = self.get_hyperparameter_grid()
            
            search = RandomizedSearchCV(
                estimator=base_xgb,
                param_distributions=param_grid,
                n_iter=50,
                cv=5,
                verbose=1,
                random_state=self.random_state,
                n_jobs=-1,
                scoring='neg_mean_squared_error'
            )
            
            search.fit(X_train, y_train)
            self.best_params = search.best_params_
            self.model = search.best_estimator_
            
            logger.info(f"Best XGBoost parameters: {self.best_params}")
        else:
            self.model = XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
        self.is_trained = True
        return self
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions with trained XGBoost model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X_test)


class ModelEvaluator:
    """
    Model evaluation utilities based on FGCS 2023 methodology.
    """
    
    @staticmethod
    def evaluate_model(model: BaseEstimator, X_test: np.ndarray, 
                      y_test: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate model performance using FGCS metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        accuracy = 100 - mape
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'accuracy': accuracy
        }
        
        logger.info(f"\n{model_name} Performance:")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  Accuracy: {accuracy:.2f}%")
        
        return metrics
        
    @staticmethod
    def compare_models(models: Dict[str, BaseEstimator], X_test: np.ndarray, 
                      y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models and return performance comparison.
        
        Args:
            models: Dictionary of model_name -> trained_model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for name, model in models.items():
            metrics = ModelEvaluator.evaluate_model(model, X_test, y_test, name)
            metrics['model'] = name
            results.append(metrics)
            
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('r2', ascending=False)
        
        logger.info("\nModel Comparison (sorted by R²):")
        logger.info(comparison_df.to_string(index=False))
        
        return comparison_df


class EnsembleModel:
    """
    Ensemble model combining multiple power prediction models.
    """
    
    def __init__(self, models: List[BaseEstimator], weights: Optional[List[float]] = None):
        """
        Initialize ensemble model.
        
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
            
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Weights must sum to 1.0")
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Weighted average predictions
        """
        predictions = np.zeros(X.shape[0])
        
        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict(X)
            
        return predictions
        
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, standard_deviations)
        """
        all_predictions = np.array([model.predict(X) for model in self.models])
        
        # Weighted average
        weighted_pred = np.average(all_predictions, axis=0, weights=self.weights)
        
        # Weighted standard deviation
        weighted_std = np.sqrt(np.average((all_predictions - weighted_pred) ** 2, 
                                         axis=0, weights=self.weights))
        
        return weighted_pred, weighted_std
