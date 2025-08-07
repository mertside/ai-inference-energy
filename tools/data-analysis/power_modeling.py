#!/usr/bin/env python3
"""
AI Inference Power and Performance Modeling for Optimal Frequency Selection.

This script implements power and performance models for AI inference workloads
based on proven methodologies from FGCS and ICPP papers. It extends the original
approach to handle AI-specific workload characteristics and provides models
suitable for real-time optimal frequency prediction.

Features:
- Linear power models (proven FGCS approach): P_f = α·FP_act + β·DRAM_act + γ·f + C ± λ
- Polynomial performance models (proven ICPP approach): T_f = T_base + polynomial terms
- AI workload-specific feature integration
- Cross-architecture portability (V100, A100, H100)
- Single-run prediction capability
- Model validation and accuracy metrics

Requirements:
    - Aggregated results CSV from aggregate_results.py
    - Python 3.8+ with scikit-learn, pandas, numpy

Author: Mert Side
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Add parent directory to path for imports
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    from utils import setup_logging
except ImportError:
    def setup_logging(level="INFO"):
        logging.basicConfig(level=getattr(logging, level))
        return logging.getLogger(__name__)


class AIInferencePowerModel:
    """
    Power modeling for AI inference workloads based on proven FGCS methodology.
    
    Implements linear power model: P_f = α·FP_act + β·DRAM_act + γ·f + C ± λ
    with architectural scaling factors for cross-GPU portability.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the power model.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or setup_logging()
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.architecture_factors = {
            # Architectural scaling factors (based on SM count normalization)
            'v100': 80 / 56,    # 80 SMs relative to base
            'a100': 108 / 56,   # 108 SMs relative to base  
            'h100': 132 / 56    # 132 SMs relative to base
        }
        self.training_stats = {}

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix for power modeling.
        
        Based on proven features from FGCS/ICPP papers plus AI-specific extensions.
        
        Args:
            df: DataFrame with experimental results
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        features_list = []
        feature_names = []
        
        # Core proven features (your FGCS/ICPP results)
        core_features = ['fp_active', 'dram_active', 'sm_app_clock']
        
        for feature in core_features:
            if feature in df.columns:
                features_list.append(df[feature].values.reshape(-1, 1))
                feature_names.append(feature)
            else:
                self.logger.warning(f"Core feature {feature} not found in data")
        
        # Architectural scaling factor (your proven portability approach)
        if 'gpu' in df.columns:
            arch_factors = df['gpu'].map(self.architecture_factors).fillna(1.0)
            features_list.append(arch_factors.values.reshape(-1, 1))
            feature_names.append('architecture_factor')
        
        # AI workload-specific features (extensions)
        ai_features = [
            'tensor_core_util', 'mixed_precision_ratio', 'sequence_processing_ratio',
            'attention_compute_ratio', 'denoising_compute_ratio', 'patch_processing_efficiency',
            'encoder_decoder_ratio', 'beam_search_memory'
        ]
        
        for feature in ai_features:
            if feature in df.columns:
                feature_data = df[feature].fillna(0.0)  # Fill missing AI features with 0
                features_list.append(feature_data.values.reshape(-1, 1))
                feature_names.append(feature)
        
        # Frequency interaction terms (proven to be important)
        if 'sm_app_clock' in df.columns and 'fp_active' in df.columns:
            freq_fp_interaction = (df['sm_app_clock'] * df['fp_active']).values.reshape(-1, 1)
            features_list.append(freq_fp_interaction)
            feature_names.append('freq_fp_interaction')
        
        if 'sm_app_clock' in df.columns and 'dram_active' in df.columns:
            freq_dram_interaction = (df['sm_app_clock'] * df['dram_active']).values.reshape(-1, 1)
            features_list.append(freq_dram_interaction)
            feature_names.append('freq_dram_interaction')
        
        if not features_list:
            raise ValueError("No valid features found for power modeling")
        
        # Combine all features
        feature_matrix = np.hstack(features_list)
        
        self.logger.info(f"Prepared {feature_matrix.shape[1]} features for power modeling")
        self.logger.debug(f"Features: {feature_names}")
        
        return feature_matrix, feature_names

    def train(self, df: pd.DataFrame, target_column: str = 'avg_power') -> Dict[str, Any]:
        """
        Train the power model using your proven linear regression approach.
        
        Args:
            df: Training data DataFrame
            target_column: Column name for power target variable
            
        Returns:
            Training statistics and model performance
        """
        self.logger.info(f"Training power model using {len(df)} samples")
        
        # Prepare features
        X, feature_names = self.prepare_features(df)
        self.feature_names = feature_names
        
        # Prepare target (average power during execution)
        if target_column not in df.columns:
            self.logger.error(f"Target column {target_column} not found")
            return {}
        
        y = df[target_column].values
        
        # Remove invalid samples
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.logger.info(f"Using {len(X)} valid samples for training")
        
        # Create model pipeline (your proven linear approach)
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        training_stats = {
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'feature_count': X.shape[1],
            'sample_count': len(X),
            'feature_names': feature_names
        }
        
        # Cross-validation (your proven validation approach)
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        training_stats['cv_r2_mean'] = cv_scores.mean()
        training_stats['cv_r2_std'] = cv_scores.std()
        
        # Feature importance (from linear regression coefficients)
        regressor = self.model.named_steps['regressor']
        feature_importance = np.abs(regressor.coef_)
        importance_dict = dict(zip(feature_names, feature_importance))
        training_stats['feature_importance'] = importance_dict
        
        self.training_stats = training_stats
        
        self.logger.info(f"Power model training completed:")
        self.logger.info(f"  Test R²: {training_stats['test_r2']:.3f}")
        self.logger.info(f"  Test MAE: {training_stats['test_mae']:.2f} W")
        self.logger.info(f"  CV R²: {training_stats['cv_r2_mean']:.3f} ± {training_stats['cv_r2_std']:.3f}")
        
        return training_stats

    def predict(self, features_dict: Dict[str, float]) -> float:
        """
        Predict power consumption for given features.
        
        Args:
            features_dict: Dictionary of feature values
            
        Returns:
            Predicted power consumption in Watts
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare feature vector
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in features_dict:
                feature_vector.append(features_dict[feature_name])
            elif feature_name == 'architecture_factor':
                # Default architecture factor
                feature_vector.append(1.0)
            elif 'interaction' in feature_name:
                # Calculate interaction terms
                if feature_name == 'freq_fp_interaction':
                    feature_vector.append(
                        features_dict.get('sm_app_clock', 0) * features_dict.get('fp_active', 0)
                    )
                elif feature_name == 'freq_dram_interaction':
                    feature_vector.append(
                        features_dict.get('sm_app_clock', 0) * features_dict.get('dram_active', 0)
                    )
                else:
                    feature_vector.append(0.0)
            else:
                # Default to 0 for missing features
                feature_vector.append(0.0)
        
        # Make prediction
        X = np.array(feature_vector).reshape(1, -1)
        prediction = self.model.predict(X)[0]
        
        return prediction

    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'architecture_factors': self.architecture_factors,
            'training_stats': self.training_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Saved power model to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.architecture_factors = model_data['architecture_factors']
        self.training_stats = model_data['training_stats']
        
        self.logger.info(f"Loaded power model from {filepath}")


class AIInferencePerformanceModel:
    """
    Performance modeling for AI inference workloads based on proven ICPP methodology.
    
    Implements polynomial performance model with AI workload-specific extensions.
    """

    def __init__(self, degree: int = 2, logger: Optional[logging.Logger] = None):
        """
        Initialize the performance model.
        
        Args:
            degree: Polynomial degree (your proven approach uses degree=2)
            logger: Optional logger instance
        """
        self.logger = logger or setup_logging()
        self.degree = degree
        self.model = None
        self.feature_names = []
        self.training_stats = {}

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix for performance modeling.
        
        Args:
            df: DataFrame with experimental results
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        features_list = []
        feature_names = []
        
        # Core features for performance modeling
        core_features = ['sm_app_clock', 'fp_active', 'dram_active']
        
        for feature in core_features:
            if feature in df.columns:
                features_list.append(df[feature].values.reshape(-1, 1))
                feature_names.append(feature)
        
        # AI workload-specific features that affect performance
        performance_features = [
            'tensor_core_util', 'sequence_processing_ratio', 'attention_compute_ratio',
            'patch_processing_efficiency', 'encoder_decoder_ratio'
        ]
        
        for feature in performance_features:
            if feature in df.columns:
                feature_data = df[feature].fillna(0.0)
                features_list.append(feature_data.values.reshape(-1, 1))
                feature_names.append(feature)
        
        # Workload-specific indicators (one-hot encoding)
        if 'workload' in df.columns:
            for workload in ['llama', 'stablediffusion', 'vit', 'whisper']:
                workload_indicator = (df['workload'] == workload).astype(float)
                features_list.append(workload_indicator.values.reshape(-1, 1))
                feature_names.append(f'workload_{workload}')
        
        if not features_list:
            raise ValueError("No valid features found for performance modeling")
        
        # Combine all features
        feature_matrix = np.hstack(features_list)
        
        self.logger.info(f"Prepared {feature_matrix.shape[1]} features for performance modeling")
        return feature_matrix, feature_names

    def train(self, df: pd.DataFrame, target_column: str = 'execution_time') -> Dict[str, Any]:
        """
        Train the performance model using polynomial regression approach.
        
        Args:
            df: Training data DataFrame
            target_column: Column name for performance target variable
            
        Returns:
            Training statistics and model performance
        """
        self.logger.info(f"Training performance model using {len(df)} samples")
        
        # Prepare features
        X, feature_names = self.prepare_features(df)
        self.feature_names = feature_names
        
        # Prepare target
        if target_column not in df.columns:
            self.logger.error(f"Target column {target_column} not found")
            return {}
        
        y = df[target_column].values
        
        # Remove invalid samples
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.logger.info(f"Using {len(X)} valid samples for training")
        
        # Create model pipeline (your proven polynomial approach)
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=self.degree, include_bias=False)),
            ('regressor', LinearRegression())
        ])
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        training_stats = {
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'feature_count': X.shape[1],
            'sample_count': len(X),
            'polynomial_degree': self.degree,
            'feature_names': feature_names
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        training_stats['cv_r2_mean'] = cv_scores.mean()
        training_stats['cv_r2_std'] = cv_scores.std()
        
        self.training_stats = training_stats
        
        self.logger.info(f"Performance model training completed:")
        self.logger.info(f"  Test R²: {training_stats['test_r2']:.3f}")
        self.logger.info(f"  Test MAE: {training_stats['test_mae']:.3f} s")
        self.logger.info(f"  CV R²: {training_stats['cv_r2_mean']:.3f} ± {training_stats['cv_r2_std']:.3f}")
        
        return training_stats

    def predict(self, features_dict: Dict[str, float]) -> float:
        """
        Predict execution time for given features.
        
        Args:
            features_dict: Dictionary of feature values
            
        Returns:
            Predicted execution time in seconds
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare feature vector
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in features_dict:
                feature_vector.append(features_dict[feature_name])
            elif feature_name.startswith('workload_'):
                # Default workload indicators to 0
                feature_vector.append(0.0)
            else:
                # Default to 0 for missing features
                feature_vector.append(0.0)
        
        # Make prediction
        X = np.array(feature_vector).reshape(1, -1)
        prediction = self.model.predict(X)[0]
        
        return prediction

    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'degree': self.degree,
            'training_stats': self.training_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Saved performance model to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.degree = model_data['degree']
        self.training_stats = model_data['training_stats']
        
        self.logger.info(f"Loaded performance model from {filepath}")


class AIOptimalFrequencyPredictor:
    """
    Optimal frequency predictor combining power and performance models.
    
    Implements your proven single-run prediction approach for real-time deployment.
    """

    def __init__(self, power_model: AIInferencePowerModel, 
                 performance_model: AIInferencePerformanceModel,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the optimal frequency predictor.
        
        Args:
            power_model: Trained power model
            performance_model: Trained performance model
            logger: Optional logger instance
        """
        self.power_model = power_model
        self.performance_model = performance_model
        self.logger = logger or setup_logging()

    def predict_optimal_frequency(self, features: Dict[str, float], 
                                 available_frequencies: List[int],
                                 performance_constraint_pct: float = 5.0,
                                 baseline_performance: Optional[float] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Predict optimal frequency using your proven approach.
        
        Args:
            features: Feature dictionary from single run at max frequency
            available_frequencies: List of available GPU frequencies
            performance_constraint_pct: Performance degradation constraint
            baseline_performance: Baseline performance for constraint calculation
            
        Returns:
            Tuple of (optimal_frequency, optimization_details)
        """
        predictions = []
        
        for freq in available_frequencies:
            # Update features with current frequency
            freq_features = features.copy()
            freq_features['sm_app_clock'] = freq
            
            # Predict power and performance
            predicted_power = self.power_model.predict(freq_features)
            predicted_time = self.performance_model.predict(freq_features)
            
            # Calculate energy and EDP
            predicted_energy = predicted_power * predicted_time
            edp = predicted_energy * predicted_time
            
            predictions.append({
                'frequency': freq,
                'power': predicted_power,
                'execution_time': predicted_time,
                'energy': predicted_energy,
                'edp': edp
            })
        
        # Apply performance constraint if baseline is provided
        if baseline_performance is not None:
            max_acceptable_time = baseline_performance * (1 + performance_constraint_pct / 100)
            valid_predictions = [p for p in predictions if p['execution_time'] <= max_acceptable_time]
        else:
            valid_predictions = predictions
        
        if not valid_predictions:
            # No frequency meets constraint, return maximum frequency
            max_freq = max(available_frequencies)
            return max_freq, {'error': 'No frequency meets performance constraint'}
        
        # Find optimal frequency (minimum EDP)
        optimal = min(valid_predictions, key=lambda x: x['edp'])
        
        # Calculate optimization details
        baseline_prediction = next(p for p in predictions if p['frequency'] == max(available_frequencies))
        
        details = {
            'optimal_frequency': optimal['frequency'],
            'predicted_power': optimal['power'],
            'predicted_time': optimal['execution_time'],
            'predicted_energy': optimal['energy'],
            'predicted_edp': optimal['edp'],
            'baseline_energy': baseline_prediction['energy'],
            'energy_savings_pct': (1 - optimal['energy'] / baseline_prediction['energy']) * 100,
            'performance_impact_pct': (optimal['execution_time'] / baseline_prediction['execution_time'] - 1) * 100,
            'valid_frequencies': len(valid_predictions),
            'total_frequencies': len(predictions)
        }
        
        return optimal['frequency'], details


def train_ai_models(df: pd.DataFrame, output_dir: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Train both power and performance models for AI inference workloads.
    
    Args:
        df: Training data DataFrame
        output_dir: Output directory for saved models
        logger: Logger instance
        
    Returns:
        Training results summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Train power model
    logger.info("Training AI inference power model")
    power_model = AIInferencePowerModel(logger=logger)
    power_stats = power_model.train(df, target_column='avg_power')
    power_model.save_model(output_path / 'ai_power_model.pkl')
    results['power_model'] = power_stats
    
    # Train performance model
    logger.info("Training AI inference performance model")
    performance_model = AIInferencePerformanceModel(degree=2, logger=logger)
    performance_stats = performance_model.train(df, target_column='execution_time')
    performance_model.save_model(output_path / 'ai_performance_model.pkl')
    results['performance_model'] = performance_stats
    
    # Save training summary
    with open(output_path / 'training_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved trained models to {output_path}")
    return results


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Train AI inference power and performance models for optimal frequency selection"
    )
    parser.add_argument(
        "-d", "--data",
        required=True,
        help="Path to aggregated results CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        default="ai_models",
        help="Output directory for trained models (default: ai_models)"
    )
    parser.add_argument(
        "--power-target",
        default="avg_power",
        help="Target column for power modeling (default: avg_power)"
    )
    parser.add_argument(
        "--performance-target",
        default="execution_time",
        help="Target column for performance modeling (default: execution_time)"
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=2,
        help="Polynomial degree for performance model (default: 2)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    try:
        # Load data
        logger.info(f"Loading training data from {args.data}")
        df = pd.read_csv(args.data)
        logger.info(f"Loaded {len(df)} training samples")
        
        # Train models
        results = train_ai_models(df, args.output, logger)
        
        # Display results
        print("=" * 60)
        print("AI INFERENCE MODEL TRAINING RESULTS")
        print("=" * 60)
        
        if 'power_model' in results:
            power_stats = results['power_model']
            print(f"\nPower Model Performance:")
            print(f"  Test R²: {power_stats.get('test_r2', 0):.3f}")
            print(f"  Test MAE: {power_stats.get('test_mae', 0):.2f} W")
            print(f"  CV R²: {power_stats.get('cv_r2_mean', 0):.3f} ± {power_stats.get('cv_r2_std', 0):.3f}")
        
        if 'performance_model' in results:
            perf_stats = results['performance_model']
            print(f"\nPerformance Model Performance:")
            print(f"  Test R²: {perf_stats.get('test_r2', 0):.3f}")
            print(f"  Test MAE: {perf_stats.get('test_mae', 0):.3f} s")
            print(f"  CV R²: {perf_stats.get('cv_r2_mean', 0):.3f} ± {perf_stats.get('cv_r2_std', 0):.3f}")
        
        logger.info("Model training completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
