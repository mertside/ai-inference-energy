"""
Data preprocessing module for AI inference energy profiling.
Handles data cleaning, normalization, and transformation for power modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess data for power modeling."""
    
    def __init__(self, normalization_method: str = 'log1p'):
        """
        Initialize the data preprocessor.
        
        Args:
            normalization_method: Method for normalization ('log1p', 'minmax', 'standard', 'none')
        """
        self.normalization_method = normalization_method
        self.scalers = {}
        self.feature_ranges = {}
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw profiling data.
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data")
        clean_data = data.copy()
        
        # Remove system columns
        columns_to_remove = ['Entity', '#', 'Unnamed: 0']
        for col in columns_to_remove:
            if col in clean_data.columns:
                del clean_data[col]
                
        # Remove header rows that appear in data
        if 'POWER' in clean_data.columns:
            clean_data = clean_data[clean_data['POWER'] != 'POWER']
            
        # Drop rows with all NaN values
        clean_data = clean_data.dropna(how='all')
        
        # Convert numeric columns
        numeric_columns = ['POWER', 'FP32A', 'FP64A', 'DRAMA', 'sm_app_clock', 
                          'power_usage', 'runtime', 'run_time', 'fp_active', 
                          'dram_active', 'fp32_active', 'fp64_active']
        
        for col in numeric_columns:
            if col in clean_data.columns:
                clean_data[col] = pd.to_numeric(clean_data[col], errors='coerce')
                
        logger.info(f"Cleaned data shape: {clean_data.shape}")
        return clean_data
    
    def normalize_features(self, 
                          data: pd.DataFrame, 
                          features: List[str] = None,
                          fit_scalers: bool = True) -> pd.DataFrame:
        """
        Normalize features using the specified method.
        
        Args:
            data: Data to normalize
            features: List of features to normalize
            fit_scalers: Whether to fit scalers or use existing ones
            
        Returns:
            DataFrame with normalized features
        """
        if features is None:
            features = ['power_usage', 'fp_active', 'dram_active', 'sm_app_clock', 'runtime']
            
        logger.info(f"Normalizing features using {self.normalization_method} method")
        
        normalized_data = data.copy()
        
        for feature in features:
            if feature in data.columns:
                if self.normalization_method == 'log1p':
                    normalized_data[f'n_{feature}'] = np.log1p(data[feature])
                elif self.normalization_method == 'minmax':
                    if fit_scalers:
                        scaler = MinMaxScaler()
                        normalized_data[f'n_{feature}'] = scaler.fit_transform(
                            data[feature].values.reshape(-1, 1)).flatten()
                        self.scalers[feature] = scaler
                    else:
                        if feature in self.scalers:
                            normalized_data[f'n_{feature}'] = self.scalers[feature].transform(
                                data[feature].values.reshape(-1, 1)).flatten()
                        else:
                            logger.warning(f"No scaler found for {feature}")
                            normalized_data[f'n_{feature}'] = data[feature]
                elif self.normalization_method == 'standard':
                    if fit_scalers:
                        scaler = StandardScaler()
                        normalized_data[f'n_{feature}'] = scaler.fit_transform(
                            data[feature].values.reshape(-1, 1)).flatten()
                        self.scalers[feature] = scaler
                    else:
                        if feature in self.scalers:
                            normalized_data[f'n_{feature}'] = self.scalers[feature].transform(
                                data[feature].values.reshape(-1, 1)).flatten()
                        else:
                            logger.warning(f"No scaler found for {feature}")
                            normalized_data[f'n_{feature}'] = data[feature]
                elif self.normalization_method == 'none':
                    normalized_data[f'n_{feature}'] = data[feature]
                else:
                    logger.warning(f"Unknown normalization method: {self.normalization_method}")
                    normalized_data[f'n_{feature}'] = data[feature]
                    
        return normalized_data
    
    def denormalize_predictions(self, 
                               predictions: np.ndarray, 
                               feature_name: str) -> np.ndarray:
        """
        Denormalize predictions back to original scale.
        
        Args:
            predictions: Normalized predictions
            feature_name: Name of the feature being denormalized
            
        Returns:
            Denormalized predictions
        """
        if self.normalization_method == 'log1p':
            return np.expm1(predictions)
        elif self.normalization_method in ['minmax', 'standard']:
            if feature_name in self.scalers:
                return self.scalers[feature_name].inverse_transform(
                    predictions.reshape(-1, 1)).flatten()
            else:
                logger.warning(f"No scaler found for {feature_name}")
                return predictions
        else:
            return predictions
    
    def create_polynomial_features(self, 
                                  data: pd.DataFrame, 
                                  features: List[str],
                                  degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for enhanced modeling.
        
        Args:
            data: Input data
            features: Features to create polynomials for
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
        """
        logger.info(f"Creating polynomial features with degree {degree}")
        
        poly_data = data.copy()
        feature_data = data[features].values
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(feature_data)
        
        # Create column names for polynomial features
        feature_names = poly.get_feature_names_out(features)
        
        # Add polynomial features to the DataFrame
        for i, name in enumerate(feature_names):
            if name not in features:  # Don't duplicate original features
                poly_data[f'poly_{name}'] = poly_features[:, i]
                
        return poly_data
    
    def split_data(self, 
                   data: pd.DataFrame, 
                   target_column: str,
                   feature_columns: List[str] = None,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            data: Input data
            target_column: Name of target column
            feature_columns: List of feature columns
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with test_size={test_size}")
        
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column and col != 'application']
            
        X = data[feature_columns]
        y = data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
        logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_modeling_data(self, 
                             data: pd.DataFrame, 
                             target: str,
                             features: List[str] = None,
                             normalize: bool = True,
                             polynomial_degree: int = None) -> Dict:
        """
        Prepare data for modeling with full preprocessing pipeline.
        
        Args:
            data: Raw data
            target: Target variable name
            features: List of features to use
            normalize: Whether to normalize features
            polynomial_degree: Degree for polynomial features
            
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info("Preparing data for modeling")
        
        # Clean data
        clean_data = self.clean_data(data)
        
        # Select features
        if features is None:
            features = ['fp_active', 'dram_active', 'sm_app_clock']
            
        # Filter to available features
        available_features = [f for f in features if f in clean_data.columns]
        if not available_features:
            raise ValueError(f"No requested features found in data. Available: {list(clean_data.columns)}")
            
        # Normalize if requested
        if normalize:
            processed_data = self.normalize_features(clean_data, available_features)
            modeling_features = [f'n_{f}' for f in available_features]
        else:
            processed_data = clean_data
            modeling_features = available_features
            
        # Create polynomial features if requested
        if polynomial_degree and polynomial_degree > 1:
            processed_data = self.create_polynomial_features(
                processed_data, modeling_features, polynomial_degree)
            # Update feature list to include polynomial features
            modeling_features = [col for col in processed_data.columns 
                               if col.startswith('poly_') or col in modeling_features]
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(
            processed_data, target, modeling_features)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': modeling_features,
            'target_name': target,
            'preprocessor': self
        }
    
    def save_preprocessor_state(self, filepath: str):
        """Save preprocessor state for later use."""
        import joblib
        joblib.dump({
            'normalization_method': self.normalization_method,
            'scalers': self.scalers,
            'feature_ranges': self.feature_ranges
        }, filepath)
        logger.info(f"Preprocessor state saved to {filepath}")
    
    def load_preprocessor_state(self, filepath: str):
        """Load preprocessor state."""
        import joblib
        state = joblib.load(filepath)
        self.normalization_method = state['normalization_method']
        self.scalers = state['scalers']
        self.feature_ranges = state['feature_ranges']
        logger.info(f"Preprocessor state loaded from {filepath}")
