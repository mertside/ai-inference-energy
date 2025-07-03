"""
Core power modeling algorithms extracted from FGCS 2023 paper implementation.
Provides polynomial-based power prediction models for GPU energy profiling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class PolynomialPowerModel:
    """
    Polynomial-based power prediction model based on FGCS 2023 methodology.
    Uses FP operations and DRAM activity as primary features.
    """
    
    def __init__(self, degree: int = 2, include_bias: bool = True):
        """
        Initialize polynomial power model.
        
        Args:
            degree: Polynomial degree for feature expansion
            include_bias: Whether to include bias term
        """
        self.degree = degree
        self.include_bias = include_bias
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=include_bias)
        self.model = LinearRegression()
        self.is_trained = False
        self.feature_names = None
        
    def fit(self, features: pd.DataFrame, power_values: pd.Series) -> 'PolynomialPowerModel':
        """
        Train the polynomial power model.
        
        Args:
            features: DataFrame with columns ['fp_activity', 'dram_activity', 'sm_clock']
            power_values: Power consumption values
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training polynomial power model (degree={self.degree})")
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Transform features using polynomial expansion
        X_poly = self.poly_features.fit_transform(features)
        
        # Train the model
        self.model.fit(X_poly, power_values)
        self.is_trained = True
        
        logger.info("Polynomial power model training completed")
        return self
        
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict power consumption for given features.
        
        Args:
            features: DataFrame with same columns as training data
            
        Returns:
            Predicted power values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X_poly = self.poly_features.transform(features)
        return self.model.predict(X_poly)
        
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients for analysis."""
        if not self.is_trained:
            raise ValueError("Model must be trained before accessing coefficients")
            
        feature_names = self.poly_features.get_feature_names_out(self.feature_names)
        return dict(zip(feature_names, self.model.coef_))


class FGCSPowerModel:
    """
    Power prediction model implementing the exact methodology from FGCS 2023 paper.
    Uses log-transformed features and hardcoded coefficients from the paper.
    """
    
    def __init__(self):
        """Initialize FGCS power model with paper coefficients."""
        # Coefficients from FGCS 2023 paper
        self.coefficients = {
            'intercept': -1.0318354343254663,
            'fp_coeff': 0.84864,
            'dram_coeff': 0.09749,
            'clock_coeff': 0.77006
        }
        self.is_trained = True  # Model uses fixed coefficients
        
    def predict_power(self, fp_activity: float, dram_activity: float, 
                     sm_clock_frequencies: List[int]) -> pd.DataFrame:
        """
        Predict power consumption across frequency range.
        
        Args:
            fp_activity: Average FP operations activity
            dram_activity: Average DRAM activity
            sm_clock_frequencies: List of SM clock frequencies to evaluate
            
        Returns:
            DataFrame with columns ['sm_app_clock', 'n_sm_app_clock', 'predicted_n_power_usage']
        """
        df = pd.DataFrame(sm_clock_frequencies, columns=['sm_app_clock'])
        df['n_sm_app_clock'] = np.log1p(df['sm_app_clock'])
        
        # Apply FGCS model equation
        df['predicted_n_power_usage'] = (
            self.coefficients['intercept'] +
            self.coefficients['fp_coeff'] * fp_activity +
            self.coefficients['dram_coeff'] * dram_activity +
            self.coefficients['clock_coeff'] * df['n_sm_app_clock']
        )
        
        return df
        
    def predict_runtime(self, app_df: pd.DataFrame, baseline_time: float, 
                       fp_activity: float) -> pd.DataFrame:
        """
        Predict runtime using FGCS polynomial model.
        
        Args:
            app_df: DataFrame with frequency information
            baseline_time: Baseline execution time
            fp_activity: FP operations activity
            
        Returns:
            DataFrame with predicted runtime and energy values
        """
        # Runtime model coefficients from FGCS paper
        runtime_coeffs = [1.43847511, -0.16736726, -0.90400864, 0.48241361, 0.78898516]
        B0 = 0  # Baseline coefficient
        
        T_fmax = np.log1p(baseline_time)
        
        # Calculate predicted runtime using polynomial model
        app_df['predicted_n_run_time'] = (
            T_fmax + B0 +
            runtime_coeffs[0] * fp_activity +
            runtime_coeffs[1] * (7.230563 - app_df['n_sm_app_clock']) +
            runtime_coeffs[2] * (fp_activity ** 2) +
            runtime_coeffs[3] * fp_activity * (7.230563 - app_df['n_sm_app_clock']) +
            runtime_coeffs[4] * ((7.230563 - app_df['n_sm_app_clock']) ** 2)
        )
        
        # Convert log predictions back to real values
        app_df['predicted_n_to_r_power_usage'] = np.expm1(app_df['predicted_n_power_usage'])
        app_df['predicted_n_to_r_run_time'] = np.expm1(app_df['predicted_n_run_time'])
        app_df['predicted_n_to_r_energy'] = (
            app_df['predicted_n_to_r_run_time'] * app_df['predicted_n_to_r_power_usage']
        )
        app_df['predicted_n_energy'] = np.log1p(app_df['predicted_n_to_r_energy'])
        
        return app_df


class PerformanceMetricsCalculator:
    """Calculate average performance metrics from profiling data."""
    
    @staticmethod
    def calculate_metrics(time_data_file: str, n_runs: int = 3) -> Tuple[float, float]:
        """
        Calculate average FP and DRAM activity metrics.
        
        Args:
            time_data_file: Path to CSV file with profiling data
            n_runs: Number of runs to average over
            
        Returns:
            Tuple of (fp_avg, dram_avg)
        """
        logger.info(f"Computing average metrics for {n_runs} runs from {time_data_file}")
        
        fp64_values = []
        fp32_values = []
        dram_values = []
        
        # Read and process data
        df = pd.read_csv(time_data_file, delim_whitespace=True, on_bad_lines='skip')
        
        # Clean data
        columns_to_remove = ['Entity', '#']
        for col in columns_to_remove:
            if col in df.columns:
                del df[col]
                
        # Remove header rows
        if 'POWER' in df.columns:
            df = df[df['POWER'] != 'POWER']
            
        df = df.dropna(axis=0)
        
        # Convert to numeric
        numeric_columns = ['FP32A', 'FP64A', 'DRAMA']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate metrics
        if 'FP64A' in df.columns:
            fp64_data = df[df['FP64A'] > 0]
            if not fp64_data.empty:
                fp64_values.append(fp64_data['FP64A'].mean())
                
        if 'FP32A' in df.columns:
            fp32_data = df[df['FP32A'] > 0]
            if not fp32_data.empty:
                fp32_values.append(fp32_data['FP32A'].mean())
                
        if 'DRAMA' in df.columns:
            dram_data = df[df['DRAMA'] > 0]
            if not dram_data.empty:
                dram_values.append(dram_data['DRAMA'].mean())
        
        # Calculate averages
        fp64_avg = np.mean(fp64_values) if fp64_values else 0
        fp32_avg = np.mean(fp32_values) if fp32_values else 0
        dram_avg = np.mean(dram_values) if dram_values else 0
        
        # Combine FP metrics according to FGCS methodology
        if not np.isnan(fp32_avg) and not np.isnan(fp64_avg):
            fp_avg = fp32_avg / 2 + fp64_avg
        elif np.isnan(fp64_avg):
            fp_avg = fp32_avg / 2
        elif np.isnan(fp32_avg):
            fp_avg = fp64_avg
        else:
            fp_avg = 0
            
        logger.info(f"Calculated metrics - FP: {fp_avg:.4f}, DRAM: {dram_avg:.4f}")
        return fp_avg, dram_avg
        
    @staticmethod
    def get_baseline_runtime(profiled_data_file: str, app_name: str) -> float:
        """
        Get baseline runtime from profiled data.
        
        Args:
            profiled_data_file: Path to performance data file
            app_name: Application name for scaling
            
        Returns:
            Baseline runtime in seconds
        """
        df = pd.read_csv(profiled_data_file)
        df.columns = ['sm_app_clock', 'perf', 'runtime']
        
        runtime_avg = df['runtime'].mean()
        
        # Apply application-specific scaling
        if app_name.lower() == 'dgemm':
            runtime_avg = runtime_avg / 1000
            
        return round(runtime_avg, 1)


# GPU frequency configurations for different architectures (corrected)
GPU_FREQUENCY_CONFIGS = {
    'V100': [1380, 1372, 1365, 1357, 1350, 1342, 1335, 1327, 1320, 1312, 1305, 1297, 1290, 1282, 1275, 1267, 
             1260, 1252, 1245, 1237, 1230, 1222, 1215, 1207, 1200, 1192, 1185, 1177, 1170, 1162, 1155, 1147, 
             1140, 1132, 1125, 1117, 1110, 1102, 1095, 1087, 1080, 1072, 1065, 1057, 1050, 1042, 1035, 1027, 
             1020, 1012, 1005, 997, 990, 982, 975, 967, 960, 952, 945, 937, 930, 922, 915, 907, 900, 892, 885, 877, 
             870, 862, 855, 847, 840, 832, 825, 817, 810, 802, 795, 787, 780, 772, 765, 757, 750, 742, 735, 727, 
             720, 712, 705, 697, 690, 682, 675, 667, 660, 652, 645, 637, 630, 622, 615, 607, 600, 592, 585, 577, 
             570, 562, 555, 547, 540, 532, 525, 517, 510, 502, 495, 487, 480, 472, 465, 457, 450, 442, 435, 427, 
             420, 412, 405],  # 103 frequencies
    'A100': [1410, 1395, 1380, 1365, 1350, 1335, 1320, 1305, 1290, 1275,
             1260, 1245, 1230, 1215, 1200, 1185, 1170, 1155, 1140, 1125,
             1110, 1095, 1080, 1065, 1050, 1035, 1020, 1005, 990, 975,
             960, 945, 930, 915, 900, 885, 870, 855, 840, 825,
             810, 795, 780, 765, 750, 735, 720, 705, 690, 675,
             660, 645, 630, 615, 600, 585, 570, 555, 540, 525, 510],  # 61 frequencies
    'H100': [1755, 1740, 1725, 1710, 1695, 1680, 1665, 1650, 1635, 1620,
             1605, 1590, 1575, 1560, 1545, 1530, 1515, 1500, 1485, 1470,
             1455, 1440, 1425, 1410, 1395, 1380, 1365, 1350, 1335, 1320,
             1305, 1290, 1275, 1260, 1245, 1230, 1215, 1200, 1185, 1170,
             1155, 1140, 1125, 1110, 1095, 1080, 1065, 1050, 1035, 1020,
             1005, 990, 975, 960, 945, 930, 915, 900, 885, 870,
             855, 840, 825, 810, 795, 780, 765, 750, 735, 720,
             705, 690, 675, 660, 645, 630, 615, 600, 585, 570,
             555, 540, 525, 510, 495, 480, 465, 450, 435, 420,
             405, 390, 375, 360, 345, 330, 315, 300, 285, 270,
             255, 240, 225, 210]  # 104 frequencies
}
