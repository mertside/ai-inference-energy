"""
FGCS Power Modeling Integration Script

This module provides a high-level interface to the power modeling capabilities
extracted from the FGCS 2023 paper. It integrates:

1. Power prediction models (polynomial, random forest, XGBoost)
2. EDP optimization algorithms
3. Performance metrics calculation
4. Frequency sweep analysis
5. Visualization and reporting

Usage:
    from power_modeling.fgcs_integration import FGCSPowerModelingFramework
    
    # Initialize framework
    framework = FGCSPowerModelingFramework()
    
    # Train models on your data
    results = framework.train_and_evaluate(training_data, test_data)
    
    # Optimize application
    optimization = framework.optimize_application(app_data)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

# Import our modules
from .models.model_factory import FGCSModelFactory, ModelPipeline
from .models.fgcs_models import FGCSPowerModel, PerformanceMetricsCalculator
from .models.ensemble_models import ModelEvaluator
from ..edp_analysis.edp_calculator import FGCSEDPOptimizer, DVFSOptimizationPipeline
from .feature_engineering.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class FGCSPowerModelingFramework:
    """
    Complete power modeling framework based on FGCS 2023 methodology.
    
    This class provides a high-level interface to all the power modeling
    capabilities extracted from the FGCS paper and enhanced for the new framework.
    """
    
    def __init__(self, model_types: Optional[List[str]] = None,
                 gpu_type: str = 'V100'):
        """
        Initialize the FGCS power modeling framework.
        
        Args:
            model_types: List of model types to use
            gpu_type: GPU type for frequency configurations
        """
        self.model_types = model_types or [
            'fgcs_original', 'polynomial_deg2', 'random_forest_enhanced'
        ]
        self.gpu_type = gpu_type
        self.preprocessor = DataPreprocessor()
        self.model_pipeline = ModelPipeline(model_types=self.model_types)
        self.trained_models = {}
        self.best_model = None
        
        # GPU-specific frequency configurations (corrected)
        self.frequency_configs = {
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
        
        logger.info(f"FGCS Framework initialized for {gpu_type} with models: {self.model_types}")
    
    def load_profiling_data(self, data_file: str, app_name: str = "application") -> Dict[str, Any]:
        """
        Load and preprocess profiling data from CSV file.
        
        Args:
            data_file: Path to profiling data CSV file
            app_name: Application name for processing
            
        Returns:
            Dictionary with processed data and metrics
        """
        logger.info(f"Loading profiling data from {data_file}")
        
        try:
            # Calculate performance metrics using FGCS methodology
            fp_activity, dram_activity = PerformanceMetricsCalculator.calculate_metrics(
                data_file, n_runs=3
            )
            
            # Load the raw data for further processing
            raw_data = pd.read_csv(data_file, delim_whitespace=True, on_bad_lines='skip')
            cleaned_data = self.preprocessor.clean_data(raw_data)
            
            result = {
                'fp_activity': fp_activity,
                'dram_activity': dram_activity,
                'raw_data': raw_data,
                'cleaned_data': cleaned_data,
                'app_name': app_name,
                'data_file': data_file
            }
            
            logger.info(f"Data loaded successfully: FP={fp_activity:.4f}, DRAM={dram_activity:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load profiling data: {str(e)}")
            raise
    
    def train_models(self, training_data: pd.DataFrame, target_column: str = 'power',
                    test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train all models in the pipeline.
        
        Args:
            training_data: DataFrame with features and target
            target_column: Name of the target column
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training power prediction models")
        
        # Prepare features and targets
        feature_columns = [col for col in training_data.columns if col != target_column]
        X = training_data[feature_columns].values
        y = training_data[target_column].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train models
        results = self.model_pipeline.train_models(X_train, y_train, X_test, y_test)
        
        self.trained_models = results['models']
        self.best_model = results['best_model']
        
        logger.info(f"Model training complete. Best model: {self.best_model[0] if self.best_model else 'None'}")
        return results
    
    def predict_power_sweep(self, fp_activity: float, dram_activity: float,
                          frequencies: Optional[List[int]] = None,
                          model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Predict power consumption across frequency range.
        
        Args:
            fp_activity: FP operations activity
            dram_activity: DRAM activity
            frequencies: List of frequencies (default: GPU-specific range)
            model_name: Model to use (default: best model)
            
        Returns:
            DataFrame with power predictions across frequencies
        """
        if frequencies is None:
            frequencies = self.frequency_configs.get(self.gpu_type, self.frequency_configs['V100'])
        
        logger.info(f"Predicting power across {len(frequencies)} frequencies")
        
        # Use the model pipeline to make predictions
        result = self.model_pipeline.predict_power_across_frequencies(
            fp_activity, dram_activity, frequencies, model_name
        )
        
        return result
    
    def optimize_application(self, fp_activity: float, dram_activity: float,
                           baseline_runtime: float, app_name: str = "Application",
                           frequencies: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Perform complete application optimization using FGCS methodology.
        
        Args:
            fp_activity: FP operations activity
            dram_activity: DRAM activity
            baseline_runtime: Baseline execution time
            app_name: Application name
            frequencies: Frequencies to evaluate
            
        Returns:
            Complete optimization results
        """
        logger.info(f"Optimizing {app_name} using FGCS methodology")
        
        if not self.trained_models:
            # Use FGCS original model if no models trained
            logger.info("No trained models found, using FGCS original model")
            power_model = FGCSModelFactory.create_fgcs_power_model()
        else:
            power_model = self.best_model[1] if self.best_model else list(self.trained_models.values())[0]
        
        # Create optimization pipeline
        optimizer = DVFSOptimizationPipeline(power_model)
        
        if frequencies is None:
            frequencies = self.frequency_configs.get(self.gpu_type, self.frequency_configs['V100'])
        
        # Run optimization
        results = optimizer.optimize_application(
            fp_activity, dram_activity, baseline_runtime, frequencies, app_name
        )
        
        return results
    
    def analyze_from_file(self, profiling_file: str, performance_file: Optional[str] = None,
                         app_name: str = "Application") -> Dict[str, Any]:
        """
        Complete analysis pipeline from profiling files.
        
        Args:
            profiling_file: Path to profiling data file
            performance_file: Optional path to performance data file
            app_name: Application name
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Running complete analysis for {app_name}")
        
        # Load data
        profiling_data = self.load_profiling_data(profiling_file, app_name)
        
        # Get baseline runtime
        if performance_file:
            baseline_runtime = PerformanceMetricsCalculator.get_baseline_runtime(
                performance_file, app_name
            )
        else:
            # Use a default baseline
            baseline_runtime = 1.0
            logger.warning("No performance file provided, using default baseline runtime")
        
        # Run optimization
        optimization_results = self.optimize_application(
            profiling_data['fp_activity'],
            profiling_data['dram_activity'],
            baseline_runtime,
            app_name
        )
        
        # Combine results
        complete_results = {
            'profiling_data': profiling_data,
            'baseline_runtime': baseline_runtime,
            'optimization_results': optimization_results,
            'summary': self._generate_summary(optimization_results),
            'metadata': {
                'gpu_type': self.gpu_type,
                'models_used': self.model_types,
                'best_model': self.best_model[0] if self.best_model else 'fgcs_original'
            }
        }
        
        return complete_results
    
    def _generate_summary(self, optimization_results: Dict) -> Dict[str, str]:
        """Generate a human-readable summary of optimization results."""
        opt_results = optimization_results['optimization_results']
        recommendations = optimization_results['recommendations']
        
        primary_rec = recommendations['primary_recommendation']
        
        summary = {
            'optimal_frequency': f"{primary_rec['frequency']} MHz",
            'energy_savings': primary_rec['expected_energy_savings'],
            'performance_impact': primary_rec['expected_performance_impact'],
            'recommendation': primary_rec['reason'],
            'edp_frequency': f"{opt_results['edp_optimal']['frequency']} MHz",
            'ed2p_frequency': f"{opt_results['ed2p_optimal']['frequency']} MHz"
        }
        
        return summary
    
    def save_results(self, results: Dict, output_dir: str = "results") -> Dict[str, str]:
        """
        Save analysis results to files.
        
        Args:
            results: Results dictionary from analysis
            output_dir: Output directory
            
        Returns:
            Dictionary of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        app_name = results.get('profiling_data', {}).get('app_name', 'application')
        
        # Save frequency sweep data
        if 'frequency_sweep_data' in results['optimization_results']:
            sweep_file = output_path / f"{app_name}_frequency_sweep.csv"
            results['optimization_results']['frequency_sweep_data'].to_csv(sweep_file, index=False)
        
        # Save optimization results
        opt_file = output_path / f"{app_name}_optimization_results.json"
        import json
        with open(opt_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = self._convert_for_json(results['optimization_results'])
            json.dump(json_results, f, indent=2)
        
        # Save summary
        summary_file = output_path / f"{app_name}_summary.txt"
        with open(summary_file, 'w') as f:
            summary = results['summary']
            f.write(f"FGCS Power Modeling Analysis Summary - {app_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Optimal Frequency: {summary['optimal_frequency']}\n")
            f.write(f"Expected Energy Savings: {summary['energy_savings']}\n")
            f.write(f"Performance Impact: {summary['performance_impact']}\n")
            f.write(f"Recommendation: {summary['recommendation']}\n\n")
            f.write(f"EDP Optimal Frequency: {summary['edp_frequency']}\n")
            f.write(f"ED²P Optimal Frequency: {summary['ed2p_frequency']}\n")
        
        saved_files = {
            'frequency_sweep': str(sweep_file) if 'frequency_sweep_data' in results['optimization_results'] else None,
            'optimization_results': str(opt_file),
            'summary': str(summary_file)
        }
        
        logger.info(f"Results saved to {output_dir}")
        return saved_files
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj


# Convenience function for quick analysis
def analyze_application(profiling_file: str, performance_file: Optional[str] = None,
                       app_name: str = "Application", gpu_type: str = 'V100',
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick analysis function for single application.
    
    Args:
        profiling_file: Path to profiling data
        performance_file: Optional performance data file
        app_name: Application name
        gpu_type: GPU type (V100, A100, H100)
        output_dir: Optional output directory for results
        
    Returns:
        Complete analysis results
    """
    framework = FGCSPowerModelingFramework(gpu_type=gpu_type)
    results = framework.analyze_from_file(profiling_file, performance_file, app_name)
    
    if output_dir:
        framework.save_results(results, output_dir)
    
    return results
