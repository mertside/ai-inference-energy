#!/usr/bin/env python3
"""
Comprehensive Power Modeling Framework Test Suite

This test suite consolidates and refines all power modeling tests into a single,
well-organized test file covering:
1. Model accuracy and training
2. EDP/ED2P calculations and optimization
3. End-to-end pipeline functionality
4. GPU frequency configurations
5. Framework integration

Consolidated from multiple test files and optimized for clarity and maintainability.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import unittest
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from power_modeling import FGCSPowerModelingFramework, analyze_application
from power_modeling.models.fgcs_models import FGCSPowerModel
from power_modeling.models.model_factory import PolynomialPowerModel
from power_modeling.models.ensemble_models import EnhancedRandomForestModel
from power_modeling.models.model_factory import FGCSModelFactory, ModelPipeline
from edp_analysis.edp_calculator import EDPCalculator, FGCSEDPOptimizer, DVFSOptimizationPipeline

# Configure logging
logging.basicConfig(level=logging.ERROR)  # Reduce noise during testing
logger = logging.getLogger(__name__)


class TestPowerModelingFramework(unittest.TestCase):
    """Test cases for the power modeling framework."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data used across multiple tests."""
        np.random.seed(42)
        cls.n_samples = 100
        
        # Generate synthetic training data
        cls.training_data = pd.DataFrame({
            'fp_activity': np.random.uniform(0.1, 0.8, cls.n_samples),
            'dram_activity': np.random.uniform(0.05, 0.4, cls.n_samples),
            'sm_clock': np.random.choice(range(800, 1401, 50), cls.n_samples),
            'power': np.random.uniform(150, 300, cls.n_samples)
        })
        
        # Test frequencies
        cls.test_frequencies = [1000, 1100, 1200, 1300, 1400]
        
        # Test application parameters
        cls.test_fp_activity = 0.3
        cls.test_dram_activity = 0.15
        cls.test_baseline_runtime = 1.0
    
    def test_framework_initialization(self):
        """Test framework initialization with different GPU types."""
        # Test default initialization
        framework_default = FGCSPowerModelingFramework()
        self.assertIsNotNone(framework_default)
        self.assertEqual(framework_default.gpu_type, 'V100')
        
        # Test with different GPU types
        for gpu_type in ['V100', 'A100', 'H100']:
            framework = FGCSPowerModelingFramework(gpu_type=gpu_type)
            self.assertEqual(framework.gpu_type, gpu_type)
            self.assertIn(gpu_type, framework.frequency_configs)
    
    def test_model_imports_and_creation(self):
        """Test that all required models can be imported and created."""
        # Test model factory
        fgcs_model = FGCSModelFactory.create_fgcs_power_model()
        self.assertIsInstance(fgcs_model, FGCSPowerModel)
        
        poly_model = FGCSModelFactory.create_polynomial_model(degree=2)
        self.assertIsInstance(poly_model, PolynomialPowerModel)
        
        rf_model = FGCSModelFactory.create_enhanced_random_forest()
        self.assertIsInstance(rf_model, EnhancedRandomForestModel)
        
        pipeline = FGCSModelFactory.create_model_pipeline()
        self.assertIsInstance(pipeline, ModelPipeline)
    
    def test_random_forest_optimization(self):
        """Test Random Forest hyperparameter optimization (fixes previous issues)."""
        # Create model
        model = EnhancedRandomForestModel()
        
        # Verify hyperparameter grid doesn't contain max_samples
        grid = model.get_hyperparameter_grid()
        self.assertNotIn('max_samples', grid, "max_samples should be removed from hyperparameter grid")
        
        # Test training without optimization
        X = np.random.rand(50, 3)
        y = np.random.rand(50) * 200 + 50
        
        model.fit(X, y, optimize=False)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(X))
        
        # Test training with optimization (small grid for speed)
        model_opt = EnhancedRandomForestModel(optimization_method='random', n_iter=3)
        model_opt.fit(X, y, optimize=True)
        predictions_opt = model_opt.predict(X)
        self.assertEqual(len(predictions_opt), len(X))
    
    def test_model_training_pipeline(self):
        """Test the complete model training pipeline."""
        framework = FGCSPowerModelingFramework()
        
        # Test model training
        training_results = framework.train_models(self.training_data, target_column='power')
        
        self.assertIn('models', training_results)
        self.assertIn('best_model', training_results)
        self.assertIsNotNone(training_results['best_model'])
        self.assertGreater(len(training_results['models']), 0)
    
    def test_power_prediction(self):
        """Test power prediction functionality."""
        framework = FGCSPowerModelingFramework()
        
        # Train models first
        framework.train_models(self.training_data, target_column='power')
        
        # Test power sweep prediction
        power_sweep = framework.predict_power_sweep(
            fp_activity=self.test_fp_activity,
            dram_activity=self.test_dram_activity,
            frequencies=self.test_frequencies
        )
        
        self.assertEqual(len(power_sweep), len(self.test_frequencies))
        self.assertIn('frequency', power_sweep.columns)
        self.assertIn('power', power_sweep.columns)
        
        # Verify all frequencies are present
        for freq in self.test_frequencies:
            self.assertIn(freq, power_sweep['frequency'].values)
    
    def test_edp_calculations(self):
        """Test EDP and ED²P calculations."""
        calculator = EDPCalculator()
        
        # Test basic calculations
        energy = np.array([100, 150, 200])
        delay = np.array([1.0, 1.2, 1.4])
        
        # Test EDP calculation
        edp = calculator.calculate_edp(energy, delay)
        expected_edp = energy * delay
        np.testing.assert_allclose(edp, expected_edp)
        
        # Test ED²P calculation
        ed2p = calculator.calculate_ed2p(energy, delay)
        expected_ed2p = energy * (delay ** 2)
        np.testing.assert_allclose(ed2p, expected_ed2p)
        
        # Test with DataFrame
        df = pd.DataFrame({
            'energy': energy,
            'execution_time': delay,
            'frequency': [1000, 1100, 1200]
        })
        
        optimal_config = calculator.find_optimal_configuration(df, metric='edp')
        self.assertIn('optimal_frequency', optimal_config)
        self.assertIn('optimal_energy', optimal_config)
        self.assertIn('optimal_delay', optimal_config)
        self.assertIn('optimal_score', optimal_config)
    
    def test_fgcs_edp_optimizer(self):
        """Test FGCS EDP optimizer functionality."""
        # Create synthetic FGCS-format data
        frequencies = self.test_frequencies
        df = pd.DataFrame({
            'sm_app_clock': frequencies,
            'predicted_n_to_r_energy': [100, 120, 140, 160, 180],
            'predicted_n_to_r_run_time': [1.5, 1.3, 1.1, 0.9, 0.8],
            'predicted_n_to_r_power_usage': [66.7, 92.3, 127.3, 177.8, 225.0]
        })
        
        # Test EDP optimization
        edp_freq, edp_time, edp_power, edp_energy = FGCSEDPOptimizer.edp_optimal(df)
        
        self.assertIn(edp_freq, frequencies)
        self.assertGreater(edp_time, 0)
        self.assertGreater(edp_power, 0)
        self.assertGreater(edp_energy, 0)
        
        # Test ED²P optimization
        ed2p_freq, ed2p_time, ed2p_power, ed2p_energy = FGCSEDPOptimizer.ed2p_optimal(df)
        
        self.assertIn(ed2p_freq, frequencies)
        self.assertGreater(ed2p_time, 0)
        
        # Test full optimization analysis
        results = FGCSEDPOptimizer.analyze_dvfs_optimization(df, "TestApp")
        
        self.assertIn('edp_optimal', results)
        self.assertIn('ed2p_optimal', results)
        self.assertIn('min_energy', results)
        self.assertIn('min_time', results)
    
    def test_gpu_frequency_configurations(self):
        """Test GPU frequency configurations are correct."""
        expected_counts = {'V100': 131, 'A100': 61, 'H100': 104}
        expected_ranges = {
            'V100': (405, 1380),
            'A100': (510, 1410),
            'H100': (210, 1755)
        }
        
        for gpu_type in ['V100', 'A100', 'H100']:
            framework = FGCSPowerModelingFramework(gpu_type=gpu_type)
            frequencies = framework.frequency_configs[gpu_type]
            
            # Check frequency count
            actual_count = len(frequencies)
            expected_count = expected_counts[gpu_type]
            self.assertEqual(actual_count, expected_count, 
                           f"{gpu_type} frequency count mismatch: {actual_count} != {expected_count}")
            
            # Check frequency range
            min_freq, max_freq = min(frequencies), max(frequencies)
            expected_min, expected_max = expected_ranges[gpu_type]
            self.assertEqual(min_freq, expected_min)
            self.assertEqual(max_freq, expected_max)
            
            # Check frequencies are sorted in descending order
            self.assertEqual(frequencies, sorted(frequencies, reverse=True))
    
    def test_optimization_pipeline(self):
        """Test the complete optimization pipeline."""
        framework = FGCSPowerModelingFramework()
        
        # Train models first
        framework.train_models(self.training_data, target_column='power')
        
        # Test optimization
        optimization_results = framework.optimize_application(
            fp_activity=self.test_fp_activity,
            dram_activity=self.test_dram_activity,
            baseline_runtime=self.test_baseline_runtime,
            app_name="TestApp"
        )
        
        self.assertIn('optimization_results', optimization_results)
        self.assertIn('recommendations', optimization_results)
        self.assertIn('frequency_sweep_data', optimization_results)
        self.assertIn('input_parameters', optimization_results)
        
        # Verify optimization results structure
        opt_results = optimization_results['optimization_results']
        self.assertIn('edp_optimal', opt_results)
        self.assertIn('ed2p_optimal', opt_results)
        
        # Verify recommendations structure
        recommendations = optimization_results['recommendations']
        self.assertIn('primary_recommendation', recommendations)
    
    def test_end_to_end_pipeline(self):
        """Test the complete end-to-end pipeline."""
        framework = FGCSPowerModelingFramework()
        
        # Test model training
        training_results = framework.train_models(self.training_data, target_column='power')
        self.assertIn('models', training_results)
        self.assertIn('best_model', training_results)
        self.assertIsNotNone(training_results['best_model'])
        
        # Test power prediction
        power_sweep = framework.predict_power_sweep(
            fp_activity=self.test_fp_activity,
            dram_activity=self.test_dram_activity,
            frequencies=self.test_frequencies
        )
        self.assertEqual(len(power_sweep), len(self.test_frequencies))
        self.assertIn('frequency', power_sweep.columns)
        self.assertIn('power', power_sweep.columns)
        
        # Test optimization
        optimization_results = framework.optimize_application(
            fp_activity=self.test_fp_activity,
            dram_activity=self.test_dram_activity,
            baseline_runtime=self.test_baseline_runtime,
            app_name="TestApp"
        )
        self.assertIn('optimization_results', optimization_results)
        self.assertIn('recommendations', optimization_results)
    
    def test_quick_analysis_function(self):
        """Test the quick analysis function with file input."""
        # Create temporary test data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate synthetic profiling data
            profiling_data = pd.DataFrame({
                'app_name': ['TestApp'] * 50,
                'fp_activity': np.random.uniform(0.1, 0.8, 50),
                'dram_activity': np.random.uniform(0.05, 0.4, 50),
                'sm_clock': np.random.choice(range(800, 1401, 50), 50),
                'power': np.random.uniform(150, 300, 50)
            })
            
            profiling_file = os.path.join(temp_dir, "profiling.csv")
            profiling_data.to_csv(profiling_file, index=False)
            
            # Test quick analysis
            results = analyze_application(
                profiling_file=profiling_file,
                app_name="TestApp",
                gpu_type="V100"
            )
            
            # Verify results structure
            self.assertIn('summary', results)
            self.assertIn('optimization_results', results)
            self.assertIn('profiling_data', results)
            
            # Verify summary content
            summary = results['summary']
            self.assertIn('optimal_frequency', summary)
            self.assertIn('energy_savings', summary)
            self.assertIn('performance_impact', summary)


class TestValidationAndMetrics(unittest.TestCase):
    """Test cases for model validation and metrics."""
    
    def test_model_validation_pipeline(self):
        """Test the model validation pipeline."""
        # Generate test data
        np.random.seed(42)
        n_samples = 200
        
        X = np.random.rand(n_samples, 3)
        y = np.random.rand(n_samples) * 200 + 50
        
        # Split data
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Test individual models
        models = {
            'Enhanced_RF': EnhancedRandomForestModel(),
            'Polynomial_Deg2': PolynomialPowerModel(degree=2),
        }
        
        for name, model in models.items():
            with self.subTest(model=name):
                # Test training
                if hasattr(model, 'fit'):
                    if name == 'Enhanced_RF':
                        model.fit(X_train, y_train, optimize=False)  # Skip optimization for speed
                    else:
                        feature_names = ['fp_activity', 'dram_activity', 'sm_clock']
                        X_train_df = pd.DataFrame(X_train, columns=feature_names)
                        model.fit(X_train_df, pd.Series(y_train))
                    
                    # Test prediction
                    if name == 'Enhanced_RF':
                        predictions = model.predict(X_test)
                    else:
                        X_test_df = pd.DataFrame(X_test, columns=feature_names)
                        predictions = model.predict(X_test_df)
                    
                    self.assertEqual(len(predictions), len(X_test))
                    self.assertTrue(np.all(np.isfinite(predictions)))


if __name__ == '__main__':
    # Set up test environment
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Run tests
    unittest.main(verbosity=2)
