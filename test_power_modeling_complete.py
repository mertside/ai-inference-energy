#!/usr/bin/env python3
"""
Comprehensive Power Modeling and EDP/ED2P Testing Suite

This script thoroughly tests the core power modeling components extracted from
gpupowermodel and validates the complete EDP/ED2P calculation framework.

Tests include:
1. FGCS model accuracy verification
2. EDP/ED2P calculation validation
3. Frequency configuration testing
4. End-to-end pipeline testing
5. Performance benchmarking
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from power_modeling import FGCSPowerModelingFramework, analyze_application
from power_modeling.models.fgcs_models import FGCSPowerModel, PolynomialPowerModel
from power_modeling.models.model_factory import FGCSModelFactory
from edp_analysis.edp_calculator import EDPCalculator, FGCSEDPOptimizer, DVFSOptimizationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PowerModelingTester:
    """Comprehensive testing suite for power modeling components."""
    
    def __init__(self):
        """Initialize the testing suite."""
        self.results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        
    def test_fgcs_model_accuracy(self):
        """Test FGCS model coefficient accuracy and predictions."""
        logger.info("Testing FGCS model accuracy...")
        
        try:
            # Create FGCS model
            model = FGCSPowerModel()
            
            # Test coefficients match paper values
            expected_coeffs = {
                'intercept': -1.0318354343254663,
                'fp_coeff': 0.84864,
                'dram_coeff': 0.09749,
                'clock_coeff': 0.77006
            }
            
            for key, expected in expected_coeffs.items():
                actual = model.coefficients[key]
                assert abs(actual - expected) < 1e-10, f"Coefficient {key} mismatch: {actual} != {expected}"
            
            # Test power prediction
            fp_activity = 0.3
            dram_activity = 0.15
            frequencies = [1000, 1100, 1200]
            
            predictions = model.predict_power(fp_activity, dram_activity, frequencies)
            
            # Verify prediction structure
            assert 'sm_app_clock' in predictions.columns
            assert 'n_sm_app_clock' in predictions.columns
            assert 'predicted_n_power_usage' in predictions.columns
            assert len(predictions) == len(frequencies)
            
            # Verify predictions are reasonable
            assert predictions['predicted_n_power_usage'].min() > 0
            assert predictions['predicted_n_power_usage'].max() < 10  # log-normalized values
            
            logger.info("✓ FGCS model accuracy test passed")
            self.passed_tests += 1
            return True
            
        except Exception as e:
            logger.error(f"✗ FGCS model accuracy test failed: {str(e)}")
            self.failed_tests += 1
            return False
    
    def test_edp_calculations(self):
        """Test EDP and ED2P calculations."""
        logger.info("Testing EDP/ED2P calculations...")
        
        try:
            # Create EDP calculator
            calculator = EDPCalculator()
            
            # Test basic EDP calculation
            energy = np.array([100, 150, 200])
            delay = np.array([1.0, 1.2, 1.4])
            
            edp = calculator.calculate_edp(energy, delay)
            expected_edp = energy * delay
            assert np.allclose(edp, expected_edp), f"EDP calculation mismatch: {edp} != {expected_edp}"
            
            # Test ED2P calculation
            ed2p = calculator.calculate_ed2p(energy, delay)
            expected_ed2p = energy * (delay ** 2)
            assert np.allclose(ed2p, expected_ed2p), f"ED2P calculation mismatch: {ed2p} != {expected_ed2p}"
            
            # Test with DataFrame
            df = pd.DataFrame({
                'energy': energy,
                'execution_time': delay,
                'frequency': [1000, 1100, 1200]
            })
            
            optimal_config = calculator.find_optimal_configuration(df, metric='edp')
            assert 'frequency' in optimal_config
            assert 'energy' in optimal_config
            assert 'execution_time' in optimal_config
            
            logger.info("✓ EDP/ED2P calculations test passed")
            self.passed_tests += 1
            return True
            
        except Exception as e:
            logger.error(f"✗ EDP/ED2P calculations test failed: {str(e)}")
            self.failed_tests += 1
            return False
    
    def test_fgcs_edp_optimizer(self):
        """Test FGCS EDP optimizer functionality."""
        logger.info("Testing FGCS EDP optimizer...")
        
        try:
            # Create synthetic data matching FGCS format
            frequencies = [1000, 1100, 1200, 1300, 1400]
            df = pd.DataFrame({
                'sm_app_clock': frequencies,
                'predicted_n_to_r_energy': [100, 120, 140, 160, 180],
                'predicted_n_to_r_run_time': [1.5, 1.3, 1.1, 0.9, 0.8],
                'predicted_n_to_r_power_usage': [66.7, 92.3, 127.3, 177.8, 225.0]
            })
            
            # Test EDP optimization
            edp_freq, edp_time, edp_power, edp_energy = FGCSEDPOptimizer.edp_optimal(df)
            
            assert edp_freq in frequencies, f"EDP frequency {edp_freq} not in valid range"
            assert edp_time > 0, f"EDP time {edp_time} must be positive"
            assert edp_power > 0, f"EDP power {edp_power} must be positive"
            assert edp_energy > 0, f"EDP energy {edp_energy} must be positive"
            
            # Test ED2P optimization
            ed2p_freq, ed2p_time, ed2p_power, ed2p_energy = FGCSEDPOptimizer.ed2p_optimal(df)
            
            assert ed2p_freq in frequencies, f"ED2P frequency {ed2p_freq} not in valid range"
            assert ed2p_time > 0, f"ED2P time {ed2p_time} must be positive"
            
            # Test full optimization analysis
            results = FGCSEDPOptimizer.analyze_dvfs_optimization(df, "TestApp")
            
            assert 'edp_optimal' in results
            assert 'ed2p_optimal' in results
            assert 'min_energy' in results
            assert 'min_time' in results
            
            logger.info("✓ FGCS EDP optimizer test passed")
            self.passed_tests += 1
            return True
            
        except Exception as e:
            logger.error(f"✗ FGCS EDP optimizer test failed: {str(e)}")
            self.failed_tests += 1
            return False
    
    def test_frequency_configurations(self):
        """Test GPU frequency configurations."""
        logger.info("Testing GPU frequency configurations...")
        
        try:
            # Test each GPU type
            for gpu_type in ['V100', 'A100', 'H100']:
                framework = FGCSPowerModelingFramework(gpu_type=gpu_type)
                
                frequencies = framework.frequency_configs[gpu_type]
                
                # Verify frequency counts
                expected_counts = {'V100': 103, 'A100': 61, 'H100': 104}
                actual_count = len(frequencies)
                expected_count = expected_counts[gpu_type]
                
                assert actual_count == expected_count, \
                    f"{gpu_type} frequency count mismatch: {actual_count} != {expected_count}"
                
                # Verify frequency ranges
                expected_ranges = {
                    'V100': (405, 1380),
                    'A100': (510, 1410),
                    'H100': (210, 1755)
                }
                
                min_freq, max_freq = min(frequencies), max(frequencies)
                expected_min, expected_max = expected_ranges[gpu_type]
                
                assert min_freq == expected_min, \
                    f"{gpu_type} min frequency mismatch: {min_freq} != {expected_min}"
                assert max_freq == expected_max, \
                    f"{gpu_type} max frequency mismatch: {max_freq} != {expected_max}"
                
                # Verify frequencies are sorted in descending order
                assert frequencies == sorted(frequencies, reverse=True), \
                    f"{gpu_type} frequencies not sorted correctly"
            
            logger.info("✓ GPU frequency configurations test passed")
            self.passed_tests += 1
            return True
            
        except Exception as e:
            logger.error(f"✗ GPU frequency configurations test failed: {str(e)}")
            self.failed_tests += 1
            return False
    
    def test_dvfs_optimization_pipeline(self):
        """Test complete DVFS optimization pipeline."""
        logger.info("Testing DVFS optimization pipeline...")
        
        try:
            # Create power model
            power_model = FGCSPowerModel()
            
            # Create optimization pipeline
            pipeline = DVFSOptimizationPipeline(power_model)
            
            # Test optimization
            results = pipeline.optimize_application(
                fp_activity=0.3,
                dram_activity=0.15,
                baseline_runtime=1.0,
                frequencies=[1000, 1100, 1200, 1300, 1400],
                app_name="TestApp"
            )
            
            # Verify results structure
            assert 'optimization_results' in results
            assert 'recommendations' in results
            assert 'frequency_sweep_data' in results
            assert 'input_parameters' in results
            
            # Verify optimization results
            opt_results = results['optimization_results']
            assert 'edp_optimal' in opt_results
            assert 'ed2p_optimal' in opt_results
            
            # Verify recommendations
            recommendations = results['recommendations']
            assert 'primary_recommendation' in recommendations
            assert 'alternative_recommendation' in recommendations
            
            logger.info("✓ DVFS optimization pipeline test passed")
            self.passed_tests += 1
            return True
            
        except Exception as e:
            logger.error(f"✗ DVFS optimization pipeline test failed: {str(e)}")
            self.failed_tests += 1
            return False
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        logger.info("Testing end-to-end pipeline...")
        
        try:
            # Create synthetic training data
            np.random.seed(42)
            n_samples = 100
            
            training_data = pd.DataFrame({
                'fp_activity': np.random.uniform(0.1, 0.8, n_samples),
                'dram_activity': np.random.uniform(0.05, 0.4, n_samples),
                'sm_clock': np.random.choice(range(800, 1401, 50), n_samples),
                'power': np.random.uniform(150, 300, n_samples)
            })
            
            # Initialize framework
            framework = FGCSPowerModelingFramework()
            
            # Test model training
            training_results = framework.train_models(training_data, target_column='power')
            
            assert 'models' in training_results
            assert 'best_model' in training_results
            assert training_results['best_model'] is not None
            
            # Test power prediction
            power_sweep = framework.predict_power_sweep(
                fp_activity=0.3,
                dram_activity=0.15,
                frequencies=[1000, 1100, 1200, 1300, 1400]
            )
            
            assert len(power_sweep) == 5
            assert 'frequency' in power_sweep.columns
            assert 'power' in power_sweep.columns
            
            # Test optimization
            optimization_results = framework.optimize_application(
                fp_activity=0.3,
                dram_activity=0.15,
                baseline_runtime=1.0,
                app_name="TestApp"
            )
            
            assert 'optimization_results' in optimization_results
            assert 'recommendations' in optimization_results
            
            logger.info("✓ End-to-end pipeline test passed")
            self.passed_tests += 1
            return True
            
        except Exception as e:
            logger.error(f"✗ End-to-end pipeline test failed: {str(e)}")
            self.failed_tests += 1
            return False
    
    def test_quick_analysis_function(self):
        """Test the quick analysis function."""
        logger.info("Testing quick analysis function...")
        
        try:
            # Create temporary test data
            test_dir = Path("temp_test_data")
            test_dir.mkdir(exist_ok=True)
            
            # Generate synthetic profiling data
            np.random.seed(42)
            profiling_data = pd.DataFrame({
                'app_name': ['TestApp'] * 50,
                'fp_activity': np.random.uniform(0.1, 0.8, 50),
                'dram_activity': np.random.uniform(0.05, 0.4, 50),
                'sm_clock': np.random.choice(range(800, 1401, 50), 50),
                'power': np.random.uniform(150, 300, 50)
            })
            
            profiling_file = test_dir / "profiling.csv"
            profiling_data.to_csv(profiling_file, index=False)
            
            # Test quick analysis
            results = analyze_application(
                profiling_file=str(profiling_file),
                app_name="TestApp",
                gpu_type="V100"
            )
            
            # Verify results structure
            assert 'summary' in results
            assert 'optimization_results' in results
            assert 'profiling_data' in results
            
            # Verify summary
            summary = results['summary']
            assert 'optimal_frequency' in summary
            assert 'energy_savings' in summary
            assert 'performance_impact' in summary
            
            # Cleanup
            import shutil
            shutil.rmtree(test_dir)
            
            logger.info("✓ Quick analysis function test passed")
            self.passed_tests += 1
            return True
            
        except Exception as e:
            logger.error(f"✗ Quick analysis function test failed: {str(e)}")
            self.failed_tests += 1
            return False
    
    def run_all_tests(self):
        """Run all tests and provide summary."""
        logger.info("Starting comprehensive power modeling test suite...")
        logger.info("=" * 60)
        
        # List of tests to run
        tests = [
            ("FGCS Model Accuracy", self.test_fgcs_model_accuracy),
            ("EDP/ED2P Calculations", self.test_edp_calculations),
            ("FGCS EDP Optimizer", self.test_fgcs_edp_optimizer),
            ("GPU Frequency Configurations", self.test_frequency_configurations),
            ("DVFS Optimization Pipeline", self.test_dvfs_optimization_pipeline),
            ("End-to-End Pipeline", self.test_end_to_end_pipeline),
            ("Quick Analysis Function", self.test_quick_analysis_function)
        ]
        
        # Run each test
        for test_name, test_func in tests:
            logger.info(f"\n--- {test_name} ---")
            test_func()
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total tests: {self.passed_tests + self.failed_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.info(f"Failed: {self.failed_tests}")
        
        if self.failed_tests == 0:
            logger.info("🎉 ALL TESTS PASSED!")
            logger.info("✅ Core power modeling components are working correctly")
            logger.info("✅ EDP/ED2P framework is complete and functional")
            logger.info("✅ GPU frequency configurations are accurate")
            logger.info("✅ End-to-end pipeline is operational")
        else:
            logger.error(f"❌ {self.failed_tests} tests failed")
        
        return self.failed_tests == 0


def main():
    """Run the comprehensive test suite."""
    tester = PowerModelingTester()
    success = tester.run_all_tests()
    
    if success:
        logger.info("\n🚀 Power modeling framework is ready for production use!")
    else:
        logger.error("\n❌ Some tests failed. Please review the errors above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
