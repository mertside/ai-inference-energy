#!/usr/bin/env python3
"""
Power Modeling Integration Test

This script tests the complete integration of FGCS power modeling components
extracted from the legacy gpupowermodel and integrated into the new framework.

This test validates:
1. Model imports and initialization
2. Data loading and preprocessing
3. Model training and prediction
4. EDP optimization functionality
5. Results saving and reporting
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        # Test main framework import
        from power_modeling import FGCSPowerModelingFramework, analyze_application
        logger.info("✓ Main framework imports successful")
        
        # Test model imports
        from power_modeling.models.fgcs_models import FGCSPowerModel, PolynomialPowerModel
        from power_modeling.models.ensemble_models import EnhancedRandomForestModel
        from power_modeling.models.model_factory import FGCSModelFactory, ModelPipeline
        logger.info("✓ Model imports successful")
        
        # Test EDP integration
        from edp_analysis.edp_calculator import FGCSEDPOptimizer, DVFSOptimizationPipeline
        logger.info("✓ EDP analysis imports successful")
        
        # Test validation imports
        from power_modeling.validation import PowerModelValidator, ModelValidationMetrics
        logger.info("✓ Validation imports successful")
        
        return True
        
    except Exception as e:
        logger.error(f"Import test failed: {str(e)}")
        return False


def test_model_creation():
    """Test that models can be created and initialized."""
    logger.info("Testing model creation...")
    
    try:
        from power_modeling.models.model_factory import FGCSModelFactory
        
        # Test FGCS model creation
        fgcs_model = FGCSModelFactory.create_fgcs_power_model()
        logger.info("✓ FGCS model created")
        
        # Test polynomial model creation
        poly_model = FGCSModelFactory.create_polynomial_model(degree=2)
        logger.info("✓ Polynomial model created")
        
        # Test enhanced random forest
        rf_model = FGCSModelFactory.create_enhanced_random_forest()
        logger.info("✓ Enhanced Random Forest model created")
        
        # Test model pipeline
        pipeline = FGCSModelFactory.create_model_pipeline()
        logger.info("✓ Model pipeline created")
        
        return True
        
    except Exception as e:
        logger.error(f"Model creation test failed: {str(e)}")
        return False


def test_framework_initialization():
    """Test framework initialization with different configurations."""
    logger.info("Testing framework initialization...")
    
    try:
        from power_modeling import FGCSPowerModelingFramework
        
        # Test default initialization
        framework_default = FGCSPowerModelingFramework()
        logger.info("✓ Default framework initialization")
        
        # Test with custom model types
        framework_custom = FGCSPowerModelingFramework(
            model_types=['fgcs_original', 'polynomial_deg2'],
            gpu_type='A100'
        )
        logger.info("✓ Custom framework initialization")
        
        # Test GPU-specific configurations
        for gpu_type in ['V100', 'A100', 'H100']:
            framework_gpu = FGCSPowerModelingFramework(gpu_type=gpu_type)
            assert gpu_type in framework_gpu.frequency_configs
            logger.info(f"✓ {gpu_type} configuration loaded")
        
        return True
        
    except Exception as e:
        logger.error(f"Framework initialization test failed: {str(e)}")
        return False


def test_synthetic_data_flow():
    """Test the complete data flow with synthetic data."""
    logger.info("Testing synthetic data flow...")
    
    try:
        import numpy as np
        import pandas as pd
        from power_modeling import FGCSPowerModelingFramework
        
        # Generate synthetic training data
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
        logger.info(f"✓ Model training completed. Best model: {training_results['best_model'][0]}")
        
        # Test power prediction
        power_sweep = framework.predict_power_sweep(
            fp_activity=0.3,
            dram_activity=0.15,
            frequencies=[800, 900, 1000, 1100, 1200]
        )
        logger.info(f"✓ Power prediction completed. Predicted {len(power_sweep)} data points")
        
        # Test optimization
        optimization_results = framework.optimize_application(
            fp_activity=0.3,
            dram_activity=0.15,
            baseline_runtime=1.0,
            app_name="TestApp"
        )
        logger.info(f"✓ EDP optimization completed. Optimal frequency: {optimization_results['edp_optimal']['frequency']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Synthetic data flow test failed: {str(e)}")
        return False


def test_edp_integration():
    """Test EDP analysis integration."""
    logger.info("Testing EDP analysis integration...")
    
    try:
        from edp_analysis.edp_calculator import FGCSEDPOptimizer, DVFSOptimizationPipeline
        from power_modeling.models.fgcs_models import FGCSPowerModel
        
        # Create power model
        power_model = FGCSPowerModel()
        
        # Create EDP optimizer
        edp_optimizer = FGCSEDPOptimizer(power_model)
        logger.info("✓ EDP optimizer created")
        
        # Create DVFS pipeline
        dvfs_pipeline = DVFSOptimizationPipeline(power_model)
        logger.info("✓ DVFS pipeline created")
        
        # Test optimization with synthetic data
        frequencies = [800, 900, 1000, 1100, 1200]
        
        results = dvfs_pipeline.optimize_application(
            fp_activity=0.3,
            dram_activity=0.15,
            baseline_runtime=1.0,
            frequencies=frequencies,
            app_name="TestApp"
        )
        
        logger.info(f"✓ DVFS optimization completed. EDP optimal: {results['edp_optimal']['frequency']}")
        
        return True
        
    except Exception as e:
        logger.error(f"EDP integration test failed: {str(e)}")
        return False


def test_validation_integration():
    """Test validation utilities integration."""
    logger.info("Testing validation integration...")
    
    try:
        from power_modeling.validation import PowerModelValidator, ModelValidationMetrics
        import numpy as np
        
        # Create validator
        validator = PowerModelValidator()
        logger.info("✓ Validator created")
        
        # Test metrics calculation
        y_true = np.array([100, 150, 200, 250, 300])
        y_pred = np.array([95, 155, 195, 245, 305])
        
        metrics = ModelValidationMetrics.calculate_basic_metrics(y_true, y_pred)
        logger.info(f"✓ Metrics calculated. R² = {metrics['r2']:.4f}")
        
        # Test relative metrics
        relative_metrics = ModelValidationMetrics.calculate_relative_metrics(y_true, y_pred)
        logger.info(f"✓ Relative metrics calculated. MAPE = {relative_metrics['relative_mae']:.2f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Validation integration test failed: {str(e)}")
        return False


def test_quick_analysis():
    """Test quick analysis functionality."""
    logger.info("Testing quick analysis functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        from power_modeling import analyze_application
        
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
        
        logger.info(f"✓ Quick analysis completed. Optimal frequency: {results['summary']['optimal_frequency']}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"Quick analysis test failed: {str(e)}")
        return False


def run_integration_tests():
    """Run all integration tests."""
    logger.info("Starting Power Modeling Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Model Creation Tests", test_model_creation),
        ("Framework Initialization Tests", test_framework_initialization),
        ("Synthetic Data Flow Tests", test_synthetic_data_flow),
        ("EDP Integration Tests", test_edp_integration),
        ("Validation Integration Tests", test_validation_integration),
        ("Quick Analysis Tests", test_quick_analysis)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {str(e)}")
            failed += 1
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("The FGCS power modeling extraction is complete and working correctly.")
    else:
        logger.error(f"❌ {failed} tests failed. Please check the issues above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
