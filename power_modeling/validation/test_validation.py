"""
Model Validation Script

This script provides comprehensive validation for all power models in the framework.
It tests model accuracy, cross-validation performance, and energy-specific metrics.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from power_modeling.validation import PowerModelValidator, generate_validation_report
from power_modeling.models.model_factory import FGCSModelFactory, ModelPipeline
from power_modeling.models.fgcs_models import FGCSPowerModel, PolynomialPowerModel
from power_modeling.models.ensemble_models import EnhancedRandomForestModel, XGBoostPowerModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate realistic test data for model validation."""
    np.random.seed(42)
    
    # Generate features
    fp_activity = np.random.uniform(0.1, 0.8, n_samples)
    dram_activity = np.random.uniform(0.05, 0.4, n_samples)
    frequencies = np.random.choice(range(800, 1401, 50), n_samples)
    
    # Generate realistic power values
    power = (
        100 +  # Base power
        200 * fp_activity +  # FP contribution
        80 * dram_activity +  # DRAM contribution
        0.05 * frequencies +  # Frequency contribution
        np.random.normal(0, 8, n_samples)  # Noise
    )
    
    X = np.column_stack([fp_activity, dram_activity, frequencies])
    y = power
    
    return X, y


def test_individual_models():
    """Test individual model implementations."""
    logger.info("Testing individual model implementations")
    
    # Generate test data
    X, y = generate_test_data(500)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Test models
    models = {
        'FGCS_Original': FGCSPowerModel(),
        'Polynomial_Deg2': PolynomialPowerModel(degree=2),
        'Enhanced_RF': EnhancedRandomForestModel(),
    }
    
    # Add XGBoost if available
    try:
        models['XGBoost'] = XGBoostPowerModel()
    except ImportError:
        logger.warning("XGBoost not available, skipping XGBoost model")
    
    # Train models
    trained_models = {}
    for name, model in models.items():
        try:
            logger.info(f"Training {name}")
            
            # Prepare features for training
            if hasattr(model, 'fit'):
                # For models that expect pandas DataFrame
                feature_names = ['fp_activity', 'dram_activity', 'sm_clock']
                X_train_df = pd.DataFrame(X_train, columns=feature_names)
                y_train_series = pd.Series(y_train)
                
                model.fit(X_train_df, y_train_series)
                trained_models[name] = model
                logger.info(f"✓ {name} trained successfully")
            else:
                logger.warning(f"Model {name} does not have fit method")
                
        except Exception as e:
            logger.error(f"Failed to train {name}: {str(e)}")
    
    return trained_models, X_test, y_test


def test_model_pipeline():
    """Test the complete model pipeline."""
    logger.info("Testing model pipeline")
    
    # Generate test data
    X, y = generate_test_data(500)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create pipeline
    pipeline = ModelPipeline(
        model_types=['fgcs_original', 'polynomial_deg2', 'random_forest_enhanced']
    )
    
    # Train models
    results = pipeline.train_models(X_train, y_train, X_test, y_test)
    
    logger.info(f"Pipeline training completed. Best model: {results['best_model'][0]}")
    
    return results


def run_comprehensive_validation():
    """Run comprehensive model validation."""
    logger.info("Running comprehensive model validation")
    
    # Test individual models
    trained_models, X_test, y_test = test_individual_models()
    
    if not trained_models:
        logger.error("No models were successfully trained")
        return
    
    # Initialize validator
    validator = PowerModelValidator()
    
    # Run model comparison
    comparison_results = validator.compare_models(trained_models, X_test, y_test)
    
    # Generate validation report
    report = generate_validation_report(comparison_results)
    
    print("\n" + "="*50)
    print("POWER MODEL VALIDATION REPORT")
    print("="*50)
    print(report)
    
    # Save report
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "model_validation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Validation report saved to {report_file}")
    
    # Test frequency predictions
    logger.info("Testing frequency predictions")
    
    best_model_name = comparison_results['best_model']
    best_model = trained_models[best_model_name]
    
    freq_validation = validator.validate_frequency_predictions(
        best_model, 
        fp_activity=0.3, 
        dram_activity=0.15,
        frequencies=[800, 900, 1000, 1100, 1200, 1300, 1400]
    )
    
    print(f"\nFrequency Prediction Validation ({best_model_name}):")
    print(freq_validation['predictions'])
    print(f"Monotonic increasing: {freq_validation['characteristics']['monotonic_increasing']}")
    print(f"Power range: {freq_validation['characteristics']['power_range']}")
    print(f"Reasonable range: {freq_validation['characteristics']['reasonable_range']}")
    
    return comparison_results


def main():
    """Main validation function."""
    try:
        logger.info("Starting power model validation")
        
        # Run comprehensive validation
        results = run_comprehensive_validation()
        
        # Test model pipeline
        pipeline_results = test_model_pipeline()
        
        logger.info("✅ All validation tests completed successfully!")
        
        return results, pipeline_results
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
