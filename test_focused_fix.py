#!/usr/bin/env python3
"""Focused test to verify the ModelEvaluator fix works in context."""

import logging
import numpy as np
import pandas as pd
import tempfile
import os
from power_modeling.fgcs_integration import FGCSPowerModelingFramework
from power_modeling.models.model_factory import ModelPipeline

# Setup logging
logging.basicConfig(level=logging.ERROR)

def test_end_to_end_pipeline():
    """Test the end-to-end pipeline that was failing."""
    print("Testing end-to-end pipeline...")
    
    # Create framework
    framework = FGCSPowerModelingFramework(gpu_type='V100', model_types=['polynomial_deg2'])
    
    # Create test data files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create mock profiling data
        profiling_file = os.path.join(tmp_dir, 'profiling.csv')
        with open(profiling_file, 'w') as f:
            f.write('col1,col2,col3,col4,col5\n')
            for i in range(50):
                f.write(f'{np.random.rand()},{np.random.rand()},{np.random.rand()},{np.random.rand()},{np.random.rand()}\n')
        
        # Run the pipeline
        try:
            result = framework.analyze_from_file(profiling_file, app_name='TestApp')
            print(f'✓ End-to-end pipeline successful: {len(result)} result keys')
            return True
        except Exception as e:
            print(f'✗ End-to-end pipeline failed: {e}')
            import traceback
            traceback.print_exc()
            return False

def test_model_pipeline():
    """Test the ModelPipeline directly."""
    print("Testing ModelPipeline.train_models...")
    
    # Create sample data
    X = np.random.rand(50, 5)
    y = np.random.rand(50) * 100
    X_train, X_test = X[:40], X[40:]
    y_train, y_test = y[:40], y[40:]
    
    try:
        pipeline = ModelPipeline(model_types=['polynomial_deg2'])
        result = pipeline.train_models(X_train, y_train, X_test, y_test)
        print(f'✓ ModelPipeline successful: {len(result["models"])} models trained')
        return True
    except Exception as e:
        print(f'✗ ModelPipeline failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running focused tests to verify ModelEvaluator fix...")
    print("=" * 50)
    
    # Test 1: Direct ModelPipeline test
    success1 = test_model_pipeline()
    
    print()
    
    # Test 2: End-to-end pipeline test
    success2 = test_end_to_end_pipeline()
    
    print()
    print("=" * 50)
    if success1 and success2:
        print("✓ All tests passed! ModelEvaluator fix is working.")
    else:
        print("✗ Some tests failed.")
