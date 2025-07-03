#!/usr/bin/env python3
"""Test end-to-end pipeline to find the issue."""

import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from power_modeling import FGCSPowerModelingFramework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_end_to_end():
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
        print('1. Training models...')
        training_results = framework.train_models(training_data, target_column='power')
        
        assert 'models' in training_results
        assert 'best_model' in training_results
        assert training_results['best_model'] is not None
        print('✓ Model training passed')
        
        # Test power prediction
        print('2. Testing power prediction...')
        power_sweep = framework.predict_power_sweep(
            fp_activity=0.3,
            dram_activity=0.15,
            frequencies=[1000, 1100, 1200, 1300, 1400]
        )
        
        assert len(power_sweep) == 5
        assert 'frequency' in power_sweep.columns
        assert 'power' in power_sweep.columns
        print('✓ Power prediction passed')
        
        # Test optimization
        print('3. Testing optimization...')
        optimization_results = framework.optimize_application(
            fp_activity=0.3,
            dram_activity=0.15,
            baseline_runtime=1.0,
            app_name="TestApp"
        )
        
        print('Optimization results keys:', list(optimization_results.keys()))
        assert 'optimization_results' in optimization_results
        assert 'recommendations' in optimization_results
        print('✓ Optimization passed')
        
        print('\n✓ All end-to-end tests passed!')
        return True
        
    except Exception as e:
        print(f'\n✗ Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_end_to_end()
    sys.exit(0 if success else 1)
