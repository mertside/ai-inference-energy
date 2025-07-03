#!/usr/bin/env python3
"""Debug the end-to-end test failure."""

import logging
import numpy as np
import pandas as pd
from power_modeling.fgcs_integration import FGCSPowerModelingFramework

logging.basicConfig(level=logging.ERROR)

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
    print("Initializing framework...")
    framework = FGCSPowerModelingFramework()
    
    # Test model training
    print('Starting model training...')
    training_results = framework.train_models(training_data, target_column='power')
    print('Model training completed')
    print(f"Training results keys: {list(training_results.keys())}")
    print(f"Best model: {training_results.get('best_model')}")
    
    # Test power prediction
    print('Starting power prediction...')
    power_sweep = framework.predict_power_sweep(
        fp_activity=0.3,
        dram_activity=0.15,
        frequencies=[1000, 1100, 1200, 1300, 1400]
    )
    print('Power prediction completed')
    print(f"Power sweep shape: {power_sweep.shape}")
    print(f"Power sweep columns: {list(power_sweep.columns)}")
    
    print('✓ END-TO-END TEST PASSED!')
    
except Exception as e:
    print(f'✗ END-TO-END TEST FAILED: {e}')
    import traceback
    traceback.print_exc()
