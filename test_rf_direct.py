#!/usr/bin/env python3
"""
Test just the Random Forest hyperparameter grid fix directly.
"""

import sys
sys.path.insert(0, '/Users/MertSide/Developer/GitProjects/ai-inference-energy')

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Import the class directly
from power_modeling.models.ensemble_models import EnhancedRandomForestModel

def test_rf_grid():
    print("Testing Random Forest hyperparameter grid...")
    
    # Create model
    model = EnhancedRandomForestModel()
    
    # Get grid
    grid = model.get_hyperparameter_grid()
    print(f"Grid has {len(grid)} parameters")
    
    # Check if max_samples is in grid
    has_max_samples = 'max_samples' in grid
    print(f"Has max_samples: {has_max_samples}")
    
    if has_max_samples:
        print(f"max_samples values: {grid['max_samples']}")
    
    # Test with sample data
    X_train = np.random.rand(50, 5)
    y_train = np.random.rand(50)
    
    print("\nTesting model training without optimization...")
    try:
        model.fit(X_train, y_train, optimize=False)
        print("✓ Training successful")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False
    
    print("\nTesting model training with optimization (small grid)...")
    try:
        model_opt = EnhancedRandomForestModel(optimization_method='random', n_iter=3)
        model_opt.fit(X_train, y_train, optimize=True)
        print("✓ Optimization successful")
    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        return False
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    test_rf_grid()
