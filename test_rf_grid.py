#!/usr/bin/env python3
"""
Simple test to verify Random Forest hyperparameter grid fix.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Test the exact hyperparameter grid from our fixed code
param_grid = {
    'n_estimators': [200, 500, 800, 1000],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [10, 30, 50, 80, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['squared_error', 'absolute_error'],
    'bootstrap': [True]
    # max_samples completely removed
}

print("Testing Random Forest hyperparameter grid:")
for key, values in param_grid.items():
    print(f"  {key}: {values}")

# Test with minimal data
X_train = np.random.rand(50, 5)
y_train = np.random.rand(50)

print("\nTesting RandomizedSearchCV with fixed grid...")
try:
    # Create base model
    base_rf = RandomForestRegressor(random_state=42, n_jobs=1)
    
    # Test with a few iterations to ensure it works
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_grid,
        n_iter=5,  # Small number for quick test
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=1,
        scoring='neg_mean_squared_error'
    )
    
    search.fit(X_train, y_train)
    print("✓ RandomizedSearchCV completed successfully!")
    print(f"Best score: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
