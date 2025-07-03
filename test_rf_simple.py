#!/usr/bin/env python3
"""
Simple test to check Random Forest optimization
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from power_modeling.models.ensemble_models import EnhancedRandomForestModel
import numpy as np
from sklearn.model_selection import train_test_split

def test_random_forest_optimization():
    """Test Random Forest hyperparameter optimization"""
    print("Testing Random Forest optimization...")
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = np.random.rand(100) * 200 + 50  # Power values between 50-250W
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = EnhancedRandomForestModel()
    print("Starting Random Forest training...")
    model.fit(X_train, y_train)
    print("Random Forest training completed successfully!")
    
    # Test prediction
    predictions = model.predict(X_test)
    print(f"Made {len(predictions)} predictions")
    print(f"Prediction range: {predictions.min():.2f} - {predictions.max():.2f}")
    
    return True

if __name__ == "__main__":
    try:
        test_random_forest_optimization()
        print("✓ Random Forest optimization test passed")
    except Exception as e:
        print(f"✗ Random Forest optimization test failed: {e}")
        import traceback
        traceback.print_exc()
