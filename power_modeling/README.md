# Power Modeling Framework

This directory contains the complete power modeling framework extracted and modernized from the FGCS 2023 paper implementation. The framework provides comprehensive tools for GPU power prediction, EDP optimization, and energy-efficient frequency selection.

## 🚀 Quick Start

### Installation Requirements
```bash
# Required Python packages
pip install pandas numpy scikit-learn xgboost
pip install matplotlib seaborn  # For visualization
```

**Compatibility**: Python 3.6+ (tested with Python 3.6, 3.8, 3.9, 3.10+)

### Simple Analysis
```python
from power_modeling import analyze_application

# Quick analysis of an application
results = analyze_application(
    profiling_file="data/app_profiling.csv",
    app_name="MyApplication",
    gpu_type="V100"
)

print(f"Optimal frequency: {results['summary']['optimal_frequency']}")
print(f"Energy savings: {results['summary']['energy_savings']}")
```

### Full Framework Usage
```python
from power_modeling import FGCSPowerModelingFramework
import pandas as pd

# Initialize framework
framework = FGCSPowerModelingFramework(
    model_types=['fgcs_original', 'polynomial_deg2', 'random_forest_enhanced'],
    gpu_type='V100'
)

# Load and analyze data
results = framework.analyze_from_file(
    profiling_file="data/profiling.csv",
    performance_file="data/performance.csv",
    app_name="MyApp"
)

# Save results
framework.save_results(results, output_dir="results/")
```

## 📊 Model Types

### 1. FGCS Original Model
- **Description**: Exact implementation from FGCS 2023 paper
- **Features**: FP operations, DRAM activity, SM clock frequency
- **Use Case**: Reproduction of original paper results

### 2. Polynomial Models
- **Description**: Polynomial regression with configurable degree
- **Features**: Polynomial expansion of input features
- **Use Case**: Simple, interpretable power prediction

### 3. Enhanced Random Forest
- **Description**: Tuned Random Forest with feature engineering
- **Features**: Ensemble of decision trees with optimized hyperparameters
- **Use Case**: High accuracy with feature importance analysis

### 4. XGBoost Model
- **Description**: Gradient boosting with advanced regularization
- **Features**: GPU-accelerated training, automatic feature selection
- **Use Case**: Maximum prediction accuracy for complex patterns

## 🔧 Training Custom Models

```python
import pandas as pd
from power_modeling import FGCSPowerModelingFramework

# Prepare training data
training_data = pd.DataFrame({
    'fp_activity': [0.1, 0.2, 0.3, 0.4, 0.5],
    'dram_activity': [0.05, 0.1, 0.15, 0.2, 0.25],
    'sm_clock': [1000, 1100, 1200, 1300, 1400],
    'power': [150, 180, 210, 240, 270]
})

# Train models
framework = FGCSPowerModelingFramework()
training_results = framework.train_models(
    training_data, 
    target_column='power',
    test_size=0.2
)

print(f"Best model: {training_results['best_model'][0]}")
print(f"R² score: {training_results['best_model'][1]:.4f}")
```

## 📈 Power Prediction

```python
# Predict power across frequency range
power_sweep = framework.predict_power_sweep(
    fp_activity=0.3,
    dram_activity=0.15,
    frequencies=[800, 900, 1000, 1100, 1200]
)

print(power_sweep)
```

## ⚡ EDP Optimization

```python
# Optimize application for energy-delay product
optimization_results = framework.optimize_application(
    fp_activity=0.3,
    dram_activity=0.15,
    baseline_runtime=1.0,
    app_name="MyApp"
)

print(f"EDP optimal frequency: {optimization_results['edp_optimal']['frequency']}")
print(f"ED²P optimal frequency: {optimization_results['ed2p_optimal']['frequency']}")
```

## 🎯 GPU Support

### Supported GPUs
- **V100**: Tesla V100 (default)
- **A100**: A100 series
- **H100**: H100 series

### Frequency Ranges
- **V100**: 405-1380 MHz (7 MHz steps)
- **A100**: 210-1410 MHz (predefined frequencies)
- **H100**: 210-1980 MHz (17 MHz steps)

## 📁 File Structure

```
power_modeling/
├── __init__.py                    # Main exports
├── fgcs_integration.py           # High-level framework interface
├── models/
│   ├── fgcs_models.py           # FGCS and polynomial models
│   ├── ensemble_models.py       # Random Forest, XGBoost models
│   └── model_factory.py         # Model factory and pipeline
├── feature_engineering/
│   └── preprocessing.py         # Data preprocessing utilities
└── validation/
    └── model_validation.py      # Model validation utilities
```

## 🔍 Data Format

### Profiling Data Format
```csv
app_name,fp_activity,dram_activity,sm_clock,power
MyApp,0.25,0.12,1000,165.5
MyApp,0.25,0.12,1100,182.3
MyApp,0.25,0.12,1200,198.7
```

### Performance Data Format
```csv
app_name,frequency,runtime,throughput
MyApp,1000,1.25,800
MyApp,1100,1.15,850
MyApp,1200,1.08,900
```

## 📊 Output Analysis

### Optimization Results
```python
{
    'edp_optimal': {
        'frequency': 1050,
        'power': 175.2,
        'runtime': 1.12,
        'edp': 196.224
    },
    'ed2p_optimal': {
        'frequency': 950,
        'power': 158.3,
        'runtime': 1.18,
        'ed2p': 220.175
    },
    'recommendations': {
        'primary_recommendation': {
            'frequency': 1050,
            'expected_energy_savings': '12.5%',
            'expected_performance_impact': '8.0%',
            'reason': 'Optimal EDP trade-off'
        }
    }
}
```

## 🧪 Validation and Testing

The framework includes comprehensive validation:
- Cross-validation for model training
- Performance metrics calculation
- Statistical significance testing
- Model comparison and selection

### Recent Robustness Improvements (Latest Release)

**Production-Ready Error Handling**:
- ✅ **EDP Calculation Robustness**: Enhanced error handling for division by zero in EDP improvement calculations
- ✅ **Model Access Fixes**: Fixed `TypeError` in model access - no longer subscriptable errors
- ✅ **Baseline Validation**: Automatic detection and logging of invalid baseline values (zero or NaN)
- ✅ **Graceful Fallbacks**: Robust fallback behavior when baseline data is insufficient
- ✅ **Runtime Warning Elimination**: All warnings resolved through comprehensive validation

**Error Handling Features**:
```python
# The framework now automatically handles edge cases
results = framework.optimize_application(
    fp_activity=0.3,
    dram_activity=0.15,
    baseline_runtime=0.0  # Previously would cause runtime warnings
)
# Now logs warning and handles gracefully: "Invalid baseline runtime for EDP calculation"

# Model access is now robust
training_results = framework.train_models(training_data, target_column='power')
best_model = training_results['best_model']  # Direct access - no subscriptable errors
print(f"Best model: {best_model['name']}")  # Proper dictionary access
```

## 🔬 Research Extensions

The framework is designed for extensibility:
- Custom model integration
- Additional optimization metrics
- Multi-objective optimization
- Uncertainty quantification

## 🛠️ Troubleshooting and Best Practices

### Common Issues and Solutions

**Data Quality Checks**:
- Ensure all profiling data contains positive, non-zero values
- Validate that baseline measurements are collected under consistent conditions
- Check for missing or NaN values in input datasets

**Model Training Best Practices**:
- Use sufficient training data (minimum 100 samples recommended)
- Validate frequency ranges match your target GPU
- Cross-validate models before deployment

**EDP Optimization Tips**:
- Ensure baseline runtime measurements are accurate and representative
- Consider multiple optimization metrics (EDP, ED²P) for different use cases
- Validate optimization results with actual hardware measurements

### Logging and Debugging

The framework provides comprehensive logging:
```python
import logging
logging.basicConfig(level=logging.INFO)

# Framework will now log detailed information about:
# - Model training progress
# - Data validation warnings
# - EDP calculation issues
# - Optimization convergence
```

## 📚 References

1. FGCS 2023 Paper: "Energy-Efficient GPU Frequency Selection for Deep Learning Inference"
2. Original `gpupowermodel` implementation
3. Enhanced methodology for modern AI workloads

## 🤝 Contributing

To add new models or features:
1. Implement model in appropriate module
2. Add to `model_factory.py`
3. Update `fgcs_integration.py` if needed
4. Add tests in `validation/`
5. Update documentation

## 📄 License

This framework is part of the AI Inference Energy project by Mert Side.
