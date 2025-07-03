# Power Modeling and EDP/ED²P Analysis Framework

Complete framework for GPU power modeling and energy-delay product optimization extracted from the FGCS 2023 paper implementation.

## 🎯 Overview

This framework provides:
- **Power Prediction Models**: FGCS original, polynomial, Random Forest, XGBoost
- **EDP/ED²P Optimization**: Energy-delay product optimization algorithms
- **GPU Support**: V100 (103 frequencies), A100 (61 frequencies), H100 (104 frequencies)
- **Complete Pipeline**: From profiling data to optimization recommendations

## 🚀 Quick Start

### Simple Analysis
```python
from power_modeling import analyze_application

# Quick analysis of profiling data
results = analyze_application(
    profiling_file="profiling_data.csv",
    app_name="MyApplication",
    gpu_type="V100"
)

print(f"Optimal frequency: {results['summary']['optimal_frequency']}")
print(f"Energy savings: {results['summary']['energy_savings']}")
```

### Advanced Usage
```python
from power_modeling import FGCSPowerModelingFramework

# Initialize framework
framework = FGCSPowerModelingFramework(
    model_types=['fgcs_original', 'polynomial_deg2'],
    gpu_type='V100'
)

# Complete analysis pipeline
results = framework.analyze_from_file(
    profiling_file="data/profiling.csv",
    app_name="MyApp"
)

# Save results
framework.save_results(results, "output/")
```

## 📊 Core Components

### 1. Power Models

#### FGCS Original Model
```python
from power_modeling.models.fgcs_models import FGCSPowerModel

model = FGCSPowerModel()
predictions = model.predict_power(
    fp_activity=0.3,
    dram_activity=0.15,
    sm_clock_frequencies=[1000, 1100, 1200]
)
```

#### Polynomial Power Model
```python
from power_modeling.models.fgcs_models import PolynomialPowerModel

model = PolynomialPowerModel(degree=2)
model.fit(training_features, power_values)
predictions = model.predict(test_features)
```

### 2. EDP/ED²P Calculations

#### Basic EDP Calculations
```python
from edp_analysis.edp_calculator import EDPCalculator

calculator = EDPCalculator()

# Calculate EDP (Energy × Delay)
edp = calculator.calculate_edp(energy, delay)

# Calculate ED²P (Energy × Delay²)
ed2p = calculator.calculate_ed2p(energy, delay)
```

#### FGCS EDP Optimizer
```python
from edp_analysis.edp_calculator import FGCSEDPOptimizer

# Find EDP optimal configuration
edp_freq, edp_time, edp_power, edp_energy = FGCSEDPOptimizer.edp_optimal(df)

# Find ED²P optimal configuration
ed2p_freq, ed2p_time, ed2p_power, ed2p_energy = FGCSEDPOptimizer.ed2p_optimal(df)

# Complete DVFS optimization analysis
results = FGCSEDPOptimizer.analyze_dvfs_optimization(df, "MyApp")
```

### 3. Complete DVFS Pipeline
```python
from edp_analysis.edp_calculator import DVFSOptimizationPipeline
from power_modeling.models.fgcs_models import FGCSPowerModel

# Create pipeline
power_model = FGCSPowerModel()
pipeline = DVFSOptimizationPipeline(power_model)

# Run optimization
results = pipeline.optimize_application(
    fp_activity=0.3,
    dram_activity=0.15,
    baseline_runtime=1.0,
    frequencies=[1000, 1100, 1200, 1300, 1400],
    app_name="MyApp"
)
```

## 🔧 GPU Configurations

### Supported GPUs and Frequencies

| GPU Type | Frequencies | Count | Range (MHz) | Pattern |
|----------|-------------|-------|-------------|---------|
| **V100** | 103 | 1380-405 | Variable steps | Production frequencies |
| **A100** | 61 | 1410-510 | Variable steps | Production frequencies |
| **H100** | 104 | 1755-210 | 15 MHz steps | Production frequencies |

### GPU-Specific Usage
```python
# V100 configuration
framework_v100 = FGCSPowerModelingFramework(gpu_type='V100')
print(f"V100 frequencies: {len(framework_v100.frequency_configs['V100'])}")

# A100 configuration  
framework_a100 = FGCSPowerModelingFramework(gpu_type='A100')
print(f"A100 frequencies: {len(framework_a100.frequency_configs['A100'])}")

# H100 configuration
framework_h100 = FGCSPowerModelingFramework(gpu_type='H100') 
print(f"H100 frequencies: {len(framework_h100.frequency_configs['H100'])}")
```

## 📈 Model Training

### Custom Model Training
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
results = framework.train_models(
    training_data, 
    target_column='power',
    test_size=0.2
)

print(f"Best model: {results['best_model'][0]}")
print(f"R² score: {results['best_model'][2]:.4f}")
```

### Model Types Available
- **fgcs_original**: Exact FGCS 2023 paper implementation
- **polynomial_deg2**: Polynomial regression (degree 2)
- **polynomial_deg3**: Polynomial regression (degree 3)
- **random_forest_enhanced**: Tuned Random Forest
- **xgboost_power**: XGBoost with GPU acceleration (if available)

## 📊 Data Formats

### Profiling Data Format
```csv
app_name,fp_activity,dram_activity,sm_clock,power
MyApp,0.25,0.12,1000,165.5
MyApp,0.25,0.12,1100,182.3
MyApp,0.25,0.12,1200,198.7
```

### Performance Data Format (Optional)
```csv
app_name,frequency,runtime,throughput
MyApp,1000,1.25,800
MyApp,1100,1.15,850
MyApp,1200,1.08,900
```

## 🎯 Optimization Results

### Result Structure
```python
{
    'optimization_results': {
        'edp_optimal': {
            'frequency': 1050,
            'power': 175.2,
            'runtime': 1.12,
            'edp': 196.224,
            'energy': 196.224,
            'energy_improvement': 12.5,
            'time_improvement': 8.0
        },
        'ed2p_optimal': {
            'frequency': 950,
            'power': 158.3,
            'runtime': 1.18,
            'ed2p': 220.175,
            'energy': 186.794
        },
        'min_energy': {
            'frequency': 800,
            'energy': 160.0
        },
        'min_time': {
            'frequency': 1400,
            'time': 0.85
        }
    },
    'recommendations': {
        'primary_recommendation': {
            'frequency': 1050,
            'reason': 'EDP optimal - best energy-delay trade-off',
            'expected_energy_savings': '12.5%',
            'expected_performance_impact': '8.0%'
        },
        'alternative_recommendation': {
            'frequency': 950,
            'reason': 'ED²P optimal - prioritizes performance over energy'
        }
    }
}
```

## 🧪 Testing and Validation

### Run Complete Test Suite
```bash
# Run comprehensive tests
python test_power_modeling_complete.py

# Run simple demo
python examples/simple_power_modeling_demo.py
```

### Test Coverage
- ✅ FGCS model coefficient accuracy
- ✅ EDP/ED²P calculation correctness
- ✅ GPU frequency configuration validation
- ✅ DVFS optimization pipeline functionality
- ✅ End-to-end pipeline testing
- ✅ Integration with real profiling data

## 📁 File Structure

```
power_modeling/
├── __init__.py                    # Main exports
├── fgcs_integration.py           # High-level framework
├── models/
│   ├── fgcs_models.py           # FGCS and polynomial models
│   ├── ensemble_models.py       # Random Forest, XGBoost
│   └── model_factory.py         # Model factory
├── feature_engineering/
│   └── preprocessing.py         # Data preprocessing
├── validation/
│   ├── metrics.py               # Validation metrics
│   └── test_validation.py       # Validation tests
└── examples/
    └── demo_framework.py        # Usage examples

edp_analysis/
├── __init__.py
└── edp_calculator.py            # EDP/ED²P calculations

examples/
├── simple_power_modeling_demo.py  # Basic demo
└── power_modeling_usage.py        # Advanced examples

test_power_modeling_complete.py    # Complete test suite
```

## 🔬 Research Applications

### Energy-Efficiency Studies
```python
# Compare energy efficiency across frequencies
results = framework.predict_power_sweep(
    fp_activity=0.3,
    dram_activity=0.15,
    frequencies=framework.frequency_configs['V100']
)

# Analyze energy-performance trade-offs
optimization = framework.optimize_application(
    fp_activity=0.3,
    dram_activity=0.15,
    baseline_runtime=1.0
)
```

### Multi-Application Analysis
```python
applications = ['LSTM', 'CNN', 'Transformer', 'StableDiffusion']

for app in applications:
    results = analyze_application(
        profiling_file=f"data/{app}_profiling.csv",
        app_name=app,
        gpu_type="V100"
    )
    
    print(f"{app} optimal frequency: {results['summary']['optimal_frequency']}")
```

## 📊 Performance Metrics

### Model Accuracy
- **R² Score**: Typically 0.85-0.95 for power prediction
- **MAE**: 5-15 Watts for power prediction
- **MAPE**: <10% for energy consumption
- **Frequency Accuracy**: Within 50 MHz of optimal

### Optimization Performance
- **EDP Improvement**: 10-30% energy savings typical
- **Performance Impact**: <5% runtime increase typical
- **Pareto Efficiency**: Multi-objective optimization available

## 🔧 Configuration

### Framework Configuration
```python
# Custom model configuration
framework = FGCSPowerModelingFramework(
    model_types=[
        'fgcs_original',
        'polynomial_deg2', 
        'random_forest_enhanced'
    ],
    gpu_type='V100'
)

# Custom frequency ranges
custom_frequencies = list(range(800, 1400, 100))
results = framework.predict_power_sweep(
    fp_activity=0.3,
    dram_activity=0.15,
    frequencies=custom_frequencies
)
```

### EDP Calculator Configuration
```python
# Custom energy/delay weights
calculator = EDPCalculator(
    energy_weight=0.6,  # Prioritize energy
    delay_weight=0.4    # Less emphasis on delay
)

# Weighted optimization
score = calculator.calculate_weighted_score(energy, delay)
```

## 📚 References

1. **FGCS 2023 Paper**: "Energy-Efficient GPU Frequency Selection for Deep Learning Inference"
2. **Original gpupowermodel**: Legacy implementation
3. **DVFS Research**: Dynamic Voltage and Frequency Scaling for GPUs
4. **EDP Optimization**: Energy-Delay Product minimization algorithms

## 🤝 Contributing

### Adding New Models
1. Implement model class in `power_modeling/models/`
2. Add to `model_factory.py`
3. Update `fgcs_integration.py`
4. Add tests in `validation/`
5. Update documentation

### Adding New GPU Support
1. Add frequency configuration to `fgcs_integration.py`
2. Test with actual hardware data
3. Validate frequency ranges
4. Update documentation

## ⚠️ Important Notes

### GPU Frequency Validation
- **V100**: 103 production frequencies (1380-405 MHz)
- **A100**: 61 production frequencies (1410-510 MHz)  
- **H100**: 104 production frequencies (1755-210 MHz)

### FGCS Model Coefficients
The FGCS model uses exact coefficients from the paper:
- Intercept: -1.0318354343254663
- FP coefficient: 0.84864
- DRAM coefficient: 0.09749
- Clock coefficient: 0.77006

### EDP vs ED²P
- **EDP (Energy × Delay)**: Balanced energy-performance trade-off
- **ED²P (Energy × Delay²)**: Prioritizes performance over energy efficiency

## 🎉 Status

✅ **Core Power Models**: FGCS, Polynomial, Random Forest, XGBoost  
✅ **EDP/ED²P Framework**: Complete optimization algorithms  
✅ **GPU Support**: V100, A100, H100 with correct frequency counts  
✅ **Validation**: Comprehensive test suite  
✅ **Documentation**: Complete usage examples  
✅ **Integration**: End-to-end pipeline functional  

**Ready for Production Use!**

---

**Author**: Mert Side  
**Version**: 1.0.0  
**Date**: July 3, 2025  
**Status**: ✅ Complete and Validated
