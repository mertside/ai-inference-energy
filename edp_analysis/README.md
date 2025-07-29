# EDP Analysis Module

Enhanced Energy-Delay Product (EDP) and Energy-Delay² Product (ED²P) optimization framework inspired by FGCS 2023 methodology.

## 🔍 Overview

The EDP Analysis module provides a comprehensive framework for energy-performance optimization in GPU applications, implementing advanced methodologies from the "Energy-efficient DVFS scheduling for mixed-criticality systems" paper (FGCS 2023). This framework is designed for production use with robust error handling, advanced feature selection, and comprehensive visualization capabilities.

## ✨ Enhanced Features

### 🧠 **Advanced Feature Selection**
- FGCS-inspired feature engineering (FP activity, DRAM activity, clock frequencies)
- Statistical and model-based feature selection methods
- Correlation-based feature filtering and validation
- GPU-specific feature importance analysis

### 🎯 **Multi-Objective Optimization**
- EDP (Energy × Delay) optimization for balanced trade-offs
- ED²P (Energy × Delay²) optimization for performance-prioritized scenarios
- Pareto frontier analysis for multi-objective decision making
- Configuration recommendations with justification

### 📊 **Comprehensive Visualization**
- **Time-series profiling plots**: Any metric vs normalized time with multi-frequency overlays
- **Production-ready CLI tool**: `plot_metric_vs_time.py` for immediate visualization
- Feature importance plots for EDP optimization
- FGCS model validation and comparison plots
- Energy efficiency analysis and breakdown
- Performance scaling and throughput analysis
- Interactive EDP dashboards and comprehensive reports

### 🔬 **FGCS 2023 Integration**
- Exact implementation of FGCS power and performance models
- Validated coefficient sets for V100, A100, and H100 GPUs
- Log-transformed feature engineering for improved accuracy
- Statistical validation and model comparison

## 📊 Key Metrics

- **EDP (Energy-Delay Product)**: `Energy × Execution_Time`
- **ED²P (Energy-Delay² Product)**: `Energy × Execution_Time²`
- **Energy Improvement**: Percentage reduction in energy consumption vs baseline
- **Time Improvement**: Percentage reduction in execution time vs baseline
- **Feature Importance**: Relative contribution of features to EDP optimization
- **Energy Efficiency**: Performance per watt and energy per operation

## 🚀 Quick Usage

### Basic EDP Analysis
```python
from edp_analysis import EDPCalculator, EnergyProfiler, PerformanceProfiler

# Initialize components
calculator = EDPCalculator()
energy_profiler = EnergyProfiler()
performance_profiler = PerformanceProfiler()

# Calculate EDP metrics
results = calculator.calculate_edp_metrics(
    energy_data=your_energy_measurements,
    delay_data=your_performance_data,
    frequencies=[800, 900, 1000, 1100, 1200]
)

print(f"EDP optimal frequency: {results['edp_optimal']['frequency']} MHz")
print(f"Expected energy savings: {results['edp_optimal']['energy_improvement']:.1f}%")
```

### Enhanced Analysis with Feature Selection
```python
from edp_analysis import calculate_edp_with_features, analyze_feature_importance_for_edp

# Advanced EDP analysis with feature selection
enhanced_results = calculate_edp_with_features(
    df=profiling_data,
    energy_col='energy',
    delay_col='execution_time',
    use_feature_selection=True,
    gpu_type='V100'
)

# Analyze feature importance
feature_analysis = analyze_feature_importance_for_edp(
    df=profiling_data,
    target_metrics=['energy', 'execution_time'],
    gpu_type='V100'
)

print(f"Top features for EDP optimization: {feature_analysis['top_features']}")
```

### Comprehensive Visualization
```python
from edp_analysis.visualization import EDPPlotter, PowerPlotter, PerformancePlotter

# Create comprehensive dashboard
edp_plotter = EDPPlotter()
dashboard = edp_plotter.create_comprehensive_edp_dashboard(
    profiling_data=your_data,
    optimization_results=optimization_results,
    feature_importance=feature_importance,
    app_name="Your Application"
)

# Power analysis with FGCS validation
power_plotter = PowerPlotter()
validation_plot = power_plotter.plot_fgcs_power_validation(
    df=validation_data,
    app_name="Your Application"
)
```

### 🎯 **Time-Series Visualization CLI Tool**

The module now includes a production-ready command-line tool for immediate profiling data visualization:

```bash
# Navigate to your project directory
cd /path/to/ai-inference-energy

# Plot GPU utilization for LLAMA on V100 at multiple frequencies
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu V100 \
    --app LLAMA \
    --frequencies 510,960,1380 \
    --metric GPUTL \
    --run 2

# Plot power consumption with custom title and save to file
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu A100 \
    --app VIT \
    --frequencies 1200,1410 \
    --metric POWER \
    --title "Power Analysis - Vision Transformer on A100" \
    --save results/power_analysis.png \
    --no-show

# List available metrics for your data
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu V100 \
    --app LLAMA \
    --list-metrics
```

**Available Metrics**: POWER, GPUTL, MCUTL, TMPTR, DRAMA, FBTTL, TENSO, SMACT, FP32A, and 15+ more DCGMI metrics.

**Key Features**:
- Multi-frequency comparison with color-coded lines
- Normalized time axis (0-1) for fair comparison
- Professional styling with 300 DPI output for publications
- Automatic DCGMI CSV parsing and error handling
- Support for all GPU types and applications in the framework

See [`visualization/README.md`](visualization/README.md) for complete usage guide.

## 🔧 Core Features

### EDP Calculator (`edp_calculator.py`)

**Primary Functions**:
- `calculate_edp_metrics()` - Comprehensive EDP/ED²P analysis
- `find_optimal_frequency()` - Find frequency that minimizes EDP or ED²P
- `calculate_improvement_metrics()` - Compare against baseline configuration

**Input Data Requirements**:
- Power consumption measurements (Watts)
- Runtime/performance data (seconds or relative performance)
- Frequency configurations (MHz)
- Baseline configuration for comparison

## 🛡️ Robustness and Error Handling

### Recent Improvements (v1.0.0 - Production Ready)

**Production-Ready Error Handling**:
- ✅ **Division by Zero Protection**: Comprehensive validation prevents runtime errors from invalid baseline values
- ✅ **NaN Prevention**: Automatic detection and handling of mathematical edge cases
- ✅ **Graceful Degradation**: Safe fallback behavior when calculations cannot be performed
- ✅ **Enhanced Logging**: Detailed error messages and warnings for debugging

**Data Validation Enhancements**:
- ✅ Input data sanity checks with bounds validation
- ✅ Frequency range validation against hardware specifications
- ✅ Power consumption reasonableness checks with outlier detection
- ✅ Runtime measurement validation with statistical analysis

**Robustness Features**:
```python
# The module automatically handles all edge cases
results = calculate_edp_metrics(
    power_data=[150, 160, 170],
    runtime_data=[1.0, 0.9, 0.8],
    baseline_frequency=1000,
    target_frequencies=[900, 1000, 1100],
    baseline_runtime=0.0  # Invalid baseline - handled gracefully
)
# Logs: "Warning: Invalid baseline runtime (0.0) for EDP calculation"
# Returns: Valid results with appropriate fallback values

# Mathematical edge cases are handled automatically
results = framework.analyze_dvfs_optimization(data_with_edge_cases, "TestApp")
# ✅ No runtime warnings, comprehensive error handling
```

## 📈 Optimization Strategies

### EDP vs ED²P Trade-offs

**EDP Optimization**:
- Balances energy and performance equally
- Good for general-purpose energy efficiency
- Typically favors moderate frequency reductions

**ED²P Optimization**:
- Heavily penalizes longer execution times
- Better for latency-sensitive applications
- Usually recommends higher frequencies than EDP

### Use Cases

**Batch Processing**: Use EDP optimization for maximum energy efficiency
**Interactive Applications**: Use ED²P optimization to maintain responsiveness
**Mixed Workloads**: Compare both metrics and choose based on requirements

## 🔗 Integration with Power Modeling

The EDP analysis module integrates seamlessly with the power modeling framework:

```python
from power_modeling import FGCSPowerModelingFramework
from edp_analysis.edp_calculator import calculate_edp_metrics

# Power modeling for prediction
framework = FGCSPowerModelingFramework()
power_predictions = framework.predict_power_sweep(...)

# EDP optimization for frequency selection
edp_results = calculate_edp_metrics(
    power_data=power_predictions['power'],
    runtime_data=power_predictions['runtime'],
    baseline_frequency=1000
)
```

## 📝 Output Format

```python
{
    'edp_optimal': {
        'frequency': 1050,      # MHz
        'power': 175.2,         # Watts
        'runtime': 1.12,        # seconds
        'edp': 196.224,         # Joule·seconds
        'energy_improvement': 12.5,  # % vs baseline
        'time_improvement': -8.0     # % vs baseline (negative = slower)
    },
    'ed2p_optimal': {
        'frequency': 1150,
        'power': 185.4,
        'runtime': 1.05,
        'ed2p': 204.547,
        'energy_improvement': 8.2,
        'time_improvement': -5.0
    },
    'all_results': [
        # Detailed results for each frequency tested
    ]
}
```

## ⚠️ Important Notes

### Baseline Configuration
- Always use consistent measurement conditions for baseline
- Ensure baseline represents typical operating conditions
- Validate baseline measurements before optimization

### Measurement Quality
- Use multiple measurement runs for statistical confidence
- Account for thermal effects and GPU state
- Consider application-specific performance patterns

### Frequency Constraints
- Respect GPU hardware frequency limits
- Consider thermal and power constraints
- Validate that target frequencies are achievable

## 🤝 Contributing

To improve the EDP analysis module:
1. Add new optimization metrics beyond EDP/ED²P
2. Implement multi-objective optimization algorithms
3. Add uncertainty quantification for measurements
4. Extend support for other hardware architectures

## 📚 References

1. "Energy-Efficient GPU Frequency Selection for Deep Learning Inference" (FGCS 2023)
2. EDP optimization methodology for GPU computing
3. Energy-delay trade-off analysis in parallel computing

## 📄 Version History

- **v1.0.0 (Latest)**: Production-ready release with comprehensive error handling and robustness improvements
  - ✅ Division by zero protection in all EDP calculations
  - ✅ Enhanced data validation and input sanitization
  - ✅ Runtime warning elimination through mathematical edge case handling
  - ✅ Graceful fallback behavior for invalid baseline configurations
- **v0.9.x**: Core EDP/ED²P optimization algorithms
- **v0.8.x**: Initial implementation with basic EDP calculation functionality

**🚀 Status**: Production Ready - Validated and Tested
