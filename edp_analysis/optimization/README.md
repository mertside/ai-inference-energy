# EDP Analysis - Optimization

This module provides tools for finding optimal GPU frequencies using Energy-Delay Product (EDP) and Energy-Delay² Product (ED²P) optimization methods.

## Overview

The optimization pipeline analyzes aggregated profiling data to identify optimal frequency configurations that minimize energy consumption while maintaining acceptable performance across different AI applications.

## Features

- **Multiple Optimization Methods**: EDP, ED²P, energy-only, and performance-only optimization
- **Comprehensive Analysis**: Per-application and cross-application optimization
- **Baseline Comparison**: Energy savings analysis vs maximum frequency baseline
- **Statistical Summaries**: Mean, std, min, max energy savings across configurations
- **Visualization**: EDP curves, energy savings charts, and optimization summaries
- **JSON Export**: Machine-readable results for downstream analysis

## Usage

### Basic Optimization

```bash
# Analyze all configurations with EDP and ED²P
python find_optimal_frequencies.py \
    --data ../data_aggregation/complete_aggregation.csv \
    --output optimal_frequencies.json

# Filter by specific GPU
python find_optimal_frequencies.py \
    --data ../data_aggregation/complete_aggregation.csv \
    --gpu V100 \
    --methods edp ed2p energy performance

# Analyze specific application
python find_optimal_frequencies.py \
    --data ../data_aggregation/complete_aggregation.csv \
    --app LLAMA \
    --methods edp
```

### Visualization

```bash
# Create all optimization plots
python plot_edp_optimization.py \
    --data ../data_aggregation/complete_aggregation.csv \
    --optimal optimal_frequencies.json \
    --output-dir optimization_plots

# Plot specific configuration
python plot_edp_optimization.py \
    --data ../data_aggregation/complete_aggregation.csv \
    --gpu V100 --app LLAMA
```

## Optimization Methods

### Energy-Delay Product (EDP)

Minimizes the product of energy consumption and execution time:
```
EDP = Energy × Execution_Time
```

**Best for**: Balanced energy-performance optimization

### Energy-Delay² Product (ED²P)

Emphasizes performance over energy by squaring the delay term:
```
ED²P = Energy × (Execution_Time)²
```

**Best for**: Performance-conscious energy optimization

### Energy-Only

Minimizes energy consumption regardless of performance impact:
```
Minimize: Energy
```

**Best for**: Maximum energy savings scenarios

### Performance-Only

Minimizes execution time regardless of energy impact:
```
Minimize: Execution_Time
```

**Best for**: Baseline performance comparison

## Key Results (Example)

Our analysis of 480 configurations across 2 GPUs and 4 applications revealed:

### Energy Savings Summary
- **EDP Optimization**: 91.6% ± 8.1% average energy savings
- **ED²P Optimization**: 91.6% ± 8.1% average energy savings
- **Range**: 74.0% - 98.9% energy savings across all configurations

### Best Configurations
- **Top Performer**: V100+WHISPER @ 600 MHz (98.9% energy savings)
- **Most Consistent**: VIT applications (95.7-96.1% savings on both GPUs)

### Optimal Frequency Distribution
- **Low Frequencies**: VIT applications (525 MHz optimal)
- **Mid Frequencies**: WHISPER applications (600-1170 MHz range)
- **High Frequencies**: STABLEDIFFUSION and LLAMA (705-1230 MHz range)

## Input Data Format

Requires aggregated data with the following columns:
- `gpu`: GPU type (V100, A100, H100)
- `application`: Application name
- `frequency`: GPU frequency in MHz
- `avg_power`: Average power consumption in Watts
- `execution_time`: Execution time in seconds
- `energy`: Energy consumption in Joules

## Output Format

### JSON Results Structure

```json
{
  "summary": {
    "total_gpus": 2,
    "total_applications": 4,
    "total_combinations": 8,
    "optimization_methods": ["edp", "ed2p"],
    "analysis_timestamp": "2025-07-31T12:58:34.993862",
    "statistics": {
      "average_energy_savings": {
        "edp": {"mean": 91.6, "std": 8.1, "min": 74.0, "max": 98.9}
      },
      "best_configurations": {
        "edp": {"configuration": "V100_WHISPER", "frequency": 600, "energy_savings": 98.9}
      }
    }
  },
  "configurations": {
    "V100_LLAMA": {
      "gpu": "V100",
      "application": "LLAMA",
      "total_frequencies": 59,
      "frequency_range": {"min": 510, "max": 1380},
      "optimizations": {
        "edp": {
          "frequency": 1155,
          "energy": 3752.7,
          "execution_time": 44.55,
          "baseline_comparison": {
            "energy_savings_percent": 74.0,
            "time_penalty_percent": -88.3
          }
        }
      }
    }
  }
}
```

## Visualization Outputs

### EDP Curve Analysis
- **Individual plots** for each GPU+application combination
- **Four-panel layout**: Energy, Execution Time, EDP, ED²P vs Frequency
- **Optimal point markers** showing EDP and ED²P minima
- **Professional styling** with clear legends and grid lines

### Summary Charts
- **Energy savings comparison** across all configurations
- **Optimal frequency distribution** showing method differences
- **Bar charts** with value labels for easy interpretation

## Implementation Details

### Baseline Comparison

Energy savings calculated relative to maximum frequency:
```python
energy_savings = (baseline_energy - optimal_energy) / baseline_energy × 100
```

### Statistical Analysis

Cross-configuration statistics provide:
- Mean and standard deviation of energy savings
- Best performing configurations per method
- Frequency distribution analysis

### Validation Checks

- Optimal frequency ≤ maximum available frequency
- Energy savings > 0 (sanity check)
- Reasonable execution time ranges
- Consistent results across methods

## Command Line Options

### find_optimal_frequencies.py

- `--data`: Path to aggregated profiling data CSV (required)
- `--output`: Output JSON file (default: optimal_frequencies.json)
- `--gpu`: Filter by GPU type (V100, A100, H100)
- `--app`: Filter by application (LLAMA, VIT, STABLEDIFFUSION, WHISPER)
- `--methods`: Optimization methods (default: edp ed2p)

### plot_edp_optimization.py

- `--data`: Path to aggregated profiling data CSV (required)
- `--optimal`: Path to optimal frequencies JSON file
- `--gpu`: Plot specific GPU (plots all if not specified)
- `--app`: Plot specific application (plots all if not specified)
- `--output-dir`: Output directory for plots (default: plots)

## Performance Considerations

- **Memory efficient**: Processes data in chunks for large datasets
- **Vectorized operations**: Uses pandas/numpy for fast computation
- **Parallel potential**: Individual GPU+app analyses are independent

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- json (built-in)
- pathlib (built-in)

## Use Cases

### Research Applications
- Energy efficiency studies
- GPU frequency optimization research
- AI workload characterization
- Power modeling validation

### Production Applications
- Data center energy optimization
- Deployment frequency selection
- SLA-aware energy management
- Thermal management optimization

## Troubleshooting

### Common Issues

1. **"No data found for GPU+app"**
   - Verify aggregated data contains the specified configuration
   - Check GPU and application name spelling/case

2. **"Missing required columns"**
   - Ensure aggregated data has all required columns
   - Re-run data aggregation if necessary

3. **"Invalid optimization method"**
   - Use only supported methods: edp, ed2p, energy, performance

### Debug Tips

- Check aggregated data structure: `df.info()` and `df.head()`
- Validate frequency ranges per configuration
- Review statistical summaries for anomalies
