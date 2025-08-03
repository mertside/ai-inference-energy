# GPU Frequency Optimization Framework - Simplified

A clean, organized framework for optimizing GPU frequencies to achieve **minimal performance degradation with maximum energy savings**.

## Overview

This framework provides a simplified, well-organized approach to GPU frequency optimization with four main components:

1. **📦 Core Components** (`core/`) - Fundamental analysis functions and utilities
2. **📊 Visualization** (`visualization/`) - Plotting and visualization tools  
3. **🔧 Data Aggregation** (`data_aggregation/`) - Data processing and aggregation
4. **⚡ Frequency Optimization** (`frequency_optimization/`) - Optimization algorithms and deployment tools

## Quick Start

### Basic Analysis
```bash
# Analyze all configurations
python optimize_gpu_frequency.py --data your_data.csv --output results/

# Filter to specific GPU and application
python optimize_gpu_frequency.py --gpu A100 --app STABLEDIFFUSION --output a100_sd_results/

# Generate plots and deployment package
python optimize_gpu_frequency.py --plots --deploy --output production_results/
```

### Expected Results
- **Energy Savings**: 13-29% reduction in power consumption
- **Performance Impact**: Minimal (often performance improvements)
- **Efficiency Ratios**: 15:1 to 2933:1 (energy savings per performance unit)

## Framework Structure

```
edp_analysis_simplified/
├── 📦 core/                           # Core analysis components
│   ├── analysis.py                    # Main analysis functions
│   ├── utils.py                       # Utility functions and helpers
│   └── __init__.py                    # Core package exports
├── 📊 visualization/                  # Visualization components
│   ├── plots.py                       # Plotting functions
│   └── __init__.py                    # Visualization exports
├── 🔧 data_aggregation/               # Data processing
│   ├── aggregator.py                  # Data aggregation functions
│   └── __init__.py                    # Data aggregation exports
├── ⚡ frequency_optimization/         # Optimization and deployment
│   ├── algorithms.py                  # Advanced optimization algorithms
│   ├── deployment.py                  # Deployment script generation
│   └── __init__.py                    # Optimization exports
├── optimize_gpu_frequency.py          # 🚀 Main analysis tool
├── README.md                          # This documentation
└── __init__.py                        # Framework package exports
```

## Key Features

### 🎯 Zero-Degradation Optimization
- Achieve significant energy savings with minimal or no performance impact
- Most optimal configurations actually improve performance while saving energy

### 📊 Comprehensive Analysis
- Efficiency ratio calculations (energy savings per performance impact)
- Performance categorization (Minimal/Low/Moderate/High Impact)
- Pareto frontier analysis for trade-off visualization

### 🚀 Production Ready
- Automated deployment script generation
- Safety features with status checking and reset capabilities
- Validation scripts for system compatibility

### 📈 Rich Visualizations
- Performance vs energy trade-off plots
- Optimal frequency recommendations
- Efficiency ratio comparisons
- Comprehensive optimization dashboards

## Usage Examples

### 1. Basic Analysis
```bash
python optimize_gpu_frequency.py \
  --data ../data_aggregation/complete_aggregation_run2.csv \
  --output ./basic_analysis
```

### 2. Targeted Analysis with Constraints
```bash
python optimize_gpu_frequency.py \
  --data data.csv \
  --gpu A100 \
  --app STABLEDIFFUSION \
  --max-degradation 10 \
  --min-efficiency 5.0 \
  --output ./targeted_analysis \
  --plots \
  --deploy
```

### 3. Production Deployment
```bash
# Run analysis
python optimize_gpu_frequency.py --deploy --output ./production

# Validate system
./production/validate_deployment.sh

# Deploy optimal configuration
./production/deploy_frequencies.sh "A100+STABLEDIFFUSION" deploy

# Monitor status
./production/deploy_frequencies.sh "A100+STABLEDIFFUSION" status
```

## Command Line Options

### Input/Output
- `--data FILE`: Path to aggregated profiling data CSV
- `--output DIR`: Output directory for results

### Analysis Parameters
- `--max-degradation PERCENT`: Maximum acceptable performance degradation (default: 15%)
- `--min-efficiency RATIO`: Minimum energy efficiency ratio (default: 2.0)

### Filtering
- `--gpu {A100,V100}`: Filter to specific GPU type
- `--app {LLAMA,STABLEDIFFUSION,VIT,WHISPER}`: Filter to specific application

### Output Options
- `--plots`: Generate visualization plots
- `--deploy`: Create deployment scripts and package
- `--verbose`: Enable detailed output

## Output Files

### Data Files
- `efficiency_analysis.csv`: Complete efficiency metrics for all frequency points
- `optimal_configurations.csv`: Final optimal configurations
- `analysis_results.json`: Machine-readable summary for integration

### Deployment Package
- `deploy_frequencies.sh`: Production deployment script
- `validate_deployment.sh`: System validation script
- `optimization_report.md`: Detailed optimization report

### Visualizations
- `pareto_analysis.png`: Performance vs energy trade-off plot
- `optimal_frequencies.png`: Optimal frequency recommendations
- `efficiency_ratios.png`: Efficiency ratio comparison
- `optimization_dashboard.png`: Comprehensive dashboard

## Data Format Requirements

Your CSV data should include these columns:
- `frequency`: GPU frequency in MHz
- `execution_time`: Execution time in seconds  
- `avg_power`: Average power consumption in watts
- `gpu`: GPU type (A100, V100)
- `application`: Application name (LLAMA, STABLEDIFFUSION, etc.)

Optional columns:
- `avg_mmclk`: Memory clock frequency
- `std_power`, `min_power`, `max_power`: Power statistics
- `avg_temp`, `max_temp`: Temperature metrics

## Framework Components

### Core Analysis (`core/`)
```python
from core import calculate_efficiency_metrics, find_optimal_configurations

# Calculate efficiency metrics
efficiency_df = calculate_efficiency_metrics(raw_data)

# Find optimal configurations
optimal_df = find_optimal_configurations(efficiency_df, max_degradation=10)
```

### Visualization (`visualization/`)
```python
from visualization import generate_all_plots

# Generate all standard plots
plot_paths = generate_all_plots(efficiency_df, optimal_df, output_dir)
```

### Frequency Optimization (`frequency_optimization/`)
```python
from frequency_optimization import create_deployment_package

# Create complete deployment package
files = create_deployment_package(efficiency_df, optimal_df, params, output_dir)
```

## Advanced Usage

### Custom Optimization Algorithms
```python
from frequency_optimization import pareto_frontier_optimization, multi_objective_optimization

# Use Pareto frontier optimization
pareto_optimal = pareto_frontier_optimization(efficiency_df)

# Use multi-objective optimization with custom weights
weights = {'energy_savings': 0.7, 'performance_penalty': -0.3}
weighted_optimal = multi_objective_optimization(efficiency_df, weights)
```

### Data Aggregation
```python
from data_aggregation import create_aggregated_dataset

# Aggregate raw profiling data
aggregated_file = create_aggregated_dataset(
    raw_data_dir='./profiling_runs/',
    output_file='./aggregated_data.csv'
)
```

## Best Practices

### 1. Data Quality
- Ensure comprehensive frequency sweeps (10+ points per configuration)
- Include multiple measurement runs for statistical confidence
- Validate data format using built-in validation functions

### 2. Analysis Parameters
- Start with conservative degradation limits (5-10%)
- Adjust efficiency thresholds based on operational requirements
- Use filtering to focus on specific use cases

### 3. Production Deployment
- Always test in non-production environment first
- Use validation scripts to verify system compatibility
- Monitor system stability after deployment
- Keep reset options readily available

## Troubleshooting

### Common Issues

**No optimal configurations found:**
- Increase `--max-degradation` threshold
- Decrease `--min-efficiency` requirement
- Check data quality and frequency range

**Missing required columns:**
- Verify CSV has required columns: frequency, execution_time, avg_power, gpu, application
- Use data aggregation tools to standardize column names

**Plot generation fails:**
- Ensure matplotlib is available
- Check output directory permissions
- Use `--verbose` for detailed error messages

## Support and Development

### Contributing
- Follow existing code structure and documentation standards
- Add comprehensive error handling for new features
- Include usage examples in documentation

### Extending the Framework
- Add new optimization algorithms in `frequency_optimization/algorithms.py`
- Create custom visualization functions in `visualization/plots.py`
- Enhance data processing in `data_aggregation/aggregator.py`

---

*Framework developed for sustainable AI computing research*  
*Focus: Minimal performance degradation with maximum energy savings*
