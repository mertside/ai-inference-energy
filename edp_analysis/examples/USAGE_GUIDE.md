# GPU Frequency Optimization Tools - Usage Guide

## Overview

This guide explains how to use the refactored EDP analysis tools for GPU frequency optimization. The tools focus on finding optimal frequencies that provide **minimal performance degradation with disproportionately high energy savings**.

## Quick Start

### 1. Simple Analysis (Recommended)
Use `simple_unified_analysis.py` for straightforward analysis without complex dependencies:

```bash
# Basic analysis of all configurations
python simple_unified_analysis.py --output ./results --create-plots --generate-deployment

# Filter to specific GPU and application
python simple_unified_analysis.py --gpu A100 --app STABLEDIFFUSION --output ./a100_sd_results

# Stricter performance requirements (max 5% degradation)
python simple_unified_analysis.py --max-degradation 5 --output ./strict_results

# Generate comprehensive output
python simple_unified_analysis.py --max-degradation 10 \
  --output ./production_results \
  --create-plots \
  --generate-deployment
```

### 2. Enhanced Analysis (Advanced)
Use `enhanced_frequency_analysis.py` for detailed analysis with advanced visualizations:

```bash
# Run enhanced analysis with corrected efficiency calculations
python enhanced_frequency_analysis.py
```

## Tool Options

### `simple_unified_analysis.py`

#### Required Data
- Aggregated profiling data: `../data_aggregation/complete_aggregation_run2.csv`
- Data format: CSV with columns `frequency`, `execution_time`, `avg_power`, `gpu`, `application`, etc.

#### Command Line Options

**Analysis Parameters:**
- `--max-degradation FLOAT`: Maximum acceptable performance degradation in % (default: 15.0)
- `--min-efficiency FLOAT`: Minimum energy efficiency ratio (default: 2.0)

**Filtering Options:**
- `--gpu {A100,V100}`: Filter to specific GPU type
- `--app {LLAMA,STABLEDIFFUSION,VIT,WHISPER}`: Filter to specific application

**Output Options:**
- `--output DIR`: Output directory for all results (default: ./simple_analysis_results)
- `--create-plots`: Generate visualization plots
- `--generate-deployment`: Create deployment scripts

**Example Workflows:**

```bash
# Production analysis for A100 GPUs
python simple_unified_analysis.py \
  --gpu A100 \
  --max-degradation 10 \
  --output ./a100_production \
  --create-plots \
  --generate-deployment

# Research analysis with strict criteria
python simple_unified_analysis.py \
  --max-degradation 5 \
  --min-efficiency 5.0 \
  --output ./research_strict \
  --create-plots

# Application-specific optimization
python simple_unified_analysis.py \
  --app STABLEDIFFUSION \
  --max-degradation 15 \
  --output ./stable_diffusion_opt \
  --generate-deployment
```

## Understanding Output

### Console Output
The tool provides detailed progress information:
- Data loading and filtering status
- Number of configurations analyzed
- Optimization results summary
- Top recommendations with detailed metrics

### Generated Files

#### Data Files
- `efficiency_analysis.csv`: Complete efficiency metrics for all frequency points
- `optimal_configurations.csv`: Final optimal configurations ready for deployment
- `analysis_summary.json`: Machine-readable summary for integration

#### Deployment Script
- `deploy_optimized_frequencies.sh`: Production-ready deployment script

**Usage:**
```bash
# Deploy optimal frequency for A100+LLAMA
./deploy_optimized_frequencies.sh "A100+LLAMA" deploy

# Check current GPU frequency status
./deploy_optimized_frequencies.sh "A100+LLAMA" status

# Reset to default frequencies
./deploy_optimized_frequencies.sh "A100+LLAMA" reset
```

#### Visualization Plots (with `--create-plots`)
- `pareto_analysis.png`: Performance vs energy trade-off analysis
- `optimal_frequencies.png`: Bar chart of optimal frequencies with performance impact labels
- `efficiency_ratios.png`: Energy efficiency ratios for optimal configurations

### Key Metrics Explained

**Performance Penalty (%)**:
- Positive: Performance degradation (slower execution)
- Negative: Performance improvement (faster execution)  
- Zero: No performance impact

**Energy Savings (%)**:
- Positive: Reduction in energy consumption
- Based on: `(baseline_energy - current_energy) / baseline_energy * 100`

**Efficiency Ratio**:
- Formula: `energy_savings / abs(performance_penalty)`
- Higher is better (more energy savings per unit performance impact)
- Special handling for performance improvements (very high ratios)

**Category Classification**:
- **Minimal Impact**: ≤2% performance penalty
- **Low Impact**: 2-5% performance penalty
- **Moderate Impact**: 5-10% performance penalty
- **High Impact**: 10-15% performance penalty

## Integration with Existing Workflow

### 1. Data Preparation
Ensure your aggregated data follows the expected CSV format:
```csv
frequency,execution_time,avg_power,gpu,application,...
1155,42.0,95.39,A100,LLAMA,...
915,52.8,78.15,A100,LLAMA,...
```

### 2. Production Deployment
1. Run analysis to identify optimal configurations
2. Use generated deployment script for implementation
3. Monitor performance and energy metrics
4. Validate results against baselines

### 3. Continuous Optimization
- Re-run analysis with new profiling data
- Adjust criteria based on operational requirements
- Update deployment scripts with new optimal configurations

## Troubleshooting

### Common Issues

**"Column not found" errors:**
- Ensure CSV has required columns: `frequency`, `execution_time`, `avg_power`, `gpu`, `application`
- Check column names match exactly (case-sensitive)

**No optimal configurations found:**
- Increase `--max-degradation` threshold
- Decrease `--min-efficiency` requirement
- Check if data contains valid frequency sweep results

**Plot generation fails:**
- Ensure matplotlib is available
- Use `--create-plots` flag
- Check output directory permissions

### Data Requirements

**Minimum Data:**
- At least 2 frequency points per GPU+Application combination
- Complete power and execution time measurements
- Consistent baseline (highest frequency) measurements

**Recommended Data:**
- Comprehensive frequency sweeps (10+ points per configuration)
- Multiple runs for statistical confidence
- Consistent measurement methodology across configurations

## Advanced Usage

### Custom Analysis Scripts
The modular design allows easy customization:

```python
# Import core functions
from simple_unified_analysis import calculate_efficiency_metrics, find_optimal_frequencies

# Load your data
df = pd.read_csv('your_data.csv')

# Calculate metrics
efficiency_df = calculate_efficiency_metrics(df)

# Find optimal configurations with custom criteria
optimal_df = find_optimal_frequencies(
    efficiency_df, 
    max_degradation=8.0,  # Custom threshold
    min_efficiency=3.0    # Custom efficiency requirement
)
```

### Integration with Job Schedulers
```bash
# Example SLURM job script
#!/bin/bash
#SBATCH --job-name=gpu_freq_opt
#SBATCH --time=01:00:00

cd /path/to/analysis/tools
python simple_unified_analysis.py \
  --output ./job_$SLURM_JOB_ID \
  --create-plots \
  --generate-deployment

# Auto-deploy if analysis successful
if [ $? -eq 0 ]; then
    ./job_$SLURM_JOB_ID/deploy_optimized_frequencies.sh "A100+LLAMA" deploy
fi
```

## Support and Development

### Adding New Applications
1. Ensure profiling data includes new application in `application` column
2. Update tool's application choices if using filtering
3. Validate optimization results manually before production deployment

### Extending Analysis
- Modify efficiency calculation methods in `calculate_efficiency_metrics()`
- Add new optimization criteria in `find_optimal_frequencies()`
- Enhance visualization with additional plot types

### Contributing
- Follow existing code style and documentation standards
- Add comprehensive error handling for new features
- Include example usage in documentation updates

---

*Tools developed as part of the AI Inference Energy Profiling Framework*  
*For support: Check GitHub issues or contact development team*
