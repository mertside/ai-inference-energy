# Visualization Tools

This directory contains all visualization tools for the AI Inference Energy project, providing comprehensive plotting and analysis capabilities for EDP/ED²P optimization results.

## 📁 Directory Structure

- **`visualize_edp_results.py`** - 🎨 **Real experimental data visualization** using DCGMI profiling data
- **`visualize_edp_summary.py`** - 📊 **Comprehensive summary analysis** with comparative charts
- **`README.md`** - This documentation file

## 🎯 Purpose

The visualization suite provides:

1. **Individual Scatter Plots**: Energy vs performance analysis for each GPU-workload combination
2. **Summary Comparisons**: Cross-GPU and cross-workload analysis charts
3. **Real Data Integration**: Uses actual DCGMI profiling measurements when available
4. **Publication Quality**: 300 DPI outputs suitable for research papers and presentations

## 🚀 Quick Start

### Generate All Visualizations

```bash
# From the tools/analysis/visualization directory
cd tools/analysis/visualization

# 1. Create individual scatter plots with real experimental data
python visualize_edp_results.py --input ../results/edp_optimization_results.json --output-dir edp-plots

# 2. Generate comprehensive summary visualizations
python visualize_edp_summary.py --input ../results/edp_optimization_results.json --output-dir edp-plots

# 3. View all generated visualization files
ls edp-plots/*.png
```

### Basic Usage

```bash
# Individual plots (uses real DCGMI data when available)
python visualize_edp_results.py

# Summary analysis charts
python visualize_edp_summary.py

# Custom input/output paths
python visualize_edp_results.py --input path/to/results.json --output-dir path/to/plots/
```

## 🎨 Real Data Visualization Framework

### Key Features
- **Real DCGMI Integration**: Uses actual power measurements from experimental data
- **Intelligent Fallback**: Automatically generates synthetic data when real data unavailable
- **Data Transparency**: Clear labeling of real vs synthetic data sources
- **Comprehensive Coverage**: Supports all GPU architectures (A100, H100, V100)

### Experimental Data Loading
1. **Directory Discovery**: Automatically finds result directories matching `results_{gpu}_{workload}_job_*` pattern
2. **DCGMI Data Parsing**: Loads power measurements from CSV profiles with robust column detection
3. **Timing Extraction**: Extracts execution times from `timing_summary.log` files
4. **Energy Calculation**: Computes total energy using `measured_power × measured_time`
5. **Data Validation**: Ensures minimum 3 data points for meaningful visualization

## 📊 Generated Outputs

### Individual Scatter Plots (12 files)
- `A100_llama_energy_performance_scatter.png` (61 real data points)
- `A100_stablediffusion_energy_performance_scatter.png` (61 real data points)
- `A100_vit_energy_performance_scatter.png` (61 real data points)
- `A100_whisper_energy_performance_scatter.png` (61 real data points)
- `H100_llama_energy_performance_scatter.png` (86 real data points)
- `H100_stablediffusion_energy_performance_scatter.png` (86 real data points)
- `H100_vit_energy_performance_scatter.png` (86 real data points)
- `H100_whisper_energy_performance_scatter.png` (86 real data points)
- `V100_llama_energy_performance_scatter.png` (59 real data points)
- `V100_stablediffusion_energy_performance_scatter.png` (59 real data points)
- `V100_vit_energy_performance_scatter.png` (59 real data points)
- `V100_whisper_energy_performance_scatter.png` (59 real data points)

### Summary Analysis Charts (4 files)
- `energy_savings_comparison.png` - EDP vs ED²P energy savings comparison
- `frequency_optimization_comparison.png` - Frequency reduction analysis across GPUs
- `performance_impact_analysis.png` - Performance trade-off visualization
- `comprehensive_summary.png` - 4-panel overview dashboard

### Visualization Features
- **Real Data Integration**: All individual plots use actual DCGMI profiling measurements
- **Publication Quality**: 300 DPI resolution suitable for research papers and presentations
- **Clear Annotations**: EDP and ED²P optimal points clearly marked with performance statistics
- **Data Source Transparency**: Plots clearly indicate "Real Experimental Data" vs "Synthetic Data"
- **Professional Styling**: Consistent color schemes, legends, and formatting across all visualizations

## 🔧 Technical Implementation

### Dependencies
- `matplotlib` - Plotting and visualization
- `numpy` - Numerical computations
- `pandas` - CSV data processing
- `pathlib` - File path handling
- `json` - Results file loading
- `re` - Regular expression pattern matching

### Data Quality Metrics
- **A100 Coverage**: 61/61 frequencies with real data (100% coverage)
- **H100 Coverage**: 86/86 frequencies with real data (100% coverage)
- **V100 Coverage**: 59/59 frequencies with real data (100% coverage)
- **Sampling Rate**: 50ms DCGMI power measurements for high precision
- **Total Data Points**: 824 real experimental measurements visualized

### Error Handling
- Graceful fallback to synthetic data if real data unavailable
- Robust CSV parsing with multiple column detection strategies
- Warning messages for missing or insufficient data
- Validation of data quality before plotting

## 🔗 Integration with Analysis Tools

The visualization tools are designed to work seamlessly with the EDP analysis suite:

```bash
# Complete workflow from analysis to visualization
cd tools/analysis

# 1. Run EDP optimization analysis
python edp_optimizer.py --results-dir ../../sample-collection-scripts

# 2. Generate CSV summaries
python edp_summary_tables.py --csv

# 3. Create visualizations
cd visualization
python visualize_edp_results.py --input ../results/edp_optimization_results.json --output-dir edp-plots
python visualize_edp_summary.py --input ../results/edp_optimization_results.json --output-dir edp-plots
```
