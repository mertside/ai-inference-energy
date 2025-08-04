# Visualization Framework

**Basic visualization tools for GPU profiling data analysis.**

> **Note**: This documentation references advanced visualization features that are planned for future development. Current capabilities include basic plotting tools available in this directory.

This module provides visualization capabilities for the AI Inference Energy Profiling Framework, with basic time-series analysis and plotting tools.

## 🎯 Quick Start - Main Plotting Tool

### **plot_metric_vs_time.py** - Primary CLI Tool

The main visualization script for plotting any profiling metric vs normalized time with multi-frequency comparison.

#### **Basic Usage**
```bash
# Navigate to the AI inference energy directory
cd /path/to/ai-inference-energy

# Plot GPU utilization for LLAMA on V100 at multiple frequencies
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu V100 \
    --app LLAMA \
    --frequencies 510,960,1380 \
    --metric GPUTL \
    --run 2

# Plot power consumption for Vision Transformer on A100
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu A100 \
    --app VIT \
    --frequencies 1200,1410 \
    --metric POWER

# Save plot without displaying
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu V100 \
    --app STABLEDIFFUSION \
    --frequencies 800,1200 \
    --metric DRAMA \
    --save plots/sd_dram_activity.png \
    --no-show
```

#### **Available Parameters**

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--gpu` | GPU type | V100 | V100, A100, H100 |
| `--app` | Application name | LLAMA | LLAMA, VIT, STABLEDIFFUSION, WHISPER, LSTM |
| `--frequencies` | Comma-separated frequencies (MHz) | 510,960,1380 | 800,1200,1410 |
| `--metric` | Profiling metric to plot | GPUTL | POWER, GPUTL, DRAMA, TMPTR, FBTTL |
| `--run` | Run number | 1 | 1, 2, 3, ... |
| `--data-dir` | Data directory | sample-collection-scripts | path/to/results |
| `--save` | Save path | None (show plot) | plots/my_plot.png |
| `--title` | Custom plot title | Auto-generated | "Custom Plot Title" |
| `--list-metrics` | List available metrics | False | Flag only |
| `--no-show` | Don't display plot | False | Flag only |

#### **Available Metrics**

**Core GPU Metrics:**
- `POWER` - GPU power consumption (W)
- `GPUTL` - GPU utilization (%)
- `MCUTL` - Memory controller utilization (%)
- `TMPTR` - GPU temperature (°C)
- `DRAMA` - DRAM activity (%)

**Memory Metrics:**
- `FBTTL` - Frame buffer total memory
- `FBFRE` - Frame buffer free memory
- `FBUSD` - Frame buffer used memory

**Clock Frequencies:**
- `SMCLK` - SM clock frequency
- `MMCLK` - Memory clock frequency
- `MACLK` - Memory application clock

**Activity Metrics:**
- `GRACT` - Graphics pipe activity
- `SMACT` - SM activity
- `SMOCC` - SM occupancy
- `TENSO` - Tensor pipe activity
- `FP64A`, `FP32A`, `FP16A` - Floating-point activity

#### **Examples by Use Case**

**1. Power Analysis**
```bash
# Compare power consumption across frequencies
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu V100 --app LLAMA --metric POWER --frequencies 510,960,1380
```

**2. Performance Analysis**  
```bash
# Analyze GPU utilization patterns
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu A100 --app VIT --metric GPUTL --frequencies 1200,1410
```

**3. Memory Analysis**
```bash
# Study DRAM activity for memory-intensive workloads
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu V100 --app STABLEDIFFUSION --metric DRAMA --frequencies 800,1200,1600
```

**4. Thermal Analysis**
```bash
# Monitor temperature across different frequencies
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu H100 --app WHISPER --metric TMPTR --frequencies 1000,1200,1410
```

**5. Research Workflow**
```bash
# List available metrics for a configuration
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu V100 --app LLAMA --list-metrics

# Generate publication-ready plots
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu V100 --app LLAMA --metric POWER \
    --frequencies 510,960,1380 \
    --title "Power Consumption vs Time - LLAMA Inference" \
    --save paper_figures/llama_power_analysis.png \
    --no-show
```

## 📊 Advanced Visualization Framework

### **Module Structure**

```
edp_analysis/visualization/
├── README.md                    # This file - comprehensive usage guide
├── plot_metric_vs_time.py      # 🎯 Main CLI plotting tool
├── time_series_demo.py         # Educational demo script
├── edp_plots.py                # EDP analysis visualization
├── power_plots.py              # Power analysis visualization  
├── performance_plots.py        # Performance analysis visualization
├── data_preprocessor.py        # Data loading and preprocessing
└── __init__.py                 # Module initialization
```

### **Programmatic Usage**

For advanced users who want to integrate plotting into their own scripts:

```python
# from edp_analysis.visualization  # Future visualization module import EDPPlotter, PowerPlotter, PerformancePlotter
# from edp_analysis.visualization  # Future visualization module.data_preprocessor import ProfilingDataPreprocessor

# Load profiling data
loader = ProfilingDataPreprocessor()
df = loader.load_result_directory("path/to/results", app_name="LLAMA")

# Create various plots
power_plotter = PowerPlotter()
perf_plotter = PerformancePlotter()

# Plot power vs time with frequency overlays
power_fig = power_plotter.plot_power_vs_time(
    df, 
    power_col="POWER", 
    frequency_filter=[510, 960, 1380]
)

# Create multi-metric dashboard
metrics = ["POWER", "GPUTL", "TMPTR", "DRAMA"]
dashboard_fig = perf_plotter.plot_multi_metric_dashboard(
    df, 
    metrics=metrics,
    frequency_filter=[510, 960, 1380]
)
```

## 🔧 Troubleshooting

### **Common Issues**

**1. "No data was successfully loaded"**
- Check that your data directory contains the correct result folders
- Verify GPU and application names match folder structure
- Ensure run numbers exist in the data

**2. "Metric 'X' not found in data"**
- Use `--list-metrics` to see available metrics for your data
- Check DCGMI profiling configuration captured the desired metrics

**3. "ModuleNotFoundError"**
- Run from the ai-inference-energy root directory
- Ensure matplotlib is installed: `pip install matplotlib`

**4. Empty plots or missing frequencies**
- Verify frequency values exist in your data
- Check that CSV files have the expected naming convention

### **Data Directory Structure**

The script expects this structure:
```
sample-collection-scripts/
├── results_V100_LLAMA/
│   ├── run_01_freq_510/
│   │   └── dcgmi_profile.csv
│   ├── run_01_freq_960/
│   │   └── dcgmi_profile.csv
│   └── run_01_freq_1380/
│       └── dcgmi_profile.csv
└── results_A100_VIT/
    └── ...
```

## 🎨 Output Examples

The plots generated include:
- **Multi-frequency comparison** with color-coded lines
- **Normalized time axis** (0-1) for fair comparison
- **Professional styling** with grids and legends
- **High-resolution output** (300 DPI) for publications
- **Automatic scaling** and metric-appropriate units


**Test Command Used**:
```bash
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu V100 --app LLAMA --metric POWER \
    --frequencies 510,960,1380 --no-show --save test_power_plot.png
# ✅ Successfully generated 113KB plot file
```

## �🔗 Integration with Main Framework

This visualization tool is designed to work seamlessly with:
- **DCGMI profiling data** from main framework experiments
- **Job submission scripts** in `sample-collection-scripts/`
- **EDP analysis** results from `edp_analysis/` module
- **Profiling** outputs from DCGMI and nvidia-smi profiling tools

## 🔧 Troubleshooting

### **Import Errors**
```bash
# If you see import errors, ensure dependencies are available
python -c "import matplotlib; import pandas; import numpy; print('✅ Core dependencies available')"

# For conda environments
conda install matplotlib pandas numpy

# For pip environments  
pip install matplotlib pandas numpy
```

### **Missing Data Files**
```bash
# List available metrics to verify data structure
python edp_analysis/visualization/plot_metric_vs_time.py --list-metrics

# Check your data directory structure
ls -la sample-collection-scripts/results_*/*/dcgmi_profile.csv
```

### **Plot Display Issues**
```bash
# If plots don't display, save them instead
python edp_analysis/visualization/plot_metric_vs_time.py \
    --gpu V100 --metric POWER --no-show --save my_plot.png

# Check if plot was created
ls -la *.png
```

### **Module Syntax Errors**
All visualization modules have been recently validated (July 2025) and should import without syntax errors. If you encounter issues:

```bash
# Test individual module syntax
python -c "
import ast
with open('edp_analysis/visualization/power_plots.py', 'r') as f:
    ast.parse(f.read())
print('✅ power_plots.py syntax is valid')
"
```

For complete framework usage, see:
- [`../../README.md`](../../README.md) - Main framework documentation
- [`../../documentation/USAGE_EXAMPLES.md`](../../documentation/USAGE_EXAMPLES.md) - CLI examples
- [`../README.md`](../README.md) - EDP analysis module documentation

---

## 📚 **Complete Script Usage Guide**

### **time_series_demo.py** - Interactive Demo & Tutorial

**Purpose**: Educational demonstration script that showcases advanced time-series visualization capabilities with synthetic or real data.

#### **Usage**
```bash
# Run with real data from your experiments
python edp_analysis/visualization/time_series_demo.py \
    --data-dir sample-collection-scripts/ \
    --save-plots \
    --output-dir demo_plots/

# Run with synthetic data for testing/learning
python edp_analysis/visualization/time_series_demo.py \
    --synthetic \
    --save-plots

# Quick demo without saving plots
python edp_analysis/visualization/time_series_demo.py --synthetic
```

#### **Parameters**
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--data-dir` | Directory with result folders | sample-collection-scripts/ | path/to/results/ |
| `--synthetic` | Use synthetic demo data | False | Flag only |
| `--save-plots` | Save all generated plots | False | Flag only |
| `--output-dir` | Output directory for plots | plots/ | demo_visualizations/ |

#### **What You'll Get**
- **Single Metric Visualizations**: Power, utilization, temperature plots vs normalized time
- **Multi-Metric Dashboards**: 6-panel comparative analysis across frequencies
- **Temporal Pattern Analysis**: Statistical analysis of metric stability and trends
- **Advanced Visualizations**: Correlation heatmaps and frequency analysis plots

#### **Example Outputs**
```bash
# Generated files in output directory:
single_metric_power_draw_freq_900.png
multi_freq_gpu_utilization.png
multi_metric_dashboard.png
correlation_heatmap.png
frequency_analysis.png
```

### **data_preprocessor.py** - Data Loading & Processing Utilities

**Purpose**: Standalone data loading and preprocessing utilities for custom analysis scripts.

#### **Programmatic Usage**
```python
# from edp_analysis.visualization  # Future visualization module.data_preprocessor import ProfilingDataPreprocessor

# Load profiling data from result directory
loader = ProfilingDataPreprocessor()
df = loader.load_result_directory(
    "sample-collection-scripts/results_V100_LLAMA/",
    pattern="*_profile.csv",
    app_name="LLAMA"
)

# Prepare for time-series analysis
df = loader.prepare_for_time_series(df)

# Create synthetic data for testing
synthetic_df = loader.create_synthetic_profiling_data(
    frequencies=[900, 1200, 1410],
    duration_seconds=60,
    app_name="Test Application"
)
```

#### **Key Functions**
- `load_result_directory()` - Load all CSV files from experiment results
- `prepare_for_time_series()` - Normalize timestamps and prepare data
- `create_synthetic_profiling_data()` - Generate realistic test data
- `extract_frequency_from_path()` - Parse frequency from folder names

### **power_plots.py** - Advanced Power Analysis Classes

**Purpose**: Comprehensive power analysis visualization classes for custom plotting workflows.

#### **Programmatic Usage**
```python
# from edp_analysis.visualization  # Future visualization module.power_plots import PowerPlotter, plot_power_histogram

# Initialize plotter
power_plotter = PowerPlotter(figsize=(12, 8))

# Plot power consumption over time
fig1 = power_plotter.plot_power_vs_time(
    df,
    power_col="POWER",
    frequency_filter=[510, 960, 1380],
    app_name="LLAMA",
    save_path="power_time_series.png"
)

# Power efficiency analysis
fig2 = power_plotter.plot_power_efficiency(
    df,
    frequency_col="frequency",
    power_col="POWER",
    performance_col="throughput",
    app_name="LLAMA"
)

# Multi-application comparison
app_data = {
    "LLAMA": llama_df,
    "VIT": vit_df,
    "Whisper": whisper_df
}
fig3 = power_plotter.plot_multi_application_power(
    app_data,
    save_path="multi_app_power_comparison.png"
)

# Power model validation
fig4 = power_plotter.plot_power_model_validation(
    actual_df=measured_data,
    predicted_df=model_predictions,
    save_path="model_validation.png"
)

# Standalone power histogram
fig5 = plot_power_histogram(
    df,
    power_col="POWER",
    bins=30,
    save_path="power_distribution.png"
)
```

#### **Available Plot Types**
- **Power vs Time**: Time-series with multi-frequency overlays
- **Power vs Frequency**: Frequency sweep analysis with annotations
- **Power Efficiency**: Performance-per-watt optimization analysis
- **Multi-Application**: Comparative power consumption across workloads
- **Model Validation**: Predicted vs actual power with error metrics
- **Power Breakdown**: Component-wise power analysis
- **Power Histogram**: Distribution analysis with statistics

### **performance_plots.py** - Performance & Metrics Visualization

**Purpose**: Comprehensive performance analysis and multi-metric visualization classes.

#### **Programmatic Usage**
```python
# from edp_analysis.visualization  # Future visualization module.performance_plots import PerformancePlotter

# Initialize plotter
perf_plotter = PerformancePlotter(figsize=(14, 10))

# Single metric vs normalized time
fig1 = perf_plotter.plot_metric_vs_normalized_time(
    df,
    metric_col="GPUTL",
    frequency_filter=[900, 1200, 1410],
    app_name="Vision Transformer",
    save_path="gpu_utilization_analysis.png"
)

# Multi-metric dashboard (2x3 grid)
metrics = ["POWER", "GPUTL", "TMPTR", "DRAMA", "FBTTL", "SMACT"]
fig2 = perf_plotter.plot_multi_metric_dashboard(
    df,
    metrics=metrics,
    frequency_filter=[900, 1200],
    app_name="LLAMA",
    cols=3,
    save_path="performance_dashboard.png"
)

# Frequency sweep analysis
fig3 = perf_plotter.plot_frequency_sweep_analysis(
    df,
    primary_metric="POWER",
    secondary_metrics=["GPUTL", "TMPTR"],
    save_path="frequency_sweep.png"
)

# Temporal pattern analysis
patterns = perf_plotter.analyze_temporal_patterns(
    df,
    metric_col="POWER",
    window_size="1s"
)
print(f"Power stability across frequencies: {patterns}")
```

### **edp_plots.py** - EDP Optimization Visualization

**Purpose**: Energy-Delay Product (EDP) analysis and optimization visualization for research workflows.

#### **Programmatic Usage**
```python
# from edp_analysis.visualization  # Future visualization module.edp_plots import EDPPlotter

# Initialize EDP plotter
edp_plotter = EDPPlotter(figsize=(12, 8))

# EDP vs frequency analysis
fig1 = edp_plotter.plot_edp_vs_frequency(
    df,
    frequency_col="frequency",
    energy_col="energy",
    delay_col="execution_time",
    app_name="LLAMA",
    save_path="edp_frequency_analysis.png"
)

# Pareto frontier optimization
fig2 = edp_plotter.plot_pareto_frontier(
    df,
    energy_col="energy",
    delay_col="execution_time",
    frequency_col="frequency",
    save_path="pareto_optimization.png"
)

# 3D EDP surface plot
fig3 = edp_plotter.plot_3d_edp_surface(
    df,
    frequency_col="frequency",
    voltage_col="voltage",
    edp_col="edp",
    save_path="3d_edp_surface.png"
)

# Multi-application EDP comparison
app_data = {
    "LLAMA": llama_df,
    "VIT": vit_df,
    "StableDiffusion": sd_df
}
fig4 = edp_plotter.plot_multi_app_edp_comparison(
    app_data,
    save_path="multi_app_edp.png"
)

# Optimization summary with recommendations
fig5 = edp_plotter.create_optimization_summary_plot(
    df,
    optimization_target="edp",
    save_path="optimization_summary.png"
)
```

### **Custom Analysis Workflows**

#### **Research Pipeline Example**
```python
# from edp_analysis.visualization  # Future visualization module import *

# 1. Load and preprocess data
loader = ProfilingDataPreprocessor()
df = loader.load_result_directory("results_V100_LLAMA/")
df = loader.prepare_for_time_series(df)

# 2. Create comprehensive analysis
power_plotter = PowerPlotter()
perf_plotter = PerformancePlotter()
edp_plotter = EDPPlotter()

# 3. Generate publication-ready figures
figures = {
    "power_analysis": power_plotter.plot_power_vs_time(df, save_path="fig1_power.png"),
    "performance_dashboard": perf_plotter.plot_multi_metric_dashboard(df, save_path="fig2_performance.png"),
    "edp_optimization": edp_plotter.plot_edp_vs_frequency(df, save_path="fig3_edp.png"),
    "pareto_frontier": edp_plotter.plot_pareto_frontier(df, save_path="fig4_pareto.png")
}

print("✅ All publication figures generated!")
```

#### **Interactive Analysis Session**
```python
# Load data
df = ProfilingDataPreprocessor().load_result_directory("your_results/")

# Quick exploration
print(f"Available metrics: {df.columns.tolist()}")
print(f"Frequencies tested: {sorted(df['frequency'].unique())}")
print(f"Data points: {len(df)}")

# Interactive plotting
plotter = PerformancePlotter()
plotter.plot_metric_vs_normalized_time(df, "POWER")  # Shows plot immediately
plotter.plot_metric_vs_normalized_time(df, "GPUTL")  # Compare utilization
plotter.plot_metric_vs_normalized_time(df, "TMPTR")  # Check thermal behavior
```

---

**Need Help?** Run any script with `--help` for detailed usage information.

---

## 🛠️ **Individual Script Usage Guide**

### **1. plot_metric_vs_time.py** - CLI Tool

**Purpose:** Visualization for any DCGMI metric vs normalized time

**Key Features:**
- Multi-frequency overlay plotting
- All DCGMI metrics supported
- Publication-ready output
- Batch processing capabilities

**Quick Start:**
```bash
# Basic usage - GPU utilization over time
python plot_metric_vs_time.py --gpu V100 --app LLAMA --metric GPUTL

# Power consumption analysis
python plot_metric_vs_time.py --gpu V100 --app LLAMA --metric POWER --frequencies 510,960,1380

# Save without displaying (great for batch processing)
python plot_metric_vs_time.py --gpu A100 --app VIT --metric TMPTR --no-show --save thermal_analysis.png

# List all available metrics
python plot_metric_vs_time.py --gpu V100 --app LLAMA --list-metrics
```

**Common Workflows:**
```bash
# 1. Thermal Analysis Workflow
python plot_metric_vs_time.py --gpu V100 --app LLAMA --metric TMPTR --save thermal_llama.png --no-show
python plot_metric_vs_time.py --gpu V100 --app VIT --metric TMPTR --save thermal_vit.png --no-show
python plot_metric_vs_time.py --gpu V100 --app STABLEDIFFUSION --metric TMPTR --save thermal_sd.png --no-show

# 2. Power Optimization Analysis
python plot_metric_vs_time.py --gpu V100 --app LLAMA --metric POWER --frequencies 510,800,1200,1380 --save power_comparison.png --no-show

# 3. Memory Activity Study
python plot_metric_vs_time.py --gpu A100 --app STABLEDIFFUSION --metric DRAMA --frequencies 1200,1410 --save memory_activity.png --no-show

# 4. Performance Validation
python plot_metric_vs_time.py --gpu V100 --app WHISPER --metric GPUTL --run 1 --save perf_run1.png --no-show
python plot_metric_vs_time.py --gpu V100 --app WHISPER --metric GPUTL --run 2 --save perf_run2.png --no-show
```

**Advanced Usage:**
```bash
# Custom data directory (if running from different location)
python plot_metric_vs_time.py --data-dir /path/to/sample-collection-scripts --gpu V100 --app LLAMA --metric POWER

# High-resolution output for publications
python plot_metric_vs_time.py --gpu V100 --app LLAMA --metric POWER --save high_res_power.png --no-show

# Specific run number analysis
python plot_metric_vs_time.py --gpu A100 --app VIT --metric GPUTL --run 3 --frequencies 1000,1200,1410
```

### **2. time_series_demo.py** - Educational/Research Tool

**Purpose:** Interactive demonstration and advanced analysis capabilities

**Key Features:**
- Synthetic data generation for testing
- Multi-metric correlation analysis  
- Advanced statistical visualizations
- Educational examples

**Quick Start:**
```bash
# Generate synthetic demo data and explore
python time_series_demo.py

# Load real data and run advanced analysis
python time_series_demo.py --real-data --gpu V100 --app LLAMA

# Generate correlation analysis
python time_series_demo.py --real-data --gpu A100 --app VIT --analysis correlation

# Export data for external analysis
python time_series_demo.py --real-data --gpu V100 --app LLAMA --export data_export.csv
```

**Research Workflows:**
```bash
# 1. Multi-metric Correlation Study
python time_series_demo.py --real-data --gpu V100 --app LLAMA --analysis correlation --save correlation_matrix.png

# 2. Statistical Analysis
python time_series_demo.py --real-data --gpu A100 --app VIT --analysis stats --frequencies 1200,1410

# 3. Synthetic Data Testing (for algorithm development)
python time_series_demo.py --synthetic --samples 1000 --frequencies 800,1200,1600 --noise 0.1

# 4. Data Quality Assessment
python time_series_demo.py --real-data --gpu V100 --app STABLEDIFFUSION --validate --frequencies 800,1200
```

### **3. data_preprocessor.py** - Data Loading Module

**Purpose:** Programmatic data access and preprocessing utilities

**Key Features:**
- DCGMI CSV parsing
- Frequency extraction
- Data validation
- Time-series preparation

**Programmatic Usage:**
```python
# Import the module
from data_preprocessor import load_dcgmi_data, extract_frequencies

# Load specific experiment data
df = load_dcgmi_data(
    gpu="V100", 
    app="LLAMA", 
    frequencies=[510, 960, 1380], 
    run_number=2
)

# Quick data exploration
print(f"Loaded {len(df)} samples")
print(f"Available metrics: {[col for col in df.columns if col not in ['Entity', 'frequency', 'timestamp']]}")
print(f"Frequency distribution: {df['frequency'].value_counts()}")

# Extract specific metrics for analysis
power_data = df[df['frequency'] == 960]['POWER'].values
gpu_util = df[df['frequency'] == 960]['GPUTL'].values
```

**Command Line Usage:**
```bash
# Validate data availability
python -c "
from data_preprocessor import ProfilingDataLoader
loader = ProfilingDataLoader()
result_dir = loader.find_result_directory('V100', 'LLAMA')
print(f'Result directory: {result_dir}')
"

# Quick data statistics
python -c "
from data_preprocessor import load_dcgmi_data
df = load_dcgmi_data('V100', 'LLAMA', [510, 960, 1380], 2)
print(f'Total samples: {len(df)}')
print(f'Frequency breakdown: {df.groupby(\"frequency\").size().to_dict()}')
"
```

### **4. power_plots.py** - Advanced Power Analysis

**Purpose:** Specialized power consumption visualization and analysis

**Key Features:**
- Power efficiency analysis
- Multi-application power comparison
- Power model validation
- Energy optimization plots

**Programmatic Usage:**
```python
from power_plots import PowerPlotter
from data_preprocessor import load_dcgmi_data

# Load data
df = load_dcgmi_data("V100", "LLAMA", [510, 960, 1380], 2)

# Initialize plotter
plotter = PowerPlotter(figsize=(12, 8))

# Create power analysis plots
fig1 = plotter.plot_power_vs_time(df, save_path="power_timeline.png")
fig2 = plotter.plot_power_temperature_correlation(df, save_path="power_temp_corr.png")

# Advanced energy efficiency analysis
fig3 = plotter.plot_energy_efficiency_analysis(
    df, 
    power_col="POWER",
    performance_col="GPUTL", 
    save_path="efficiency_analysis.png"
)
```

### **5. performance_plots.py** - Performance Analysis

**Purpose:** GPU performance metrics visualization and analysis

**Key Features:**
- GPU utilization analysis
- Memory utilization patterns
- Clock frequency analysis
- Multi-metric dashboards

**Programmatic Usage:**
```python
from performance_plots import PerformancePlotter
from data_preprocessor import load_dcgmi_data

# Load data
df = load_dcgmi_data("A100", "VIT", [1200, 1410], 1)

# Initialize plotter  
plotter = PerformancePlotter()

# Performance analysis
fig1 = plotter.plot_gpu_utilization_vs_time(df, save_path="gpu_util.png")
fig2 = plotter.plot_memory_utilization_analysis(df, save_path="memory_analysis.png")
fig3 = plotter.plot_clock_frequency_analysis(df, save_path="clock_analysis.png")
```

### **6. edp_plots.py** - EDP Optimization Visualization

**Purpose:** Energy-Delay Product optimization and analysis plots

**Key Features:**
- EDP surface plots
- Pareto frontier analysis
- Optimization result visualization
- Multi-dimensional EDP analysis

**Programmatic Usage:**
```python
from edp_plots import EDPPlotter, create_optimization_summary_plot

# Load optimization results
edp_results = load_edp_optimization_results("V100", "LLAMA")

# Create EDP analysis plots
plotter = EDPPlotter()
fig1 = plotter.plot_edp_surface(edp_results, save_path="edp_surface.png")
fig2 = plotter.plot_pareto_frontier(edp_results, save_path="pareto_frontier.png")

# Optimization summary
fig3 = create_optimization_summary_plot(
    edp_results, 
    title="LLAMA V100 EDP Optimization",
    save_path="optimization_summary.png"
)
```

---

## 🚀 **Quick Reference Commands**

### **Most Common Use Cases:**

```bash
# 1. Quick power analysis
python plot_metric_vs_time.py --gpu V100 --app LLAMA --metric POWER

# 2. GPU utilization check  
python plot_metric_vs_time.py --gpu A100 --app VIT --metric GPUTL

# 3. Thermal monitoring
python plot_metric_vs_time.py --gpu V100 --app STABLEDIFFUSION --metric TMPTR

# 4. Memory activity analysis
python plot_metric_vs_time.py --gpu A100 --app WHISPER --metric DRAMA

# 5. Interactive exploration
python time_series_demo.py --real-data --gpu V100 --app LLAMA

# 6. Batch plot generation (no GUI)
for metric in POWER GPUTL TMPTR DRAMA; do
    python plot_metric_vs_time.py --gpu V100 --app LLAMA --metric $metric --no-show --save ${metric}_analysis.png
done
```

### **Troubleshooting Commands:**

```bash
# Check available data
python plot_metric_vs_time.py --gpu V100 --app LLAMA --list-metrics

# Validate data paths
ls -la ../../sample-collection-scripts/results_v100_llama_job_*/

# Test script functionality
python plot_metric_vs_time.py --gpu V100 --app LLAMA --metric POWER --no-show --save test.png

# Check dependencies
python -c "import matplotlib, pandas, numpy; print('✅ All dependencies available')"
```

---

## 📁 **File Structure Reference**

```
edp_analysis/visualization/
├── plot_metric_vs_time.py      # 🎯 Main CLI tool - start here
├── time_series_demo.py         # 📚 Interactive demo and tutorials
├── data_preprocessor.py        # 🔧 Data loading utilities  
├── power_plots.py              # ⚡ Advanced power analysis
├── performance_plots.py        # 📊 GPU performance analysis
├── edp_plots.py                # 🎛️ EDP optimization plots
└── README.md                   # 📖 This comprehensive guide
```

**Ready to start?** Begin with `plot_metric_vs_time.py --help` for the main CLI tool, or `python time_series_demo.py` for interactive exploration! 🚀
