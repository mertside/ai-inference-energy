# EDP & ED²P Optimizer with Data Visualization Framework

This directory contains a comprehensive analysis suite for calculating Energy Delay Product (EDP) and Energy Delay Squared Product (ED²P) optimization to find optimal GPU frequencies for AI inference workloads, enhanced with **experimental data visualization** capabilities.

## 📁 Directory Structure

- **`edp_optimizer.py`** - Main EDP & ED²P calculation and optimization script
- **`edp_summary_tables.py`** - Summary table generator for results visualization
- **`results/`** - Output directory containing JSON and CSV results
  - **`edp_optimization_results.json`** - Primary results in JSON format
  - **`*_summary.csv`** - Comprehensive CSV with both EDP and ED²P data
  - **`*_workload_comparison_*.csv`** - Cross-GPU workload analysis files
  - **`*_frequency_analysis.csv`** - Detailed frequency scaling analysis
  - **`*_gpu_summary_*.csv`** - Per-GPU performance summaries
  - **`README.md`** - Detailed documentation of result files
- **`archived/`** - Historical analysis tools and reports

## Data Visualization Framework

### Key Enhancement: Experimental Data Integration
The visualization system is located in `visualization/` and has been **completely enhanced** to use **experimental data** from DCGMI profiling:

- **Real Power Measurements**: Direct integration with DCGMI CSV power profiles (50ms sampling)
- **Actual Execution Times**: Timing data extracted from experimental summary logs
- **Energy Accuracy**: Calculated using measured power × measured execution time
- **Data Coverage**: 824 experimental data points across all GPU-workload combinations
- **Source Transparency**: Clear labeling of data source on all visualizations

### Visualization Scripts Overview

1. **`visualization/visualize_edp_results.py`** - **Primary visualization script with data integration**
   - Uses actual DCGMI profiling data from `sample-collection-scripts/`
   - Creates individual scatter plots for each GPU-workload combination
   - Clear annotations showing EDP/ED²P optimal points with measurements

2. **`visualization/visualize_edp_summary.py`** - **Comparative analysis and summary plots**
   - Energy savings comparison between EDP and ED²P strategies
   - Frequency optimization analysis across GPU architectures
   - Performance impact visualization with statistical analysis
   - 4-panel comprehensive overview dashboard

## 🎯 Purpose

The EDP/ED²P optimizer analyzes experimental data from GPU frequency scaling experiments to:

1. **Calculate both EDP and ED²P** for each frequency point
   - **EDP = Energy × Time** (balanced optimization)
   - **ED²P = Energy × Time²** (performance-sensitive optimization)
2. **Find optimal frequencies** for both metrics while respecting performance constraints
3. **Compare energy savings** vs. manufacturer maximum frequencies
4. **Analyze performance trade-offs** between EDP and ED²P strategies
5. **Export comprehensive data** for further analysis and visualization
6. **Generate publication-quality visualizations** using **experimental data** from DCGMI profiling
7. **Provide visual validation** of optimization results with clear energy-performance trade-off analysis

## 🚀 Quick Start

### Complete Analysis Workflow with Data Visualization

```bash
# 1. Run EDP & ED²P optimization on all experimental data
python edp_optimizer.py --results-dir ../../sample-collection-scripts

# 2. Generate summary tables and CSV exports from results
python edp_summary_tables.py --csv

# 3. Create individual scatter plots using experimental data
cd visualization
python visualize_edp_results.py --input ../results/edp_optimization_results.json --output-dir edp-plots

# 4. Generate comprehensive summary visualizations
python visualize_edp_summary.py --input ../results/edp_optimization_results.json --output-dir edp-plots

# 5. View all 16 generated visualization files
ls edp-plots/*.png
```

### Basic Usage

```bash
# Run EDP & ED²P optimization with default settings (uses ../../sample-collection-scripts automatically)
python edp_optimizer.py

# Or specify custom paths
python edp_optimizer.py --results-dir ../../sample-collection-scripts

# Generate summary tables and CSV exports from results
python edp_summary_tables.py --csv

# Create visualizations with experimental data (RECOMMENDED)
cd visualization
python visualize_edp_results.py --input ../results/edp_optimization_results.json --output-dir edp-plots

# Create comparative summary analysis
python visualize_edp_summary.py --input ../results/edp_optimization_results.json --output-dir edp-plots
```

### Advanced Usage

```bash
# Custom output file and quiet mode
python edp_optimizer.py \
    --results-dir ../../sample-collection-scripts \
    --performance-threshold 5.0 \
    --output results/my_custom_results.json \
    --quiet

# Generate tables from custom results file
python edp_summary_tables.py --input results/my_custom_results.json --csv
```

### Latest Results Summary with Visualization Data

Based on the most recent analysis of all GPU-workload combinations using **experimental data**:

### Overall Performance
- **Total configurations analyzed**: 12 (A100×4, H100×4, V100×4)
- **Data points visualized**: 824 experimental measurements
  - **A100**: 61 frequency points × 4 workloads = 244 measurements
  - **H100**: 86 frequency points × 4 workloads = 344 measurements
  - **V100**: 59 frequency points × 4 workloads = 236 measurements
- **Average energy savings**: 27.1%
- **Average EDP improvement**: 31.4%
- **Configurations faster than max frequency**: 8/12 (66.7%)
- **Visualization files generated**: 16 publication-quality plots

### Best Performers by GPU
- **A100**: 33.5% average energy savings (3/4 faster than max freq)
- **H100**: 24.3% average energy savings (3/4 faster than max freq)
- **V100**: 23.3% average energy savings (2/4 faster than max freq)

### Workload-Specific Champions
- **LLaMA**: A100 (37.1% energy savings, 42.6% frequency reduction)
- **Stable Diffusion**: H100 (30.6% energy savings, 4.2% frequency reduction)
- **Vision Transformer**: H100 (40.3% energy savings, 53.8% frequency reduction)
- **Whisper**: A100 (36.5% energy savings, 38.3% frequency reduction)

## ⚙️ Methodology

### Data Processing Pipeline

1. **Data Extraction**:
   - Timing data from `experiment_summary.log` files
   - Power data from DCGMI CSV profiles (`*_profile.csv`)
   - Cold run elimination (first run at each frequency ignored)

2. **Statistical Aggregation**:
   - Multiple runs per frequency averaged
   - Outlier detection and removal (Z-score > 3)
   - Standard deviation calculation for confidence metrics

3. **EDP Calculation**:
   ```
   EDP = Energy × Execution_Time
   ```
   Where:
   - Energy = Average_Power × Duration (Joules)
   - Execution_Time = Application inference time (seconds)

4. **Optimization Constraints**:
   - Performance degradation ≤ threshold (default: 5%)
   - Constraint applied relative to fastest observed frequency
   - Optimal frequency = minimum EDP among valid frequencies

### Dual Baseline Analysis

The optimizer uses two baseline comparisons:

1. **Maximum Frequency Baseline**: Energy savings vs. manufacturer max frequency
2. **Fastest Execution Baseline**: Performance constraint enforcement

This approach accounts for the fact that maximum frequency often underperforms due to thermal throttling.

## 📈 Understanding the Results

### Energy Savings Interpretation

- **Positive values**: Energy reduction compared to max frequency
- **Range**: Typically 7.5% - 40.3% across configurations
- **Deployment Impact**: Direct translation to reduced power consumption

### Performance Impact Interpretation

- **Negative values**: Performance improvement (faster execution)
- **Positive values**: Performance degradation (slower execution)
- **Constraint**: All results respect ≤5% degradation threshold

### EDP Improvement Interpretation

- **Higher values**: Better energy-performance trade-off
- **Range**: Typically 11.1% - 58.5% across configurations
- **Significance**: Combined energy and performance optimization metric

## 🎨 Visualization Output Summary

### Generated Visualization Files (16 total)

**Individual Scatter Plots (12 files)**:
- `A100_llama_energy_performance_scatter.png` (61 data points)
- `A100_stablediffusion_energy_performance_scatter.png` (61 data points)
- `A100_vit_energy_performance_scatter.png` (61 data points)
- `A100_whisper_energy_performance_scatter.png` (61 data points)
- `H100_llama_energy_performance_scatter.png` (86 data points)
- `H100_stablediffusion_energy_performance_scatter.png` (86 data points)
- `H100_vit_energy_performance_scatter.png` (86 data points)
- `H100_whisper_energy_performance_scatter.png` (86 data points)
- `V100_llama_energy_performance_scatter.png` (59 data points)
- `V100_stablediffusion_energy_performance_scatter.png` (59 data points)
- `V100_vit_energy_performance_scatter.png` (59 data points)
- `V100_whisper_energy_performance_scatter.png` (59 data points)

**Summary Analysis Plots (4 files)**:
- `energy_savings_comparison.png` - EDP vs ED²P energy savings comparison
- `frequency_optimization_comparison.png` - Frequency reduction analysis across GPUs
- `performance_impact_analysis.png` - Performance trade-off visualization
- `comprehensive_summary.png` - 4-panel overview dashboard (largest file: 840KB)

### Visualization Features
- **Data Integration**: All individual plots use actual DCGMI profiling measurements
- **Publication Quality**: 300 DPI resolution suitable for research papers and presentations
- **Clear Annotations**: EDP and ED²P optimal points clearly marked with performance statistics
- **Professional Styling**: Consistent color schemes, legends, and formatting across all visualizations

## 🔍 Advanced Visualization Features

### Data Integration Pipeline

**Experimental Data Loading**:
1. **Directory Discovery**: Automatically finds result directories matching `results_{gpu}_{workload}_job_*` pattern
2. **DCGMI Data Parsing**: Loads power measurements from CSV profiles with robust column detection
3. **Timing Extraction**: Extracts execution times from `timing_summary.log` files
4. **Energy Calculation**: Computes total energy using `measured_power × measured_time`
5. **Data Validation**: Ensures minimum 3 data points for meaningful visualization

**Data Quality Metrics**:
- **A100 Coverage**: 61/61 frequencies with data (100% coverage)
- **H100 Coverage**: 86/86 frequencies with data (100% coverage)
- **V100 Coverage**: 59/59 frequencies with data (100% coverage)
- **Sampling Rate**: 50ms DCGMI power measurements for high precision
- **Energy Accuracy**: Direct measurement vs estimation-based calculation

### Advanced Analysis Features

The optimizer calculates frequency reductions ranging from:
- **Minimal**: 4.2% (H100 Stable Diffusion)
- **Aggressive**: 60.9% (V100 Whisper)
- **Typical**: 20-50% across most configurations

### Cross-GPU Performance Comparison

Results enable direct comparison of:
- GPU architecture efficiency for specific workloads
- Optimal frequency characteristics by workload type
- Energy scaling behavior across different hardware

### Statistical Confidence Metrics

Each optimal frequency includes:
- Number of runs averaged (typically 4)
- Timing standard deviation
- Confidence in measurement reliability

## Visualization Tools

### Individual Scatter Plot Analysis
The `visualize_edp_results.py` script creates detailed scatter plots for each GPU-workload combination:

- **Energy vs Performance Trade-offs**: Shows the Pareto frontier of energy-performance choices
- **Frequency Color-Coding**: Visualizes the entire frequency sweep with color gradients
- **Optimal Point Annotations**: Clearly marks EDP and ED²P optimal frequencies with savings data
- **Performance Metrics**: Includes detailed statistics boxes with quantitative results

### Comprehensive Summary Analysis
The `visualize_edp_summary.py` script generates comparative analysis across all configurations:

- **Energy Savings Comparison**: Side-by-side bar charts comparing EDP vs ED²P strategies
- **Frequency Selection Analysis**: Scatter plots showing how different strategies choose frequencies
- **Performance Impact Distribution**: Histograms and trade-off analysis of performance implications
- **4-Panel Dashboard**: Comprehensive overview with key insights and summary statistics

### Visualization Features
- **Publication Quality**: 300 DPI resolution suitable for papers and presentations
- **Professional Styling**: Clean, consistent design with clear legends and annotations
- **Color-Coded Analysis**: GPU architectures and optimization strategies clearly distinguished
- **Detailed Annotations**: Quantitative results displayed directly on plots
- **Export Flexibility**: PNG format with customizable output directories

### Quick Visualization Workflow
```bash
# Generate all individual plots (12 scatter plots)
cd visualization
python visualize_edp_results.py --input ../results/edp_optimization_results.json --output-dir edp-plots

# Generate all summary comparisons (4 summary plots)
python visualize_edp_summary.py --input ../results/edp_optimization_results.json --output-dir edp-plots

# View results in edp-plots/ directory
```

## 🛠️ Technical Implementation

### Input Data Requirements

The optimizer expects the standard experimental directory structure:
```
sample-collection-scripts/
├── results_<gpu>_<workload>_job_<jobid>/
│   ├── experiment_summary.log          # Timing data
│   ├── run_X_YY_freq_ZZZZ_app.out     # Application output
│   └── run_X_YY_freq_ZZZZ_profile.csv # Power profiling data
```

### CSV Power Data Parsing

Robust parsing handles different GPU naming formats:
- **A100**: `NVIDIA A100-PCIE-40GB`
- **H100**: `NVIDIA H100 NVL`
- **V100**: `NVIDIA Tesla V100-SXM2-32GB`

### Error Handling

- Graceful handling of missing files
- Data validation (positive power/timing values)
- Frequency mismatch detection and reporting
- Comprehensive warning system for data quality issues

## 📋 Output Formats

### JSON Export (`edp_optimization_results.json`)

Complete machine-readable results including:
- Optimal frequency recommendations
- Energy savings and performance metrics
- Statistical confidence data
- Baseline comparison metrics

### Console Summary

Human-readable analysis with:
- Per-GPU performance breakdowns
- Cross-workload comparisons
- Statistical reliability indicators
- Deployment recommendations

### Summary Tables (`edp_summary_tables.py`)

Formatted tables providing:
- GPU comparison matrices
- Workload-specific best performers
- Frequency reduction analysis
- Visual performance indicators

## 🎯 Deployment Recommendations

Based on the analysis results, the recommended deployment frequencies are:

### Production-Ready Configurations
1. **A100 + LLaMA**: 810 MHz (37.1% energy savings, 1.8% faster)
2. **H100 + Vision Transformer**: 825 MHz (40.3% energy savings, 30.5% faster)
3. **A100 + Whisper**: 870 MHz (36.5% energy savings, 20.0% faster)
4. **H100 + Stable Diffusion**: 1710 MHz (30.6% energy savings, same performance)

These configurations provide substantial energy savings while maintaining or improving performance compared to default maximum frequency operation.
