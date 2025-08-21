# EDP Optimizer for AI Inference Energy Project

This directory contains tools for calculating Energy Delay Product (EDP) optimization to find optimal GPU frequencies for AI inference workloads.

## 📁 Files

- **`edp_optimizer.py`** - Main EDP calculation and optimization script
- **`edp_summary_tables.py`** - Summary table generator for results visualization  
- **`edp_optimization_results.json`** - Latest optimization results in JSON format

## 🎯 Purpose

The EDP optimizer analyzes experimental data from GPU frequency scaling experiments to:

1. **Calculate Energy Delay Product (EDP)** for each frequency point
2. **Find optimal frequencies** that minimize EDP while respecting performance constraints  
3. **Compare energy savings** vs. manufacturer maximum frequencies
4. **Identify performance improvements** from frequency optimization

## 🚀 Quick Start

### Basic Usage

```bash
# Run EDP optimization on all experimental data
python edp_optimizer.py --results-dir ../../sample-collection-scripts

# Use custom performance threshold (default: 5%)
python edp_optimizer.py --performance-threshold 10.0

# Generate summary tables from results
python edp_summary_tables.py
```

### Advanced Usage

```bash
# Custom output file and quiet mode
python edp_optimizer.py \
    --results-dir ../../sample-collection-scripts \
    --performance-threshold 5.0 \
    --output my_results.json \
    --quiet

# Generate tables from custom results file
python edp_summary_tables.py --input my_results.json
```

## 📊 Latest Results Summary

Based on the most recent analysis of all GPU-workload combinations:

### Overall Performance
- **Total configurations analyzed**: 12 (A100×4, H100×4, V100×4)
- **Average energy savings**: 27.1%
- **Average EDP improvement**: 31.4%  
- **Configurations faster than max frequency**: 8/12 (66.7%)

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

## 🔍 Advanced Analysis Features

### Frequency Reduction Analysis

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
