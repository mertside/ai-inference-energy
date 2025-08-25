# Tools Directory

This directory contains all the analysis, deployment, and utility tools for the AI Inference Energy project.

## Directory Structure

# Tools Directory

This directory contains all the analysis, deployment, and utility tools for the AI Inference Energy project, including a **comprehensive data visualization framework**.

## Directory Structure

### 📊 Analysis (`analysis/`)
Core analysis scripts for EDP (Energy-Delay Product) optimization with **data visualization**:
- `edp_optimizer.py` - Main EDP & ED²P optimization engine
- `edp_summary_tables.py` - EDP results summarization and table generation
- **`visualize_results.py`** - 🎨 **Experimental data visualization** using DCGMI profiling data
- **`visualize_summary.py`** - 📊 **Comprehensive summary analysis** with comparative charts
- **`README_VISUALIZATION.md`** - Complete visualization system documentation
- **`IMPLEMENTATION_SUMMARY.md`** - Data integration achievement summary
- **`results/`** - Analysis outputs and visualizations
  - `edp_optimization_results.json` - Primary optimization results
  - `*.csv` - Detailed analysis tables
  - **`plots/`** - 🎨 **Generated visualization files (16 total)**
    - Individual GPU-workload scatter plots using DCGMI data
    - Summary comparison charts and multi-GPU analysis
    - Publication-quality 300 DPI PNG files
- `archived/` - Historical analysis scripts and reports
  - `measured_data_analysis_v*.py` - Legacy measured data analysis versions
  - `edp_analysis.py` - Original EDP analysis implementation
  - `power_modeling.py` - Power and performance modeling
  - `aggregate_data.py` - Data aggregation utilities
  - `comprehensive_report.py` - Report generation

### 📡 Data Collection (`data-collection/`)
Data collection and profiling tools for GPU workloads:
- `profile.py` - DCGMI profiling interface for detailed metrics
- `profile_smi.py` - nvidia-smi profiling fallback
- `control.sh` - Main control script for profiling automation
- `control_smi.sh` - Alternative control script using nvidia-smi
- `interactive_gpu.sh` - Interactive GPU session management
- `launch_v2.sh` - Enhanced launch script for profiling sessions
- `submit_job_*.sh` - Job submission scripts for A100, H100, and V100 GPUs

### 🚀 Deployment (`deployment/`)
Production deployment tools:
- `deployment_interface.py` - Production deployment interface for optimal frequency settings

### ⚙️ Optimal Frequency (`optimal-frequency/`)
Frequency optimization and selection tools:
- `comprehensive_optimal_selector.py` - Comprehensive frequency selector with multiple criteria
- `optimal_frequency_analysis.py` - Frequency analysis and performance evaluation
- `optimal_frequency_selection.py` - Core frequency selection algorithms
- `optimal_frequency_selector.py` - Basic frequency selector implementation
- `production_optimal_selector.py` - Production-ready frequency selector

### 🧪 Testing (`testing/`)
Testing and validation tools:
- `test_optimal_frequency.py` - Optimal frequency algorithm testing
- `test_real_data.py` - Real data validation tests
- `quick_frequency_demo.py` - Quick demonstration and testing scripts

### 🔧 Utilities (`utilities/`)
General utility tools:
- `ai_optimization_workflow.py` - End-to-end AI optimization workflows

## 🎨 Data Visualization Framework

### Key Features
- **DCGMI Integration**: Uses actual power measurements from experimental data
- **Publication Quality**: 300 DPI plots suitable for research papers and presentations  
- **Comprehensive Coverage**: 16 visualization files covering all GPU-workload combinations
- **Energy-Performance Analysis**: Detailed scatter plots showing optimization trade-offs

### Visualization Workflow
```bash
# Complete visualization generation workflow
cd tools/analysis

# 1. Run optimization analysis
python edp_optimizer.py --results-dir ../../sample-collection-scripts

# 2. Generate individual scatter plots with DCGMI data
python visualize_results.py --input results/edp_optimization_results.json --output-dir results/plots

# 3. Create summary comparison analysis
python visualize_summary.py --input results/edp_optimization_results.json --output-dir results/plots

# 4. View all 16 generated visualization files
ls results/plots/*.png
```

## Usage

Run scripts from the project root directory:

```bash
# Main EDP optimization
python tools/analysis/edp_optimizer.py

# Generate EDP summary tables
python tools/analysis/edp_summary_tables.py

# Experimental data visualization (RECOMMENDED)
python tools/analysis/visualize_results.py

# Comprehensive summary analysis charts
python tools/analysis/visualize_summary.py

# Data collection with DCGMI profiling
python tools/data-collection/profile.py

# Alternative profiling with nvidia-smi
python tools/data-collection/profile_smi.py

# Comprehensive optimal frequency selection
python tools/optimal-frequency/comprehensive_optimal_selector.py

# Production frequency selection
python tools/optimal-frequency/production_optimal_selector.py

# Deployment interface
python tools/deployment/deployment_interface.py

# Quick frequency demo
python tools/testing/quick_frequency_demo.py
```

## Tool Dependencies

- **Core Analysis**: Python 3.8+, pandas, numpy
- **EDP Optimization**: scikit-learn, matplotlib, seaborn
- **Data Visualization**: matplotlib, numpy, pandas, pathlib (for DCGMI integration)
- **Profiling**: NVIDIA DCGMI, nvidia-smi
- **Data Collection**: subprocess, threading
- **Testing**: pytest (optional)

## Output Files

Scripts generate various output files:
- `edp_optimization_results.json` - Main EDP optimization results
- `edp_optimization_results_*.csv` - Analysis summaries and comparisons
- **`results/plots/*.png`** - 🎨 **16 publication-quality visualization files**
  - Individual scatter plots for each GPU-workload combination
  - Summary analysis charts and comparative visualizations
  - Experimental data integration with 824 data points total
- `MEASURED_DATA_OPTIMAL_FREQUENCIES.md` - Legacy results report (archived)
- `measured_data_optimal_frequencies_deployment.json` - Deployment config (archived)
- Analysis reports and model outputs in respective directories

## 🎯 Visualization Output Summary

The enhanced visualization framework generates **16 total files**:

### Individual Scatter Plots (12 files)
- DCGMI data integration with 59-86 frequency points per GPU
- Energy vs execution time analysis with EDP/ED²P optimal point annotations

### Summary Analysis Charts (4 files)  
- Energy savings comparison between EDP and ED²P strategies
- Frequency optimization analysis across GPU architectures
- Performance impact visualization with statistical analysis
- Comprehensive 4-panel overview dashboard

All visualizations are **publication-ready at 300 DPI** and use **experimental data** from DCGMI profiling when available.
