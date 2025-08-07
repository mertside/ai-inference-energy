# Tools Directory Organization

This directory contains the reorganized tools and scripts for AI inference energy optimization, organized by functionality.

## Directory Structure

### `/data-analysis/`
Scripts for analyzing experimental data and extracting insights:
- `corrected_real_optimal.py` - Corrected real data-driven optimal frequency analysis
- `corrected_optimal_analysis.py` - Optimal frequency analysis with corrections
- `extract_a100_optimal.py` - Extract optimal frequencies specifically for A100 GPUs
- `real_data_optimal_analysis.py` - Analysis using real experimental data
- `data_source_summary.py` - Summary of data sources and reliability
- `aggregate_data.py` - Data aggregation utilities
- `aggregate_results.py` - Results aggregation
- `comprehensive_report.py` - Comprehensive reporting
- `edp_analysis.py` - Energy-Delay Product analysis
- `power_modeling.py` - Power modeling utilities

### `/optimal-frequency/`
Scripts for optimal frequency selection and management:
- `comprehensive_optimal_selector.py` - Comprehensive optimal frequency selection
- `production_optimal_selector.py` - Production-ready optimal frequency selector
- `real_data_optimal_selector.py` - Real data-based optimal frequency selector
- `fixed_real_optimal.py` - Fixed version of real optimal frequency selection
- `simple_real_optimal.py` - Simplified real optimal frequency selection
- `real_data_optimal_frequency.py` - Real data optimal frequency utilities
- `optimal_frequency_analysis.py` - Frequency analysis utilities
- `optimal_frequency_selection.py` - Frequency selection algorithms
- `optimal_frequency_selector.py` - Main frequency selector

### `/deployment/`
Scripts for production deployment and interfaces:
- `deployment_interface.py` - Simple deployment interface for production use

### `/testing/`
Scripts for testing and validation:
- `test_optimal_frequency.py` - Test optimal frequency selection
- `test_real_data.py` - Test real data processing
- `quick_frequency_demo.py` - Quick demonstration of frequency selection

### `/utilities/`
Utility scripts and workflows:
- `ai_optimization_workflow.py` - Complete AI optimization workflow

### `/data-collection/`
Scripts for data collection:
- `profile.py` - GPU profiling utilities
- `profile_smi.py` - nvidia-smi based profiling

## Usage

Each directory contains focused functionality. Scripts can be run from the project root directory:

```bash
# Run data analysis
python tools/data-analysis/corrected_real_optimal.py

# Use deployment interface
python tools/deployment/deployment_interface.py

# Run tests
python tools/testing/test_optimal_frequency.py
```

## Migration from sample-collection-scripts

Scripts have been reorganized from the `sample-collection-scripts` directory into this organized structure for better maintainability and clarity.
