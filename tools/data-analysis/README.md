# Data Analysis Tools

This directory contains scripts for analyzing experimental data and extracting insights about optimal GPU frequencies.

## Scripts Overview

### Core Analysis Scripts

**`corrected_real_optimal.py`**
- **Purpose**: Corrected real data-driven optimal frequency analysis
- **Features**: Handles duplicates, analyzes frequency-performance relationships
- **Usage**: `python corrected_real_optimal.py`
- **Output**: Optimal frequencies with energy savings and performance impact

**`extract_a100_optimal.py`**
- **Purpose**: Extract optimal frequencies specifically for A100 GPUs
- **Features**: Follows same methodology as H100 analysis
- **Usage**: `python extract_a100_optimal.py`
- **Note**: A100 data shows anomalous behavior requiring careful analysis

**`real_data_optimal_analysis.py`**
- **Purpose**: Comprehensive analysis using real experimental data
- **Features**: EDP optimization, performance constraint validation
- **Usage**: `python real_data_optimal_analysis.py`

### Supporting Analysis Tools

**`data_source_summary.py`**
- **Purpose**: Summary of data sources and reliability levels
- **Features**: Provides transparency on data confidence levels
- **Usage**: `python data_source_summary.py`

**`edp_analysis.py`**
- **Purpose**: Energy-Delay Product analysis
- **Features**: EDP calculation and optimization metrics

**`power_modeling.py`**
- **Purpose**: Power modeling utilities
- **Features**: Power consumption modeling and prediction

## Data Processing

All scripts process CSV data from GPU profiling experiments with the following key columns:
- `gpu`: GPU type (V100, A100, H100)
- `workload`: AI workload (llama, vit, stablediffusion, whisper)
- `frequency_mhz`: GPU frequency in MHz
- `duration_seconds`: Execution time
- `avg_power_watts`: Average power consumption
- `total_energy_joules`: Total energy consumption

## Key Findings

- **H100**: All workloads show normal frequency-performance relationships
- **A100**: Real data shows anomalous behavior (90%+ performance degradation)
- **V100**: Limited real data available

## Expected Output Format

Scripts typically output:
- Optimal frequency (MHz)
- Energy savings percentage
- Performance impact percentage
- EDP improvement
- Data source reliability indicator
