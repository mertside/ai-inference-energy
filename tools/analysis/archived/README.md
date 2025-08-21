# Analysis Tools

This directory contains all analysis scripts for the AI Inference Energy project. These tools process measured experimental data to identify optimal GPU frequencies for energy-efficient AI inference.

## Core Analysis Scripts

### Primary Analysis
- **`measured_data_analysis_v3.py`** - Main analysis script using 100% measured experimental data with corrected timing
  - Extracts optimal frequencies from NVIDIA DCGMI profiling data
  - Uses proper inference timing from .out files (not total session time)
  - Applies ≤5% performance constraint with max frequency baseline
  - Includes statistical aggregation, cold-start exclusion, and outlier detection
  - Usage: `python measured_data_analysis_v3.py`

### Data Processing
- **`aggregate_data.py`** - Consolidates DVFS profiling results across all GPUs and workloads
- **`aggregate_results.py`** - Aggregates and processes experimental results
- **`data_source_summary.py`** - Summarizes data sources and experimental coverage

### Advanced Analysis
- **`edp_analysis.py`** - Energy-Delay Product (EDP) and ED²P analysis for optimal frequency selection
- **`power_modeling.py`** - Power and performance modeling for AI inference workloads
- **`corrected_optimal_analysis.py`** - Data quality validation and corrected analysis

### Reporting
- **`comprehensive_report.py`** - Generates comprehensive analysis reports

## Usage Examples

```bash
# Main analysis using measured data
python tools/analysis/measured_data_analysis_v3.py

# Aggregate experimental data
python tools/analysis/aggregate_data.py sample-collection-scripts/

# EDP analysis
python tools/analysis/edp_analysis.py aggregated_results.csv

# Power modeling
python tools/analysis/power_modeling.py aggregated_results.csv
```

## Data Flow

1. **Data Collection** → Raw DCGMI CSV files and timing logs
2. **Aggregation** → `aggregate_data.py` consolidates data
3. **Analysis** → `measured_data_analysis_v3.py` finds optimal frequencies
4. **Validation** → `corrected_optimal_analysis.py` validates results
5. **Modeling** → `power_modeling.py` creates predictive models
6. **Reporting** → `comprehensive_report.py` generates summaries

## Script Evolution

**v3 (Current)**: `measured_data_analysis_v3.py`
- ✅ **Corrected timing extraction**: Uses proper inference timing from .out files
- ✅ **Max frequency baseline**: Proper energy optimization reference point
- ✅ **Statistical aggregation**: Cold-start exclusion and outlier detection
- ✅ **Enhanced reliability**: Multi-run averaging with standard deviations

**Previous versions**: `measured_data_analysis_v1.py`, `measured_data_analysis_v2.py`, `measured_data_analysis_corrected.py`
- Available for comparison and debugging
- Not recommended for production use

## Output Files

- `MEASURED_DATA_OPTIMAL_FREQUENCIES.md` - Main results report
- `measured_data_optimal_frequencies_deployment.json` - Deployment configuration
- `aggregated_results.csv` - Consolidated experimental data
- Various analysis reports and model outputs
