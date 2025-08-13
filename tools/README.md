# Tools Directory

This directory contains all the analysis, deployment, and utility tools for the AI Inference Energy project.

## Directory Structure

### 📊 Analysis (`analysis/`)
Core analysis scripts for processing measured experimental data:
- `measured_data_analysis.py` - Main analysis using 100% measured experimental data
- `edp_analysis.py` - Energy-Delay Product (EDP) analysis
- `power_modeling.py` - Power and performance modeling
- `aggregate_data.py` - Data aggregation utilities
- `comprehensive_report.py` - Report generation

### 📡 Data Collection (`data-collection/`)
Data collection and profiling tools:
- `profile.py` - DCGMI profiling interface
- `profile_smi.py` - nvidia-smi profiling fallback

### 🚀 Deployment (`deployment/`)
Production deployment tools:
- `deployment_interface.py` - Production deployment interface

### ⚙️ Optimal Frequency (`optimal-frequency/`)
Frequency optimization tools:
- `comprehensive_optimal_selector.py` - Comprehensive frequency selector
- `optimal_frequency_analysis.py` - Frequency analysis utilities
- `production_optimal_selector.py` - Production-ready selector

### 🧪 Testing (`testing/`)
Testing and validation tools:
- `test_optimal_frequency.py` - Optimal frequency testing
- `quick_frequency_demo.py` - Quick demonstration scripts

### 🔧 Utilities (`utilities/`)
General utility tools:
- `ai_optimization_workflow.py` - AI optimization workflows

## Usage

Run scripts from the project root directory:

```bash
# Main measured data analysis
python tools/analysis/measured_data_analysis.py

# EDP analysis
python tools/analysis/edp_analysis.py aggregated_results.csv

# Data aggregation
python tools/analysis/aggregate_data.py sample-collection-scripts/

# Deployment interface
python tools/deployment/deployment_interface.py
```

## Tool Dependencies

- **Core Analysis**: Python 3.8+, standard library
- **Advanced Analysis**: pandas, numpy, scikit-learn
- **Profiling**: NVIDIA DCGMI, nvidia-smi
- **Testing**: pytest (optional)

## Output Files

Scripts generate various output files:
- `MEASURED_DATA_OPTIMAL_FREQUENCIES.md` - Main results report
- `measured_data_optimal_frequencies_deployment.json` - Deployment config
- Analysis reports and model outputs in respective directories
