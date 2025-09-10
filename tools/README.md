# Tools Directory

This directory contains all the analysis, deployment, and utility tools for the AI Inference Energy project, organized into specialized subdirectories for better maintainability.

## Directory Structure

### 📊 Analysis (`analysis/`)
Core analysis scripts for EDP (Energy-Delay Product) optimization and visualization:
- `edp_optimizer.py` - Main EDP & ED²P optimization engine
- `edp_summary_tables.py` - EDP results summarization and table generation
- `results/` - Analysis outputs
  - `edp_optimization_results.json` - Primary optimization results
  - `*.csv` - Detailed analysis tables
- `archived/` - Historical analysis scripts and reports
  - `measured_data_analysis_v*.py` - Legacy measured data analysis versions
  - `edp_analysis.py` - Original EDP analysis implementation
  - `power_modeling.py` - Power and performance modeling
  - `aggregate_data.py` - Data aggregation utilities
  - `comprehensive_report.py` - Report generation
- **`visualization/`** - Comprehensive data visualization framework for experimental results
  - `visualize_edp_results.py` - 🎨 **Real experimental data visualization** using DCGMI profiling data
  - `visualize_edp_summary.py` - 📊 **Comprehensive summary analysis** with comparative charts
  - `README.md` - Complete visualization system documentation
  - `edp-plots/` - Generated visualization outputs directory
  - **Generated Outputs:**
    - 12 individual GPU-workload scatter plots using real DCGMI measurements
    - 4 summary comparison charts and multi-GPU analysis
    - Publication-quality 300 DPI PNG files

### 🤖 ML Prediction (`ml_prediction/`)
End-to-end pipeline for learning to predict EDP‑optimal GPU frequencies from short profiling runs:
- `build_labels.py` / `label_builder.py` — Generate ground‑truth labels (EDP & ED²P optimal) using `analysis/edp_optimizer.py`
- `build_dataset.py` / `dataset_builder.py` — Build training datasets from probe policies: `max-only`, `tri-point`, `all-freq`
- `feature_extractor.py` — Derive statistics, trend slopes, ratios, and HAL‑normalized features from DCGMI profiles
- `train_baseline.py` + `models/random_forest_predictor.py` — Baseline RandomForest with frequency snapping and confidence
- `evaluate.py` — Cross‑workload/GPU splits with frequency error + EDP gap metrics
- Outputs: `tools/ml_prediction/labels.json`, `tools/ml_prediction/datasets/*.csv`, `tools/ml_prediction/models/*.joblib`

### 📡 Data Collection (`sample-collection-scripts/`)
Data collection and profiling tools for GPU workloads (repository root):
- `profile.py` - DCGMI profiling interface for detailed metrics
- `profile_smi.py` - nvidia-smi profiling fallback
- `control.sh` - Main control script for profiling automation
- `control_smi.sh` - Alternative control script using nvidia-smi
- `interactive_gpu.sh` - Interactive GPU session management
- `launch_v2.sh` - Enhanced launch script for profiling sessions
- `submit_job_*.sh` - Job submission scripts for A100, H100, and V100 GPUs

### 🚀 Deployment (planned)
Note: Deployment interfaces are planned for a future release and are not present in this repository snapshot.

### ⚙️ Optimal Frequency (planned)
Note: Dedicated optimal-frequency tools are planned and not included in this release.

### 🧪 Testing (planned)
Note: Additional testing utilities described here are planned and not included in this release.

### 🔧 Utilities (planned)
Note: General utilities referenced here are planned and not included in this release.

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
cd visualization
python visualize_edp_results.py --input ../results/edp_optimization_results.json --output-dir edp-plots

# 3. Create summary comparison analysis
python visualize_edp_summary.py --input ../results/edp_optimization_results.json --output-dir edp-plots

# 4. View all 16 generated visualization files
ls edp-plots/*.png
```

## 🤖 ML Prediction Workflow

Run from repository root:

```bash
# 1) Build labels (EDP/ED²P) from experimental results
python -m tools.ml_prediction.build_labels \
  --results-dir sample-collection-scripts \
  --performance-threshold 5.0 \
  --output tools/ml_prediction/labels.json

# 2) Build dataset (recommend all-freq for training)
python -m tools.ml_prediction.build_dataset \
  --results-dir sample-collection-scripts \
  --labels tools/ml_prediction/labels.json \
  --output tools/ml_prediction/datasets/all_freq.csv \
  --policy all-freq

# 3) Train baseline RF and view quick metrics
python -m tools.ml_prediction.train_baseline \
  --dataset tools/ml_prediction/datasets/all_freq.csv \
  --model-out tools/ml_prediction/models/rf_all_freq.joblib

# 4) Evaluate with EDP gap (requires all-freq dataset)
python -m tools.ml_prediction.evaluate \
  --dataset tools/ml_prediction/datasets/all_freq.csv \
  --labels tools/ml_prediction/labels.json \
  --split workload --holdout-workloads llama
```

## Usage

Run scripts from the project root directory:

```bash
# Main EDP optimization
python tools/analysis/edp_optimizer.py

# Generate EDP summary tables
python tools/analysis/edp_summary_tables.py

# Experimental data visualization (RECOMMENDED)
python tools/analysis/visualization/visualize_edp_results.py

# Comprehensive summary analysis charts
python tools/analysis/visualization/visualize_edp_summary.py

# Data collection with DCGMI profiling
python sample-collection-scripts/profile.py --help

# Alternative profiling with nvidia-smi
python sample-collection-scripts/profile_smi.py --help

# (Planned components below are not available in this release.)
# Optimal frequency selection, deployment interface, and additional testing utilities
# will be added in a future version.
```

## Tool Dependencies

- **Core Analysis**: Python 3.10+, pandas, numpy
- **EDP Optimization**: scikit-learn, matplotlib, seaborn
- **Data Visualization**: matplotlib, numpy, pandas, pathlib (for DCGMI integration)
- **ML Prediction**: scikit-learn, joblib; optional `pyarrow` for Parquet I/O
- **Profiling**: NVIDIA DCGMI (primary); nvidia-smi fallback for collection (ML pipeline parses DCGMI)
- **Data Collection**: subprocess, threading
- **Testing**: pytest (optional)

## Output Files

Scripts generate various output files:
- `edp_optimization_results.json` - Main EDP optimization results
- `edp_optimization_results_*.csv` - Analysis summaries and comparisons
- **`edp-plots/*.png`** - 🎨 **16 visualization files**
  - Individual scatter plots for each GPU-workload combination
  - Summary analysis charts and comparative visualizations
  - Experimental data integration with 824 data points total
- `tools/ml_prediction/labels.json` — Ground‑truth label records per GPU–workload
- `tools/ml_prediction/datasets/*.csv` — Training/evaluation datasets (`max_only.csv`, `all_freq.csv`, …)
- `tools/ml_prediction/models/*.joblib` — Saved ML models (e.g., `rf_all_freq.joblib`)
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

## Known Limitations and Notes

- ML prediction pipeline currently parses DCGMI profiles; direct nvidia‑smi parsing within ML tools is planned.
- EDP gap evaluation relies on datasets with one row per frequency (`all-freq` policy). Sparse datasets may reduce coverage.
- HAL‑normalized features are best‑effort and skipped when unsupported for a GPU type.
- Parquet I/O requires optional dependencies (e.g., `pyarrow`); CSV works by default.

## Roadmap / TODOs

- Integrate nvidia‑smi parsing path into ML tools
- Add an inference CLI for single‑run prediction (load model → frequency + confidence)
- Hyperparameter tuning and alternative models (XGB/LightGBM/NN) with split tracking
- Percentile/worst‑case reporting and feature importances in evaluation outputs
- Packaging for tools (entry points) and reproducible environments (conda/uv lockfiles)
