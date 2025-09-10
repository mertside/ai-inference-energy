# ML Prediction Tools — User Guide

This guide explains how to train and evaluate ML models that predict EDP‑optimal GPU frequency from short profiling runs. It builds on profiling results under `sample-collection-scripts/` and the analysis pipeline in `tools/analysis/`.

## Components

- `profile_reader.py`: DCGMI profile parsing, warm‑run aggregation, optional IQR outlier filtering. (nvidia‑smi integration planned.)
- `feature_extractor.py`: Converts profiles to features (POWER, GPUTL, MCUTL, SMCLK, MMCLK, TMPTR, SMACT, DRAMA) plus ratios, trend slopes (POWER/TMPTR/GPUTL), and HAL‑normalized clocks.
- `label_builder.py`: Generates ground‑truth labels (EDP & ED²P optimal) via `tools/analysis/edp_optimizer.py`.
- `dataset_builder.py`: Builds datasets from probe policies: `max-only`, `tri-point`, `all-freq`.
- `models/random_forest_predictor.py`: Baseline RF classifier with frequency snapping and confidence estimate.
- `feature_importance.py`: Save feature importances to CSV/JSON/Markdown and render a top‑N bar plot.
- `evaluate.py`: Cross‑workload/GPU evaluation with EDP gap reporting. Trainer/evaluator print and can save aggregated feature importances.

## Feature Set (Extracted)

The dataset builder invokes `feature_extractor.py` to compute the following, when available in DCGMI profiles:

- Stats per metric: mean, std, min, max, p95 for POWER, GPUTL, MCUTL, SMCLK, MMCLK, TMPTR, SMACT, DRAMA.
- Trend slopes: POWER, TMPTR, GPUTL slopes over the last N samples (default N=30), reported per‑second using `sampling_interval_ms` (default 50 ms).
- Ratios and relationships: `mem_to_gpu_ratio` (MCUTL/GPUTL), `power_efficiency` (GPUTL/POWER), `sm_to_mem_clock_ratio` (SMCLK/MMCLK), `gputl_per_mhz` (GPUTL/SMCLK), `utilization_balance` (GPUTL − MCUTL).
- HAL normalization (best‑effort): `smclk_norm_hal_max` = mean(SMCLK)/HAL core max; `mmclk_norm_hal_mem` = mean(MMCLK)/HAL mem clock.
- Context: `gpu_type`, `probe_policy`, `sampling_interval_ms`, plus dataset‑level fields: `probe_frequency_mhz`, `probe_freq_ratio`, `max_frequency_mhz`, `duration_seconds`, `energy_estimate_j`.

Notes:
- Slopes are robust to missing values and omitted if insufficient samples (<5).
- HAL features are added only if HAL is available for the `gpu_type`.

## Environment Setup

Prerequisites (local environment):
- Python 3.10+ (code uses modern typing syntax)
- numpy, pandas, scikit‑learn, joblib (for training)

Example conda environment:
```bash
conda create -n ai-freq-ml python=3.10 -y
conda activate ai-freq-ml
# Always install packages via the env's Python to avoid mixing environments
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn joblib
```

Troubleshooting imports (macOS / multiple Pythons):
```bash
# Verify you're using the env's Python and pip
which python
python -V
python -m pip -V
```

Important: run from the repository root (so `tools` is on the Python path), or use the direct script form (both supported).

## End‑to‑End Workflow

0) Collect profiling results (once per workload/GPU)
- Use scripts under `sample-collection-scripts/` to generate `results_<GPU>_<workload>_job_<id>/` with `run_*_profile.csv` and a `timing_summary.log`.
- For datasets covering many frequencies (recommended for training + EDP gap eval), run with a DVFS sweep so multiple `probe_frequency_mhz` values exist per workload.

1) Build labels (EDP/ED²P) from experimental results
```bash
python -m tools.ml_prediction.build_labels \
  --results-dir sample-collection-scripts \
  --performance-threshold 5.0 \
  --output tools/ml_prediction/labels.json
```
Alternative (direct script):
```bash
python tools/ml_prediction/build_labels.py \
  --results-dir sample-collection-scripts \
  --performance-threshold 5.0 \
  --output tools/ml_prediction/labels.json
```

2) Build a dataset (choose a probe policy)
```bash
python -m tools.ml_prediction.build_dataset \
  --results-dir sample-collection-scripts \
  --labels tools/ml_prediction/labels.json \
  --output tools/ml_prediction/datasets/max_only.csv \
  --policy max-only
```
Other options:
```bash
# Use a single sample for every available frequency (larger dataset)
python -m tools.ml_prediction.build_dataset \
  --results-dir sample-collection-scripts \
  --labels tools/ml_prediction/labels.json \
  --output tools/ml_prediction/datasets/all_freq.csv \
  --policy all-freq
```
Alternative (direct script): same flags using `python tools/ml_prediction/build_dataset.py ...`

3) Train baseline RF and view quick metrics
Recommended (more data → better results):
```bash
python -m tools.ml_prediction.train_baseline \
  --dataset tools/ml_prediction/datasets/all_freq.csv \
  --model-out tools/ml_prediction/models/rf_all_freq.joblib
```
Quick sanity check (smaller dataset):
```bash
python -m tools.ml_prediction.train_baseline \
  --dataset tools/ml_prediction/datasets/max_only.csv \
  --model-out tools/ml_prediction/models/rf_max_only.joblib
```
Alternative (direct script): same flags using `python tools/ml_prediction/train_baseline.py ...`

Expected output (random split, all‑freq):
```
=== Evaluation (label_edp) ===
Frequency error (median): XX.X MHz
Frequency error (mean):   XX.X MHz
Within 30 MHz: YY.Y%  |  Within 60 MHz: ZZ.Z%
Model saved to tools/ml_prediction/models/rf_all_freq.joblib
```

Tips to improve baseline results:
- Prefer `--policy all-freq` to generate many more training samples (one per frequency),
  which typically improves learning substantially over `max-only` (12 samples).
- Prefer enriched features: trends (POWER/TMPTR/GPUTL) + normalized clocks via HAL; probe frequency features from `all-freq` policy significantly help.
- Evaluate with cross‑workload and cross‑GPU splits to check generalization.

4) Evaluate with EDP gap (requires `all-freq` dataset)
```bash
python -m tools.ml_prediction.evaluate \
  --dataset tools/ml_prediction/datasets/all_freq.csv \
  --labels tools/ml_prediction/labels.json \
  --split workload --holdout-workloads llama
```
The evaluator prints frequency error and EDP gap (predicted vs optimal EDP = energy × time). It avoids split leakage (e.g., excludes `workload` from features for workload holdout). If an exact (gpu, workload, freq) mapping is missing, it falls back to the nearest available frequency for EDP calculations.

5) Save feature importances (optional, trainer or evaluator)
```bash
# Trainer: save CSV/JSON/Markdown + PNG bar plot
python -m tools.ml_prediction.train_baseline \
  --dataset tools/ml_prediction/datasets/all_freq.csv \
  --model-out tools/ml_prediction/models/rf_all_freq.joblib \
  --save-fi-dir tools/ml_prediction/results/fi_baseline \
  --fi-top-n 30 \
  --fi-tag "rf_all_freq_random_20"

# Evaluator: include split context in saved metadata
python -m tools.ml_prediction.evaluate \
  --dataset tools/ml_prediction/datasets/all_freq.csv \
  --labels tools/ml_prediction/labels.json \
  --split gpu --holdout-gpus H100 \
  --save-fi-dir tools/ml_prediction/results/fi_eval_gpu_h100 \
  --fi-top-n 30 \
  --fi-tag "rf_eval_loo_gpu"
```
Outputs in the chosen directory:
- `feature_importances.csv` — full ranking (feature, importance, rank)
- `feature_importances.json` — importances + run context
- `feature_importances.md` — top‑N table with context
- `feature_importances_top.png` — horizontal bar chart (top‑N)

## Feature Importances

Both the baseline trainer and evaluator report feature importances from the fitted RandomForest model:
- Importances are aggregated back to the original feature names (one‑hot encoded categoricals are summed per source column).
- This helps identify which extracted features most influence the predicted EDP‑optimal frequency.

How to run and view importances:
```bash
# During baseline training (prints top features)
python -m tools.ml_prediction.train_baseline \
  --dataset tools/ml_prediction/datasets/all_freq.csv \
  --model-out tools/ml_prediction/models/rf_all_freq.joblib

# During evaluation with EDP gap (also prints importances before metrics)
python -m tools.ml_prediction.evaluate \
  --dataset tools/ml_prediction/datasets/all_freq.csv \
  --labels tools/ml_prediction/labels.json \
  --split random --holdout 0.2
```
Example output snippet:
```
=== Top Feature Importances (RF, aggregated) ===
probe_frequency_mhz              0.2315
power_mean                       0.1142
gputl_slope                      0.0719
gpu_type                         0.0705
smclk_norm_hal_max               0.0574
...
```

Notes:
- Importances are model‑based (Gini) summary values; they are not causal.
- For more robust attribution (especially with correlated features), consider permutation importance or SHAP in future iterations.

### Save Importances to Disk

Both commands can persist feature importances to a directory and generate a quick visual:
```bash
# Trainer: save CSV/JSON/Markdown + PNG bar plot
python -m tools.ml_prediction.train_baseline \
  --dataset tools/ml_prediction/datasets/all_freq.csv \
  --model-out tools/ml_prediction/models/rf_all_freq.joblib \
  --save-fi-dir tools/ml_prediction/results/fi_baseline \
  --fi-top-n 30 \
  --fi-tag "rf_all_freq_random_20"

# Evaluator: include split context in saved metadata
python -m tools.ml_prediction.evaluate \
  --dataset tools/ml_prediction/datasets/all_freq.csv \
  --labels tools/ml_prediction/labels.json \
  --split workload --holdout-workloads llama \
  --save-fi-dir tools/ml_prediction/results/fi_eval_workload_llama \
  --fi-top-n 30 \
  --fi-tag "rf_eval_loo_wl"
```
Outputs in the chosen directory:
- `feature_importances.csv`: full ranking (feature, importance, rank)
- `feature_importances.json`: importances + run context
- `feature_importances.md`: top‑N table with brief context
- `feature_importances_top.png`: horizontal bar chart (top‑N)

Tip: Commit the CSV/JSON to track changes across runs; the PNG/MD help with quick reviews.

## Evaluation With EDP Gap

Use the evaluation script for cross‑workload/GPU splits and EDP gap reporting (requires `--policy all-freq` dataset):
```bash
# Random 20% holdout
python -m tools.ml_prediction.evaluate \
  --dataset tools/ml_prediction/datasets/all_freq.csv \
  --labels tools/ml_prediction/labels.json \
  --split random --holdout 0.2

# Leave-one-workload-out
python -m tools.ml_prediction.evaluate \
  --dataset tools/ml_prediction/datasets/all_freq.csv \
  --labels tools/ml_prediction/labels.json \
  --split workload --holdout-workloads llama

# Leave-one-GPU-out
python -m tools.ml_prediction.evaluate \
  --dataset tools/ml_prediction/datasets/all_freq.csv \
  --labels tools/ml_prediction/labels.json \
  --split gpu --holdout-gpus H100
```
The evaluator prints frequency error and EDP gap (predicted vs optimal EDP = energy × time). It avoids split leakage (e.g., excludes `workload` from features for workload holdout). If an exact (gpu, workload, freq) mapping is missing, it falls back to the nearest available frequency for EDP calculations.

## Roadmap / TODOs

- [x] Build labels from optimizer (`build_labels.py`)
- [x] Build dataset with `max-only`, `tri-point`, `all-freq` (`build_dataset.py`)
- [x] Baseline RF training (`train_baseline.py`) with quick metrics
- [x] Enrich features using `feature_extractor.py` (stats, trends, ratios, HAL‑normalized clocks)
- [x] Evaluation script: cross‑workload and cross‑GPU splits + EDP gap
- [x] Internal refactor: shared run‑filename parser + unified timing loader
- [ ] Integrate nvidia‑smi profile parsing path in ML tools
- [ ] Confidence‑gated few‑shot (tri‑point) inference
- [ ] Hyperparameter tuning / alternative models (XGB/LightGBM/NN) with holdout‑split tracking
- [ ] Per‑workload/GPU percentiles, worst‑case reporting, and feature importances
- [ ] Inference CLI for saved models (single‑run features → frequency)
- [ ] Optional Parquet I/O with documented deps (e.g., `pyarrow`)

## Notes

- Focus on EDP gap (energy × time), not only raw MHz error — especially for holdout scenarios where curves can be flat near the optimum.
- Probe policies: `max-only` (fast sanity), `all-freq` (training), `tri-point` (few‑shot) — use `all-freq` to train.
- Energy estimate: prefers `TOTEC` deltas only when positive; otherwise mean(POWER) × duration. Duration falls back to sample count × sampling interval if timing is missing.

## Known Limitations

- Profiles are parsed from DCGMI outputs; direct nvidia‑smi parsing is not yet integrated in these ML tools.
- EDP gap reporting depends on per‑frequency rows (`all-freq` datasets). With sparse data, nearest‑frequency fallback may still omit some rows.
- Reading/writing Parquet requires optional dependencies (e.g., `pyarrow`). CSV works out of the box.
- HAL‑normalized features are best‑effort and skipped when `hardware.gpu_info` is unavailable for a GPU type.

## Recent Changes

- Consolidated run filename parsing into a shared helper and reused the canonical `timing_summary.log` loader to avoid duplication across tools.
- Added trend slopes and HAL‑normalized clock features to the feature extractor; datasets now include these automatically when rebuilt.
- Evaluator now reports EDP gap with nearest‑frequency fallback when exact keys are missing.
- Added aggregated RandomForest feature importances with options to persist CSV/JSON/Markdown and a PNG bar plot via `--save-fi-dir` in trainer and evaluator.
