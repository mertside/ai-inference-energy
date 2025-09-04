# ML Prediction Tools — User Guide

This guide explains how to train and evaluate ML models that predict EDP‑optimal GPU frequency from short profiling runs. It builds on profiling results under `sample-collection-scripts/` and the analysis pipeline in `tools/analysis/`.

## Components

- `profile_reader.py`: Robust DCGMI/nvidia‑smi parsing, warm‑run aggregation, optional IQR outlier filtering.
- `feature_extractor.py`: Converts profiles to features (POWER, GPUTL, MCUTL, SMCLK, MMCLK, TMPTR, SMACT, DRAMA) plus ratios and context.
- `label_builder.py`: Generates ground‑truth labels (EDP & ED²P optimal) via `tools/analysis/edp_optimizer.py`.
- `dataset_builder.py`: Builds datasets from probe policies: `max-only`, `tri-point`, `all-freq`.
- `models/random_forest_predictor.py`: Baseline RF classifier with frequency snapping.
- `evaluate.py`: Cross‑workload/GPU evaluation with EDP gap reporting.

## Environment Setup

Prerequisites (local environment):
- Python 3.8+
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
```bash
 python -m tools.ml_prediction.train_baseline \
  --dataset tools/ml_prediction/datasets/max_only.csv \
  --model-out tools/ml_prediction/models/rf_max_only.joblib
```
Alternative (direct script):
```bash
python tools/ml_prediction/train_baseline.py \
  --dataset tools/ml_prediction/datasets/max_only.csv \
  --model-out tools/ml_prediction/models/rf_max_only.joblib
```

Expected output (random split, all-freq):
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
- Include more informative features (e.g., probe_frequency_mhz, normalized ratio, GPUTL/MCUTL/SMCLK) by switching to the richer `feature_extractor.py` path.
- Evaluate with cross‑workload and cross‑GPU splits to check generalization.

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
The evaluator prints frequency error and EDP gap (predicted vs optimal EDP = energy × time). It avoids split leakage (e.g., excludes `workload` from features for workload holdout).

## Roadmap / TODOs

- [x] Build labels from optimizer (`build_labels.py`)
- [x] Build dataset with `max-only`, `tri-point`, `all-freq` (`build_dataset.py`)
- [x] Baseline RF training (`train_baseline.py`) with quick metrics
- [ ] Enrich features using `feature_extractor.py` (POWER/GPUTL/MCUTL/SMCLK/TMPTR/SMACT/DRAMA + trends/ratios)
- [ ] Evaluation script: cross‑workload and cross‑GPU splits + EDP gap
- [ ] Few‑shot inference (tri‑point) gated by confidence
- [ ] Advanced models (XGB/NN/ensembles) and ablations

## Notes

- Focus on EDP gap (energy × time), not only raw MHz error — especially for holdout scenarios where curves can be flat near the optimum.
- Probe policies: `max-only` (fast sanity), `all-freq` (training), `tri-point` (few‑shot) — use `all-freq` to train.
- Energy estimate: prefers `TOTEC` deltas only when positive; otherwise mean(POWER) × duration. Duration falls back to sample count × sampling interval if timing is missing.
