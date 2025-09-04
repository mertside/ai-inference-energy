# ML Prediction Package (Scaffold)

This package contains scaffolding for the ML-based frequency prediction workflow.
Files are intentionally light-weight and include clear TODOs to implement in phases,
aligned with `documentation/ML_FREQUENCY_PREDICTION_PLAN.md` (Revised Plan Addendum v1.1).

## Modules

- `profile_reader.py`: Robust DCGMI/nvidia-smi parsing, warm-run aggregation, IQR outlier filtering.
- `label_builder.py`: Wraps `tools/analysis/edp_optimizer.py` to export ground-truth labels.
- `feature_extractor.py`: Converts aggregated run profiles into model-ready features.
- `dataset_builder.py`: Builds training datasets from probe runs + labels with probe policies.
- `models/random_forest_predictor.py`: Baseline classifier scaffold with frequency snapping.

## Quick Start (End‑to‑End)

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

Important: run from the repository root (so `tools` is on the Python path), or use the direct script form shown below.

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
Alternative (direct script):
```bash
python tools/ml_prediction/build_dataset.py \
  --results-dir sample-collection-scripts \
  --labels tools/ml_prediction/labels.json \
  --output tools/ml_prediction/datasets/max_only.csv \
  --policy max-only
```

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

Expected output (example, with max-only):
```
=== Evaluation (label_edp) ===
Frequency error (median): XX.X MHz
Frequency error (mean):   XX.X MHz
Within 30 MHz: YY.Y%  |  Within 60 MHz: ZZ.Z%
Model saved to tools/ml_prediction/models/rf_max_only.joblib
```

Tips to improve baseline results:
- Prefer `--policy all-freq` to generate many more training samples (one per frequency),
  which typically improves learning substantially over `max-only` (12 samples).
- Include more informative features (e.g., probe_frequency_mhz, normalized ratio, GPUTL/MCUTL/SMCLK) by switching to the richer `feature_extractor.py` path.
- Evaluate with cross‑workload and cross‑GPU splits to check generalization.

## Roadmap / TODOs

- [x] Build labels from optimizer (`build_labels.py`)
- [x] Build dataset with `max-only`, `tri-point`, `all-freq` (`build_dataset.py`)
- [x] Baseline RF training (`train_baseline.py`) with quick metrics
- [ ] Enrich features using `feature_extractor.py` (POWER/GPUTL/MCUTL/SMCLK/TMPTR/SMACT/DRAMA + trends/ratios)
- [ ] Evaluation script: cross‑workload and cross‑GPU splits + EDP gap
- [ ] Few‑shot inference (tri‑point) gated by confidence
- [ ] Advanced models (XGB/NN/ensembles) and ablations

Notes:
- This baseline uses a lightweight feature set (mean power, duration, energy estimate, and context). You can switch to the richer feature extractor (`feature_extractor.py`) later for better accuracy.
- Probe policies: `max-only` (default) or `tri-point` (3 runs pooled). `tri-point` is basic initially and can be expanded to concatenated feature blocks.

## Implementation Order (Plan)

Implement in order:
1) `profile_reader.py` → 2) `label_builder.py` → 3) `feature_extractor.py` → 4) `dataset_builder.py` → 5) `models/random_forest_predictor.py`.

Each file provides function/class skeletons, expected inputs/outputs, and TODO blocks.
