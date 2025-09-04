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

## Getting Started

Implement in order:
1) `profile_reader.py` → 2) `label_builder.py` → 3) `feature_extractor.py` → 4) `dataset_builder.py` → 5) `models/random_forest_predictor.py`.

Each file provides function/class skeletons, expected inputs/outputs, and TODO blocks.
