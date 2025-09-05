"""ML Prediction package (scaffold).

Modules:
- profile_reader: Robust parsing and aggregation of profiling data
- label_builder: Ground-truth label generation via EDP/ED²P optimizer
- feature_extractor: Feature engineering from aggregated profiles
- dataset_builder: Dataset assembly from probe runs + labels
- models.random_forest_predictor: Baseline classifier scaffold
"""

__all__ = [
    "profile_reader",
    "label_builder",
    "feature_extractor",
    "dataset_builder",
    "models",
]
