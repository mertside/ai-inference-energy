"""
Random Forest Predictor

Responsibilities:
- Baseline discrete classifier to predict optimal frequency bin per GPU
- Frequency snapping to nearest valid clock via hardware.gpu_info
- Confidence estimate (class probability / ensemble agreement)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # pragma: no cover - optional during scaffold
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception:  # pragma: no cover
    RandomForestClassifier = None  # type: ignore
    StandardScaler = None  # type: ignore
    Pipeline = None  # type: ignore
    ColumnTransformer = None  # type: ignore
    OneHotEncoder = None  # type: ignore

try:  # pragma: no cover - use HAL to snap to legal clocks
    from hardware.gpu_info import GPUSpecifications
except Exception:  # pragma: no cover
    GPUSpecifications = None  # type: ignore


@dataclass
class PredictorConfig:
    numeric_features: Sequence[str]
    categorical_features: Sequence[str]
    gpu_type_field: str = "gpu_type"


class RandomForestFrequencyPredictor:
    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self.model = None  # type: ignore
        self.preprocessor = None  # type: ignore
        self.frequencies_per_gpu: Dict[str, List[int]] = {}

    def fit(self, X: "pd.DataFrame", y: "np.ndarray") -> None:
        """Train the RF classifier with preprocessing (scaler + one‑hot).

        Automatically infers numeric/categorical features if not provided in
        the config. Stores a `Pipeline(preprocessor -> RandomForestClassifier)`.
        """
        if RandomForestClassifier is None:
            raise RuntimeError("scikit-learn is required to train the model")

        # Infer feature sets if not provided
        numeric_features = (
            list(self.config.numeric_features) if self.config.numeric_features else list(X.select_dtypes(include=["number"]).columns)
        )
        categorical_features = (
            list(self.config.categorical_features)
            if self.config.categorical_features
            else [c for c in X.columns if c not in numeric_features]
        )

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        self.model = Pipeline(steps=[("preprocessor", self.preprocessor), ("clf", clf)])
        self.model.fit(X, y)

    def predict(self, X: "pd.DataFrame") -> List[int]:
        """Predict and snap to the nearest legal frequency per row."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        preds = self.model.predict(X)
        out: List[int] = []
        for i, p in enumerate(preds):
            try:
                gpu_type = str(X.iloc[i][self.config.gpu_type_field]).upper()
            except Exception:
                gpu_type = "A100"
            out.append(self.snap_frequency(gpu_type, int(p)))
        return out

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict from a single feature dict.

        Returns a dict with keys: `predicted_frequency`, `snapped_frequency`,
        and `confidence` (max class probability, if available).
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        row = pd.DataFrame([features])
        gpu_type = str(row.iloc[0].get(self.config.gpu_type_field, "A100")).upper()
        pred = int(self.model.predict(row)[0])
        snapped = self.snap_frequency(gpu_type, pred)
        conf = None
        try:
            # Get probability of the predicted class
            proba = self.model.named_steps["clf"].predict_proba(self.model.named_steps["preprocessor"].transform(row))
            conf = float(np.max(proba))
        except Exception:
            pass
        return {"predicted_frequency": pred, "snapped_frequency": snapped, "confidence": conf}

    def save(self, path: Path) -> None:
        """Save model artifacts using joblib."""
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        """Load model artifacts using joblib."""
        import joblib

        self.model = joblib.load(path)

    @staticmethod
    def snap_frequency(gpu_type: str, target: int) -> int:
        """Snap target to nearest supported frequency for gpu_type using HAL."""
        if GPUSpecifications is None:  # pragma: no cover
            return target
        try:
            gpu_info = GPUSpecifications(gpu_type)
            legal = gpu_info.get_available_frequencies()
            return min(legal, key=lambda x: abs(x - target)) if legal else target
        except Exception:
            return target
