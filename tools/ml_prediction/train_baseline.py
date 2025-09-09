#!/usr/bin/env python3
"""
Train a baseline RandomForest model on the built dataset and report simple metrics.

Metrics (initial):
- Frequency error (MHz): |pred - label_edp|
- % within 30 MHz of optimal

Usage:
  python -m tools.ml_prediction.train_baseline \
    --dataset tools/ml_prediction/datasets/max_only.csv \
    --model-out tools/ml_prediction/models/rf_max_only.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Support both package and direct script execution
try:
    from .models.random_forest_predictor import PredictorConfig, RandomForestFrequencyPredictor  # type: ignore
except Exception:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.ml_prediction.models.random_forest_predictor import PredictorConfig, RandomForestFrequencyPredictor  # type: ignore


def eval_frequency_error(y_true: np.ndarray, y_pred: List[int]) -> None:
    y_pred_arr = np.array(y_pred)
    err = np.abs(y_pred_arr - y_true)
    within_30 = np.mean(err <= 30) * 100.0
    within_60 = np.mean(err <= 60) * 100.0
    print(f"Frequency error (median): {np.median(err):.1f} MHz")
    print(f"Frequency error (mean):   {np.mean(err):.1f} MHz")
    print(f"Within 30 MHz: {within_30:.1f}%  |  Within 60 MHz: {within_60:.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline RF model and evaluate on holdout split")
    parser.add_argument("--dataset", required=True, help="CSV/Parquet dataset path")
    parser.add_argument("--model-out", required=True, help="Output path for joblib model")
    parser.add_argument("--holdout", type=float, default=0.2, help="Holdout fraction [0,1]")
    args = parser.parse_args()

    ds_path = Path(args.dataset).resolve()
    if ds_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(ds_path)
    else:
        df = pd.read_csv(ds_path)

    # Basic split (random). Future: cross-GPU/workload splits
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    n_hold = int(n * args.holdout)
    df_train = df.iloc[n_hold:]
    df_test = df.iloc[:n_hold]

    y_train = df_train["label_edp"].values
    y_test = df_test["label_edp"].values

    drop_cols = ["label_edp", "label_ed2p"]
    X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

    # Infer features: numeric auto; categorical includes gpu_type and probe_policy
    cfg = PredictorConfig(
        numeric_features=[],  # auto-infer numeric
        categorical_features=[c for c in ["gpu_type", "probe_policy"] if c in X_train.columns],
        gpu_type_field="gpu_type",
    )
    model = RandomForestFrequencyPredictor(cfg)
    model.fit(X_train, y_train)
    # Report feature importances (aggregated by original feature)
    try:
        fi = model.feature_importances()
        if fi:
            top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
            print("\n=== Top Feature Importances (RF, aggregated) ===")
            for name, val in top:
                print(f"{name:32s} {val:.4f}")
    except Exception:
        pass

    # Evaluate
    preds = model.predict(X_test)
    print("\n=== Evaluation (label_edp) ===")
    eval_frequency_error(y_test, preds)

    # Save model
    out_path = Path(args.model_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"\nModel saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
