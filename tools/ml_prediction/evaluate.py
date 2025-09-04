#!/usr/bin/env python3
"""
Evaluation script with cross-workload/GPU splits and EDP gap reporting.

Usage examples (run from repo root):
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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .models.random_forest_predictor import PredictorConfig, RandomForestFrequencyPredictor


def build_energy_map(df: pd.DataFrame) -> Dict[Tuple[str, str, int], float]:
    """Map (gpu, workload, frequency_mhz) -> energy_estimate_j from dataset."""
    emap: Dict[Tuple[str, str, int], float] = {}
    if "probe_frequency_mhz" not in df.columns:
        return emap
    for (gpu, wl, freq), g in df.groupby(["gpu", "workload", "probe_frequency_mhz"], dropna=True):
        energy = float(g["energy_estimate_j"].mean()) if "energy_estimate_j" in g.columns else np.nan
        emap[(str(gpu), str(wl), int(freq))] = energy
    return emap


def evaluate(
    df: pd.DataFrame, labels: Dict[Tuple[str, str], dict], split: str, holdout: float, holdout_workloads: List[str], holdout_gpus: List[str]
) -> None:
    # Build energy map for EDP gap calculations
    energy_map = build_energy_map(df)

    # Split
    if split == "random":
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n_hold = int(len(df) * holdout)
        test_df = df.iloc[:n_hold]
        train_df = df.iloc[n_hold:]
    elif split == "workload":
        mask = df["workload"].isin([w.lower() for w in holdout_workloads])
        test_df = df[mask]
        train_df = df[~mask]
    elif split == "gpu":
        mask = df["gpu"].isin([g.upper() for g in holdout_gpus])
        test_df = df[mask]
        train_df = df[~mask]
    else:
        raise ValueError(f"Unknown split: {split}")

    # Train baseline RF
    y_train = train_df["label_edp"].values
    y_test = test_df["label_edp"].values
    drop_cols = ["label_edp", "label_ed2p"]
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    cfg = PredictorConfig(
        numeric_features=[],
        categorical_features=[c for c in ["gpu_type", "probe_policy", "gpu", "workload"] if c in X_train.columns],
        gpu_type_field="gpu_type" if "gpu_type" in X_train.columns else "gpu",
    )
    model = RandomForestFrequencyPredictor(cfg)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Frequency error metrics
    err = np.abs(np.array(preds) - y_test)
    print("\n=== Frequency Metrics (label_edp) ===")
    print(f"Median error: {np.median(err):.1f} MHz | Mean error: {np.mean(err):.1f} MHz")
    print(f"Within 30 MHz: {np.mean(err <= 30)*100:.1f}% | Within 60 MHz: {np.mean(err <= 60)*100:.1f}%")

    # EDP gap metrics
    print("\n=== EDP Gap Metrics ===")
    gaps = []
    savings_deltas = []
    missing = 0
    for i in range(len(test_df)):
        gpu = str(test_df.iloc[i]["gpu"]).upper()
        wl = str(test_df.iloc[i]["workload"]).lower()
        lbl = labels.get((gpu, wl))
        if not lbl:
            continue
        opt_freq = int(lbl.get("optimal_frequency_edp_mhz"))
        max_freq = int(lbl.get("max_frequency_mhz"))
        pred_freq = int(preds[i])
        # Energies
        e_opt = energy_map.get((gpu, wl, opt_freq))
        e_pred = energy_map.get((gpu, wl, pred_freq))
        e_max = energy_map.get((gpu, wl, max_freq))
        if e_opt is None or e_pred is None or e_max is None or any(np.isnan([e_opt, e_pred, e_max])):
            missing += 1
            continue
        gap = (e_pred - e_opt) / e_opt * 100.0
        gaps.append(gap)
        s_opt = (e_max - e_opt) / e_max * 100.0
        s_pred = (e_max - e_pred) / e_max * 100.0
        savings_deltas.append(s_opt - s_pred)  # positive means predicted saves less than optimal

    if gaps:
        print(f"EDP gap vs optimal: median {np.median(gaps):.1f}% | mean {np.mean(gaps):.1f}%")
        print(f"Savings delta vs optimal: median {np.median(savings_deltas):.1f} pp | mean {np.mean(savings_deltas):.1f} pp")
    else:
        print("Insufficient energy mapping to compute EDP gap (build dataset with --policy all-freq)")
    if missing:
        print(f"Note: {missing} test rows lacked energy mapping; ensure dataset contains one row per frequency (all-freq).")


def load_labels(path: Path) -> Dict[Tuple[str, str], dict]:
    data = json.loads(path.read_text())
    return {(str(r.get("gpu")).upper(), str(r.get("workload")).lower()): r for r in data}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate model with cross-workload/GPU splits and EDP gap")
    ap.add_argument("--dataset", required=True, help="Dataset CSV/Parquet (recommend all_freq)")
    ap.add_argument("--labels", required=True, help="labels.json path")
    ap.add_argument("--split", choices=["random", "workload", "gpu"], default="random")
    ap.add_argument("--holdout", type=float, default=0.2, help="random split holdout fraction")
    ap.add_argument("--holdout-workloads", nargs="*", default=[], help="workloads to hold out for split=workload")
    ap.add_argument("--holdout-gpus", nargs="*", default=[], help="GPUs to hold out for split=gpu")
    args = ap.parse_args()

    ds_path = Path(args.dataset).resolve()
    if ds_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(ds_path)
    else:
        df = pd.read_csv(ds_path)
    labels = load_labels(Path(args.labels).resolve())

    evaluate(df, labels, args.split, args.holdout, args.holdout_workloads, args.holdout_gpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
