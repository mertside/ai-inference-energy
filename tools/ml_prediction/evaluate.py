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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models.random_forest_predictor import PredictorConfig, RandomForestFrequencyPredictor


def build_maps(df: pd.DataFrame) -> Tuple[Dict[Tuple[str, str, int], float], Dict[Tuple[str, str, int], float]]:
    """Maps for (gpu, workload, freq) -> energy_estimate_j and duration_seconds (averaged)."""
    emap: Dict[Tuple[str, str, int], float] = {}
    tmap: Dict[Tuple[str, str, int], float] = {}
    if "probe_frequency_mhz" not in df.columns:
        return emap, tmap
    group_cols = ["gpu", "workload", "probe_frequency_mhz"]
    for (gpu, wl, freq), g in df.groupby(group_cols, dropna=True):
        key = (str(gpu), str(wl), int(freq))
        if "energy_estimate_j" in g.columns:
            emap[key] = float(g["energy_estimate_j"].mean())
        if "duration_seconds" in g.columns:
            tmap[key] = float(g["duration_seconds"].mean())
    return emap, tmap


def nearest_key(keys: List[int], target: int) -> Optional[int]:
    return min(keys, key=lambda k: abs(k - target)) if keys else None


def evaluate(
    df: pd.DataFrame, labels: Dict[Tuple[str, str], dict], split: str, holdout: float, holdout_workloads: List[str], holdout_gpus: List[str]
) -> None:
    # Build energy map for EDP gap calculations
    energy_map, time_map = build_maps(df)

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

    # Choose categorical features conservatively to avoid leakage in LO(O) splits
    cat_feats: List[str] = []
    if "gpu_type" in X_train.columns:
        cat_feats.append("gpu_type")
    if "probe_policy" in X_train.columns:
        cat_feats.append("probe_policy")
    if split == "random":
        # Random split can include workload/gpu safely
        for c in ["gpu", "workload"]:
            if c in X_train.columns:
                cat_feats.append(c)
    elif split == "workload":
        # Exclude 'workload' to test generalization to unseen workloads
        if "gpu" in X_train.columns:
            cat_feats.append("gpu")
    elif split == "gpu":
        # Exclude 'gpu' to test generalization to unseen GPUs
        if "workload" in X_train.columns:
            cat_feats.append("workload")

    cfg = PredictorConfig(
        numeric_features=[],
        categorical_features=cat_feats,
        gpu_type_field="gpu_type" if "gpu_type" in X_train.columns else ("gpu" if "gpu" in X_train.columns else "gpu_type"),
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
    gaps = []  # EDP gap (%)
    savings_deltas = []  # Savings delta (pp) vs optimal
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
        # Energies and durations
        key_opt = (gpu, wl, opt_freq)
        key_pred = (gpu, wl, pred_freq)
        key_max = (gpu, wl, max_freq)

        e_opt = energy_map.get(key_opt)
        e_pred = energy_map.get(key_pred)
        e_max = energy_map.get(key_max)
        t_opt = time_map.get(key_opt)
        t_pred = time_map.get(key_pred)
        t_max = time_map.get(key_max)

        # Fallback: nearest frequency in available mapping if exact key missing
        if any(x is None for x in [e_opt, e_pred, e_max, t_opt, t_pred, t_max]):
            freqs_available = [f for (g2, w2, f) in energy_map.keys() if g2 == gpu and w2 == wl]
            if freqs_available:
                if e_opt is None or t_opt is None:
                    f = nearest_key(freqs_available, opt_freq)
                    if f is not None:
                        e_opt = energy_map.get((gpu, wl, f))
                        t_opt = time_map.get((gpu, wl, f))
                if e_pred is None or t_pred is None:
                    f = nearest_key(freqs_available, pred_freq)
                    if f is not None:
                        e_pred = energy_map.get((gpu, wl, f))
                        t_pred = time_map.get((gpu, wl, f))
                if e_max is None or t_max is None:
                    f = nearest_key(freqs_available, max_freq)
                    if f is not None:
                        e_max = energy_map.get((gpu, wl, f))
                        t_max = time_map.get((gpu, wl, f))

        if any(x is None for x in [e_opt, e_pred, e_max, t_opt, t_pred, t_max]):
            missing += 1
            continue

        # Compute EDP values (Energy × Time)
        edp_opt = float(e_opt) * float(t_opt)
        edp_pred = float(e_pred) * float(t_pred)
        edp_max = float(e_max) * float(t_max)
        if not np.isfinite(edp_opt) or not np.isfinite(edp_pred) or not np.isfinite(edp_max):
            missing += 1
            continue
        if edp_opt <= 0 or edp_max <= 0:
            missing += 1
            continue

        gap = (edp_pred - edp_opt) / edp_opt * 100.0
        gaps.append(gap)
        s_opt = (edp_max - edp_opt) / edp_max * 100.0
        s_pred = (edp_max - edp_pred) / edp_max * 100.0
        savings_deltas.append(s_opt - s_pred)  # positive means predicted saves less than optimal

    if gaps:
        print(f"EDP gap vs optimal: median {np.median(gaps):.1f}% | mean {np.mean(gaps):.1f}%")
        print(f"Savings delta vs optimal: median {np.median(savings_deltas):.1f} pp | mean {np.mean(savings_deltas):.1f} pp")
    else:
        print("Insufficient mapping to compute EDP gap (ensure dataset built with --policy all-freq)")
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
