"""
Feature Extractor

Responsibilities:
- Convert aggregated warm-run profiles into model-ready features
- Provide statistical descriptors (mean/std/p95/min/max), ratios, and context
- Optionally extend with short trend (slope) features for selected metrics
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

try:  # pragma: no cover - optional during scaffold
    import numpy as np
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore
    np = None  # type: ignore


class ProfileFeatureExtractor:
    """Extract features from a single aggregated profile DataFrame and metadata.

    Expected df columns (canonical, when available):
    POWER, GPUTL, MCUTL, SMCLK, MMCLK, TMPTR, SMACT, DRAMA
    """

    def __init__(self) -> None:
        # Extension point: make metrics/percentiles configurable if needed
        self.metrics = [
            "POWER",
            "GPUTL",
            "MCUTL",
            "SMCLK",
            "MMCLK",
            "TMPTR",
            "SMACT",
            "DRAMA",
        ]

    def _add_basic_stats(self, feats: Dict[str, Any], series, name: str) -> None:
        feats[f"{name}_mean"] = float(series.mean())
        feats[f"{name}_std"] = float(series.std())
        feats[f"{name}_min"] = float(series.min())
        feats[f"{name}_max"] = float(series.max())
        try:
            feats[f"{name}_p95"] = float(series.quantile(0.95))
        except Exception:
            feats[f"{name}_p95"] = float(series.mean())

    def extract(self, df: "pd.DataFrame", metadata: Mapping[str, Any]) -> Dict[str, Any]:
        """Extract features from a single probe's DataFrame and metadata.

        Returns: dict of feature_name -> value
        """
        feats: Dict[str, Any] = {}
        if df is None:
            return feats

        # Basic statistics per metric
        for m in self.metrics:
            if m in df.columns:
                self._add_basic_stats(feats, df[m], m.lower())

        # Derived ratios
        try:
            if "MCUTL" in df.columns and "GPUTL" in df.columns:
                gputl = max(float(df["GPUTL"].mean()), 1e-6)
                feats["mem_to_gpu_ratio"] = float(df["MCUTL"].mean()) / gputl
            if "POWER" in df.columns and "GPUTL" in df.columns:
                power = max(float(df["POWER"].mean()), 1e-6)
                feats["power_efficiency"] = float(df["GPUTL"].mean()) / power
        except Exception:
            # Keep features sparse if ratios fail
            pass

        # Context features (subset; add more as needed)
        feats.update(
            {
                "gpu_type": metadata.get("gpu_type"),
                "sampling_interval_ms": metadata.get("sampling_interval_ms", 50),
                "probe_policy": metadata.get("probe_policy"),
            }
        )

        return feats
