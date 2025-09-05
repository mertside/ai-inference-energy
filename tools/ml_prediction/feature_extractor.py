"""
Feature Extractor

Responsibilities:
- Convert aggregated warm-run profiles into model-ready features
- Provide statistical descriptors (mean/std/p95/min/max), ratios, and context
- Include short trend (slope) features for selected metrics (e.g., POWER, TMPTR, GPUTL)
- Optionally normalize clocks using HAL (GPU max/core clock; memory clock) when available
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

try:  # pragma: no cover - optional during scaffold
    import numpy as np
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore
    np = None  # type: ignore

try:  # pragma: no cover - optional; used for normalization
    from hardware.gpu_info import GPUSpecifications
except Exception:  # pragma: no cover
    GPUSpecifications = None  # type: ignore


class ProfileFeatureExtractor:
    """Extract features from a single aggregated profile DataFrame and metadata.

    Expected df columns (canonical, when available):
    POWER, GPUTL, MCUTL, SMCLK, MMCLK, TMPTR, SMACT, DRAMA
    """

    def __init__(self, trend_window: int = 30) -> None:
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
        # Compute linear slope over this many tail samples (if available)
        self.trend_window = max(int(trend_window), 5)
        self.metrics_with_trend = ["POWER", "TMPTR", "GPUTL"]

    def _add_basic_stats(self, feats: Dict[str, Any], series, name: str) -> None:
        feats[f"{name}_mean"] = float(series.mean())
        feats[f"{name}_std"] = float(series.std())
        feats[f"{name}_min"] = float(series.min())
        feats[f"{name}_max"] = float(series.max())
        try:
            feats[f"{name}_p95"] = float(series.quantile(0.95))
        except Exception:
            feats[f"{name}_p95"] = float(series.mean())

    def _add_trend(self, feats: Dict[str, Any], series, name: str, sampling_interval_ms: float) -> None:
        """Add linear slope trend for the last N samples of `series`.

        Slope is expressed per-second when sampling interval is provided (>0),
        otherwise per-sample. Uses simple linear regression (np.polyfit).
        """
        try:
            values = series.dropna().values
            if values.size < 5:
                return
            w = min(values.size, self.trend_window)
            y = values[-w:]
            x = np.arange(w, dtype=float)
            # slope per sample
            slope_per_sample = float(np.polyfit(x, y, 1)[0])
            if sampling_interval_ms and sampling_interval_ms > 0:
                slope_per_s = slope_per_sample / (sampling_interval_ms / 1000.0)
            else:
                slope_per_s = slope_per_sample
            feats[f"{name}_slope"] = float(slope_per_s)
        except Exception:
            # Keep features sparse if trend fails
            pass

    def _try_hal_norms(self, gpu_type: Optional[str]) -> Dict[str, Optional[float]]:
        """Return normalization references from HAL for the given GPU type.

        Returns dict with keys: core_max_mhz, mem_mhz (may be None if unavailable).
        """
        core_max = None
        mem_mhz = None
        if not gpu_type or GPUSpecifications is None:
            return {"core_max_mhz": core_max, "mem_mhz": mem_mhz}
        try:
            hal = GPUSpecifications(str(gpu_type).upper())
            core_min, core_max = hal.get_frequency_range()
            mem_mhz = hal.get_memory_specification().frequency_mhz
        except Exception:
            pass
        return {"core_max_mhz": float(core_max) if core_max else None, "mem_mhz": float(mem_mhz) if mem_mhz else None}

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

        # Trend features (slopes)
        sampling_interval_ms = float(metadata.get("sampling_interval_ms", 50))
        for m in self.metrics_with_trend:
            if m in df.columns:
                self._add_trend(feats, df[m], m.lower(), sampling_interval_ms)

        # Derived ratios
        try:
            if "MCUTL" in df.columns and "GPUTL" in df.columns:
                gputl = max(float(df["GPUTL"].mean()), 1e-6)
                feats["mem_to_gpu_ratio"] = float(df["MCUTL"].mean()) / gputl
            if "POWER" in df.columns and "GPUTL" in df.columns:
                power = max(float(df["POWER"].mean()), 1e-6)
                feats["power_efficiency"] = float(df["GPUTL"].mean()) / power
            # Utilization and clock relationships
            if "SMCLK" in df.columns and "MMCLK" in df.columns:
                sm = max(float(df["SMCLK"].mean()), 1e-6)
                mm = max(float(df["MMCLK"].mean()), 1e-6)
                feats["sm_to_mem_clock_ratio"] = sm / mm
            if "GPUTL" in df.columns and "SMCLK" in df.columns:
                feats["gputl_per_mhz"] = float(df["GPUTL"].mean()) / max(float(df["SMCLK"].mean()), 1e-6)
            if "MCUTL" in df.columns and "GPUTL" in df.columns:
                feats["utilization_balance"] = float(df["GPUTL"].mean()) - float(df["MCUTL"].mean())
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

        # Normalized clocks via HAL (best-effort)
        hal_norm = self._try_hal_norms(metadata.get("gpu_type"))
        try:
            if hal_norm.get("core_max_mhz") and "SMCLK" in df.columns:
                feats["smclk_norm_hal_max"] = float(df["SMCLK"].mean()) / float(hal_norm["core_max_mhz"])  # type: ignore[index]
            if hal_norm.get("mem_mhz") and "MMCLK" in df.columns:
                feats["mmclk_norm_hal_mem"] = float(df["MMCLK"].mean()) / float(hal_norm["mem_mhz"])  # type: ignore[index]
        except Exception:
            pass

        return feats
