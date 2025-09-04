"""
Dataset Builder (Scaffold)

Responsibilities:
- Assemble supervised datasets from probe runs and ground-truth labels
- Support probe policies: `max-only` (default) and `tri-point`
- Ensure no leakage from full sweep: features must come only from chosen probe runs

Outputs:
- Parquet/CSV with features + {gpu, workload, probe_policy, label_edp, label_ed2p, performance_threshold}

This is a scaffold. Implement functionality per the plan in phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

try:  # pragma: no cover - optional during scaffold
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


ProbePolicy = Literal["max-only", "tri-point"]


@dataclass
class DatasetRow:
    features: Dict[str, Any]
    gpu: str
    workload: str
    probe_policy: ProbePolicy
    label_edp: int
    label_ed2p: Optional[int]
    performance_threshold: float


class DatasetBuilder:
    def __init__(self, results_base_dir: Path, labels_file: Path) -> None:
        self.results_base_dir = results_base_dir
        self.labels_file = labels_file
        # TODO: load labels JSON into memory

    def _select_probe_runs(self, result_dir: Path, policy: ProbePolicy) -> List[Path]:
        """Return list of run profile files matching the selected probe policy.

        TODO:
        - Implement logic to pick max-only or tri-point frequencies
        - Use hardware.gpu_info to query valid ranges if needed
        """
        raise NotImplementedError("TODO: implement probe run selection")

    def _extract_features_for_runs(self, run_files: List[Path], context: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate/concatenate features from selected run files.

        TODO:
        - Use ProfileFeatureExtractor to build features per run
        - For tri-point, decide concatenation vs pooling (documented in context)
        """
        raise NotImplementedError("TODO: implement feature extraction for runs")

    def build(self, output_file: Path, policy: ProbePolicy = "max-only") -> Path:
        """Build dataset and save to output_file.

        TODO:
        - Iterate result directories
        - Load label for gpu/workload from labels_file
        - Select runs per policy; extract features
        - Write dataset (CSV/Parquet); return path
        """
        raise NotImplementedError("TODO: implement dataset builder")
