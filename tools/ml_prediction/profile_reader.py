"""
Profile Reader (Scaffold)

Responsibilities:
- Robustly parse DCGMI/nvidia-smi profiling outputs from sample-collection results
- Aggregate warm runs only (exclude first run per frequency)
- Apply optional IQR outlier filtering for timings/power/energy
- Return per-frequency aggregates suitable for feature extraction and labeling

Implementation notes:
- DCGMI output is a whitespace table with header rows; some fields may be "N/A"
- Prefer measured timing from timing_summary.log; otherwise fall back where possible
- Energy may be computed as avg_power * duration_seconds for the run

This is a scaffold. Implement functionality per the plan in phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Optional imports (defer heavy deps until implementation)
try:  # pragma: no cover - optional during scaffold
    import numpy as np
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore
    np = None  # type: ignore


@dataclass
class RunAggregate:
    """Aggregated metrics for a single frequency (warm runs only)."""

    frequency_mhz: int
    avg_power_w: float
    avg_timing_s: float
    avg_energy_j: float
    power_std_w: Optional[float] = None
    timing_std_s: Optional[float] = None
    run_count: int = 0


def parse_dcgmi_profile(file_path: Path) -> "pd.DataFrame":
    """Parse a DCGMI profile file into a DataFrame.

    TODO:
    - Implement whitespace-table parsing with header detection
    - Map columns to canonical names (POWER, GPUTL, MCUTL, SMCLK, MMCLK, TMPTR, etc.)
    - Handle N/A rows robustly

    Returns: pd.DataFrame with canonical columns
    """
    raise NotImplementedError("TODO: implement DCGMI parser")


def load_timing_summary(dir_path: Path) -> Mapping[str, Tuple[int, float]]:
    """Load timing_summary.log mapping run_id -> (frequency_mhz, duration_seconds).

    TODO:
    - Parse CSV lines skipping comments
    - Return dict keyed by run_id
    """
    raise NotImplementedError("TODO: implement timing summary loader")


def list_profile_files(dir_path: Path) -> List[Path]:
    """List per-run profile files in a results directory.

    Matches: run_*_profile.csv
    """
    return sorted(dir_path.glob("run_*_profile.csv"))


def aggregate_warm_runs(
    dir_path: Path,
    iqr_filter: bool = True,
    sampling_interval_s: float = 0.05,
) -> Dict[int, RunAggregate]:
    """Aggregate warm runs per frequency in a results directory.

    TODO:
    - For each frequency, order runs, exclude first run (cold), collect avg power
    - Compute duration from timing_summary.log per run_id
    - Optionally apply IQR-based outlier filtering to power/energy
    - Compute RunAggregate per frequency
    """
    raise NotImplementedError("TODO: implement warm-run aggregation")


def read_run_profiles(dir_path: Path) -> Dict[int, "pd.DataFrame"]:
    """Return raw per-run profile DataFrames keyed by frequency.

    TODO:
    - Parse each profile
    - Group by frequency from file name (regex)
    - Return mapping for downstream feature extraction
    """
    raise NotImplementedError("TODO: implement raw profile reader")
