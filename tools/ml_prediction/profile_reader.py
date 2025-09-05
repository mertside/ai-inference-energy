"""
Profile Reader

Responsibilities:
- Parse DCGMI/nvidia-smi profiling outputs from sample-collection results
- Aggregate warm runs only (exclude first run per frequency)
- Optional IQR outlier filtering for timings/power/energy
- Return per-frequency aggregates suitable for feature extraction and labeling

Notes:
- DCGMI output is a whitespace table with header rows; some fields may be "N/A"
- Prefer measured timing from timing_summary.log; otherwise fall back where possible
- Energy is computed as avg_power × duration_seconds for the run
"""

from __future__ import annotations

import re
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


# Shared filename pattern for per-run profiles
# Format: run_<seq>_<run_num>_freq_<MHz>_profile.csv
RUN_FILE_PATTERN = re.compile(r"run_(\d+)_(\d+)_freq_(\d+)_profile\.csv$")


def parse_run_filename(name: str) -> Optional[Tuple[int, int, int]]:
    """Parse a profile filename into (seq, run_num, frequency_mhz).

    Returns None if the name does not match the expected pattern.
    """
    m = RUN_FILE_PATTERN.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def parse_dcgmi_profile(file_path: Path) -> "pd.DataFrame":
    """Parse a DCGMI profile file into a DataFrame.

    Implements whitespace-table parsing with header detection, handles N/A
    values, and maps known DCGMI columns (e.g., POWER, GPUTL, MCUTL, SMCLK,
    MMCLK, TMPTR, etc.) when present.

    Returns: pd.DataFrame with canonical columns
    """
    if pd is None:
        raise RuntimeError("pandas is required to parse DCGMI profiles")

    header: Optional[List[str]] = None
    records: List[Dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            # Header line typically starts with '#Entity'
            if line.lstrip().startswith("#") and ("DVNAM" in line or "Entity" in line):
                # Split header tokens by whitespace
                header = [tok.strip() for tok in line.lstrip("# ").split() if tok.strip()]
                continue

            # Skip units line if present (starts with 'ID' or similar)
            if header and line.strip().startswith("ID"):
                continue

            # Data rows typically start with 'GPU'
            if header and line.startswith("GPU"):
                tokens = [tok for tok in line.split() if tok]
                # Determine how many tokens belong to the DVNAM column
                try:
                    idx_dvnam = header.index("DVNAM")
                except ValueError:
                    # Unexpected format
                    continue

                fixed_before = idx_dvnam  # tokens for columns before DVNAM
                numeric_after = len(header) - (idx_dvnam + 1)
                if len(tokens) < fixed_before + numeric_after + 1:
                    # Line too short
                    continue

                dvnam_len = len(tokens) - (fixed_before + numeric_after)
                if dvnam_len < 1:
                    # Fallback
                    dvnam_len = 1

                row_map: Dict[str, Any] = {}
                # Entity and NVIDX
                for i in range(fixed_before):
                    row_map[header[i]] = tokens[i]
                # DVNAM (may contain spaces)
                dvnam_tokens = tokens[fixed_before : fixed_before + dvnam_len]
                row_map["DVNAM"] = " ".join(dvnam_tokens)
                # Remaining numeric fields align with header after DVNAM
                numeric_tokens = tokens[fixed_before + dvnam_len :]
                for j, h in enumerate(header[idx_dvnam + 1 :]):
                    val = numeric_tokens[j] if j < len(numeric_tokens) else None
                    if val in (None, "N/A", "NA"):
                        row_map[h] = None
                    else:
                        try:
                            row_map[h] = float(val)
                        except ValueError:
                            row_map[h] = None

                records.append(row_map)

    if not records:
        # Return empty DataFrame with minimal structure
        return pd.DataFrame(columns=header or [])

    df = pd.DataFrame.from_records(records)
    return df


def load_timing_summary(dir_path: Path) -> Mapping[str, Tuple[int, float]]:
    """Load timing_summary.log mapping run_id -> (frequency_mhz, duration_seconds)."""
    timings: Dict[str, Tuple[int, float]] = {}
    timing_file = dir_path / "timing_summary.log"
    if not timing_file.exists():
        return timings

    with timing_file.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Format: run_id,frequency_mhz,duration_seconds,exit_code,status
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            run_id, freq_str, dur_str, exit_code, status = parts[:5]
            try:
                freq = int(freq_str)
                dur = float(dur_str)
            except ValueError:
                continue
            timings[run_id] = (freq, dur)
    return timings


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

    For each frequency: order runs, exclude first run (cold), compute average
    power and duration (from timing_summary.log), apply optional IQR filtering
    on energy, and return RunAggregate per frequency.
    """
    if pd is None or np is None:
        raise RuntimeError("pandas and numpy are required for aggregation")

    timing_map = load_timing_summary(dir_path)
    files = list_profile_files(dir_path)
    # Group runs per frequency
    runs_by_freq: Dict[int, List[Tuple[str, int, Path]]] = {}

    for fp in files:
        parsed = parse_run_filename(fp.name)
        if not parsed:
            continue
        seq, run_num, freq = parsed
        # Exclude cold run (run_num == 1)
        if run_num <= 1:
            continue
        run_id = f"{seq}_{run_num:02d}"
        # Must have timing info
        if run_id not in timing_map:
            continue
        runs_by_freq.setdefault(freq, []).append((run_id, run_num, fp))

    aggregates: Dict[int, RunAggregate] = {}
    for freq, entries in sorted(runs_by_freq.items()):
        avg_powers: List[float] = []
        durations: List[float] = []
        energies: List[float] = []

        for run_id, run_num, fp in sorted(entries, key=lambda x: x[1]):
            # Duration from timing file
            freq_check, dur = timing_map.get(run_id, (freq, 0.0))
            if freq_check != freq:
                # Keep going but trust filename frequency
                pass

            df = parse_dcgmi_profile(fp)
            if df.empty or "POWER" not in df.columns:
                continue
            # Compute average power for the run
            power_series = df["POWER"].dropna()
            if power_series.empty:
                continue
            avg_p = float(power_series.mean())
            energy = avg_p * float(dur)

            avg_powers.append(avg_p)
            durations.append(float(dur))
            energies.append(float(energy))

        if not avg_powers:
            continue

        # Optional IQR filtering on energy (and implicitly duration/power by index)
        if iqr_filter and len(energies) >= 3:
            arr = np.array(energies, dtype=float)
            q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            keep_idx = [i for i, e in enumerate(energies) if (e >= lower and e <= upper)]
            if keep_idx:
                avg_powers = [avg_powers[i] for i in keep_idx]
                durations = [durations[i] for i in keep_idx]
                energies = [energies[i] for i in keep_idx]

        avg_power = float(np.mean(avg_powers))
        avg_timing = float(np.mean(durations))
        avg_energy = float(np.mean(energies))
        aggregates[freq] = RunAggregate(
            frequency_mhz=freq,
            avg_power_w=avg_power,
            avg_timing_s=avg_timing,
            avg_energy_j=avg_energy,
            power_std_w=float(np.std(avg_powers)) if len(avg_powers) > 1 else 0.0,
            timing_std_s=float(np.std(durations)) if len(durations) > 1 else 0.0,
            run_count=len(avg_powers),
        )

    return aggregates


def read_run_profiles(dir_path: Path) -> Dict[int, "pd.DataFrame"]:
    """Return raw per‑run profile DataFrames keyed by frequency."""
    if pd is None:
        raise RuntimeError("pandas is required to read run profiles")

    mapping: Dict[int, "pd.DataFrame"] = {}
    for fp in list_profile_files(dir_path):
        parsed = parse_run_filename(fp.name)
        if not parsed:
            continue
        _, _, freq = parsed
        try:
            df = parse_dcgmi_profile(fp)
        except Exception:
            continue
        if freq not in mapping:
            mapping[freq] = df
        else:
            # Keep first by default; could concatenate if needed
            pass
    return mapping
