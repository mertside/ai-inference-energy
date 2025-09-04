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

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

ProbePolicy = Literal["max-only", "tri-point", "all-freq"]


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
        self.labels: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if self.labels_file.exists():
            data = json.loads(self.labels_file.read_text())
            for rec in data:
                key = (str(rec.get("gpu")).upper(), str(rec.get("workload")).lower())
                self.labels[key] = rec
        else:
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")

    @staticmethod
    def _parse_dir_name(dir_path: Path) -> Optional[Tuple[str, str, str]]:
        """Parse directory name results_<gpu>_<workload>_job_<id>.

        Returns: (gpu, workload, job_id) or None
        """
        m = re.match(r"results_([^_]+)_([^_]+)_job_(\d+)", dir_path.name)
        if not m:
            return None
        return (m.group(1).upper(), m.group(2).lower(), m.group(3))

    def _select_probe_runs(self, result_dir: Path, policy: ProbePolicy) -> List[Path]:
        """Return list of run profile files matching the selected probe policy.

        TODO:
        - Implement logic to pick max-only or tri-point frequencies
        - Use hardware.gpu_info to query valid ranges if needed
        """
        files = sorted(result_dir.glob("run_*_profile.csv"))
        if not files:
            return []
        # Extract (freq, run_num, path)
        entries: List[Tuple[int, int, Path]] = []
        pat = re.compile(r"run_(\d+)_(\d+)_freq_(\d+)_profile\.csv$")
        for fp in files:
            m = pat.search(fp.name)
            if not m:
                continue
            run_num = int(m.group(2))
            freq = int(m.group(3))
            entries.append((freq, run_num, fp))
        if not entries:
            return []

        # max-only: pick highest frequency and the first warm run (run_num > 1)
        if policy == "max-only":
            entries.sort(key=lambda x: (x[0], -x[1]), reverse=True)  # sort by freq desc, run_num asc (since negative)
            max_freq = entries[0][0]
            warm = [e for e in entries if e[0] == max_freq and e[1] > 1]
            if warm:
                # Prefer run_num == 2 if present
                warm.sort(key=lambda x: x[1])
                return [warm[0][2]]
            # Fallback: any file at max freq
            any_max = [e for e in entries if e[0] == max_freq]
            any_max.sort(key=lambda x: x[1])
            return [any_max[0][2]] if any_max else []

        # all-freq: pick one warm run per available frequency (prefer run_num == 2)
        if policy == "all-freq":
            by_freq: Dict[int, List[Tuple[int, Path]]] = {}
            for freq, run_num, fp in entries:
                by_freq.setdefault(freq, []).append((run_num, fp))
            selected: List[Path] = []
            for freq, lst in sorted(by_freq.items(), reverse=True):
                warm = [item for item in lst if item[0] > 1]
                cand = warm or lst
                cand.sort(key=lambda x: x[0])
                selected.append(cand[0][1])
            return selected

        # tri-point (basic): choose ~100%, ~70%, ~50% of max among available
        entries.sort(key=lambda x: x[0], reverse=True)
        unique_freqs = sorted({e[0] for e in entries}, reverse=True)
        if not unique_freqs:
            return []
        max_f = unique_freqs[0]
        targets = [max_f, int(round(0.7 * max_f / 15.0) * 15), int(round(0.5 * max_f / 15.0) * 15)]
        selected: List[Path] = []
        for t in targets:
            # pick nearest available frequency
            nearest = min(unique_freqs, key=lambda f: abs(f - t))
            cand = [e for e in entries if e[0] == nearest and e[1] > 1]
            if not cand:
                cand = [e for e in entries if e[0] == nearest]
            if cand:
                cand.sort(key=lambda x: x[1])
                selected.append(cand[0][2])
        # Deduplicate file paths
        seen = set()
        result = []
        for p in selected:
            if p not in seen:
                result.append(p)
                seen.add(p)
        return result

    def _extract_features_for_runs(self, run_files: List[Path], context: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate/concatenate features from selected run files.

        TODO:
        - For now, use lightweight parsing (no pandas) to extract avg power
        - Later, switch to ProfileFeatureExtractor for richer features
        """
        if not run_files:
            return {}

        def parse_power_values(path: Path) -> List[float]:
            vals: List[float] = []
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "GPU" not in line:
                        continue
                    parts = line.split()
                    # scan a reasonable slice for power value
                    for i in range(3, min(len(parts), 10)):
                        try:
                            power = float(parts[i])
                            if 0 < power < 1000:
                                vals.append(power)
                                break
                        except ValueError:
                            continue
            return vals

        # Load timing map
        def load_timing(dir_path: Path) -> Dict[str, Tuple[int, float]]:
            tmap: Dict[str, Tuple[int, float]] = {}
            tfile = dir_path / "timing_summary.log"
            if not tfile.exists():
                return tmap
            with tfile.open("r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 5:
                        continue
                    run_id, freq_str, dur_str, exit_code, status = parts[:5]
                    try:
                        tmap[run_id] = (int(freq_str), float(dur_str))
                    except ValueError:
                        continue
            return tmap

        # Compute pooled features (avg across selected runs)
        import re

        dir_path = run_files[0].parent
        timing_map = load_timing(dir_path)

        power_means: List[float] = []
        durations: List[float] = []
        energies: List[float] = []

        pat = re.compile(r"run_(\d+)_(\d+)_freq_(\d+)_profile\.csv$")
        for fp in run_files:
            m = pat.search(fp.name)
            run_id = None
            if m:
                seq, run_num = int(m.group(1)), int(m.group(2))
                run_id = f"{seq}_{run_num:02d}"
            pvals = parse_power_values(fp)
            if not pvals:
                continue
            pmean = sum(pvals) / len(pvals)
            power_means.append(pmean)
            dur = 0.0
            if run_id and run_id in timing_map:
                _, dur = timing_map[run_id]
            durations.append(dur)
            energies.append(pmean * dur)

        if not power_means:
            return {}

        # Aggregate
        avg_power = sum(power_means) / len(power_means)
        avg_duration = sum(durations) / max(len(durations), 1)
        avg_energy = sum(energies) / max(len(energies), 1)

        return {
            "power_mean": avg_power,
            "duration_seconds": avg_duration,
            "energy_estimate_j": avg_energy,
            # Context passthrough
            "gpu_type": context.get("gpu_type"),
            "sampling_interval_ms": context.get("sampling_interval_ms"),
            "probe_policy": context.get("probe_policy"),
        }

    def _extract_features_single_run(self, run_file: Path, context: Dict[str, Any], max_freq: Optional[int]) -> Dict[str, Any]:
        """Extract lightweight features for a single run file.

        Features:
        - power_mean, duration_seconds, energy_estimate_j
        - probe_frequency_mhz (from filename) and normalized ratio to max_freq if provided
        - gpu_type, sampling_interval_ms, probe_policy
        """
        import re

        power_vals: List[float] = []
        with run_file.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "GPU" not in line:
                    continue
                parts = line.split()
                for i in range(3, min(len(parts), 10)):
                    try:
                        power = float(parts[i])
                        if 0 < power < 1000:
                            power_vals.append(power)
                            break
                    except ValueError:
                        continue
        pmean = sum(power_vals) / len(power_vals) if power_vals else 0.0

        # timing
        m = re.search(r"run_(\d+)_(\d+)_freq_(\d+)_profile\.csv$", run_file.name)
        run_id = None
        probe_freq = None
        if m:
            seq, run_num, fmhz = int(m.group(1)), int(m.group(2)), int(m.group(3))
            run_id = f"{seq}_{run_num:02d}"
            probe_freq = fmhz
        dur = 0.0
        tfile = run_file.parent / "timing_summary.log"
        if run_id and tfile.exists():
            with tfile.open("r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 5:
                        continue
                    rid, freq_str, dur_str, exit_code, status = parts[:5]
                    if rid == run_id:
                        try:
                            dur = float(dur_str)
                        except ValueError:
                            pass
                        break

        feats = {
            "power_mean": pmean,
            "duration_seconds": dur,
            "energy_estimate_j": pmean * dur,
            "gpu_type": context.get("gpu_type"),
            "sampling_interval_ms": context.get("sampling_interval_ms"),
            "probe_policy": context.get("probe_policy"),
        }
        if probe_freq is not None:
            feats["probe_frequency_mhz"] = probe_freq
            if max_freq and max_freq > 0:
                feats["probe_freq_ratio"] = float(probe_freq) / float(max_freq)
        if max_freq:
            feats["max_frequency_mhz"] = max_freq
        return feats

    def build(self, output_file: Path, policy: ProbePolicy = "max-only") -> Path:
        """Build dataset and save to output_file.

        TODO:
        - Iterate result directories
        - Load label for gpu/workload from labels_file
        - Select runs per policy; extract features
        - Write dataset (CSV/Parquet); return path
        """
        rows: List[Dict[str, Any]] = []
        for dir_entry in sorted(self.results_base_dir.iterdir()):
            if not dir_entry.is_dir() or not dir_entry.name.startswith("results_"):
                continue
            parsed = self._parse_dir_name(dir_entry)
            if not parsed:
                continue
            gpu, workload, job_id = parsed
            label = self.labels.get((gpu, workload))
            if not label:
                # Skip configs without labels
                continue

            run_files = self._select_probe_runs(dir_entry, policy)
            if not run_files:
                continue

            context = {
                "gpu_type": gpu,
                "probe_policy": policy,
                # Default sampling interval (ms); can be inferred later from logs
                "sampling_interval_ms": 50,
            }
            if policy == "all-freq":
                max_f = int(label.get("max_frequency_mhz")) if label.get("max_frequency_mhz") else None
                for fp in run_files:
                    feats = self._extract_features_single_run(fp, context, max_f)
                    if not feats:
                        continue
                    row = {
                        **feats,
                        "gpu": gpu,
                        "workload": workload,
                        "probe_policy": policy,
                        "label_edp": int(label.get("optimal_frequency_edp_mhz")),
                        "label_ed2p": int(label.get("optimal_frequency_ed2p_mhz")),
                        "performance_threshold": float(label.get("performance_threshold", 5.0)),
                    }
                    rows.append(row)
            else:
                feats = self._extract_features_for_runs(run_files, context)
                if not feats:
                    continue
                row = {
                    **feats,
                    "gpu": gpu,
                    "workload": workload,
                    "probe_policy": policy,
                    "label_edp": int(label.get("optimal_frequency_edp_mhz")),
                    "label_ed2p": int(label.get("optimal_frequency_ed2p_mhz")),
                    "performance_threshold": float(label.get("performance_threshold", 5.0)),
                }
                rows.append(row)

        # Write CSV without pandas to avoid heavy deps in some environments
        import csv as _csv

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", newline="", encoding="utf-8") as f:
            if not rows:
                f.write("")
                return output_file
            # Union of keys to get stable header
            header_keys: List[str] = []
            seen: set = set()
            for r in rows:
                for k in r.keys():
                    if k not in seen:
                        header_keys.append(k)
                        seen.add(k)
            writer = _csv.DictWriter(f, fieldnames=header_keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return output_file
