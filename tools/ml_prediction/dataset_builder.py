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
        - Use ProfileFeatureExtractor to build features per run
        - For tri-point, decide concatenation vs pooling (documented in context)
        """
        if pd is None:
            raise RuntimeError("pandas is required")
        from .feature_extractor import ProfileFeatureExtractor
        from .profile_reader import parse_dcgmi_profile

        extractor = ProfileFeatureExtractor()
        if not run_files:
            return {}

        if context.get("probe_policy") == "tri-point" and len(run_files) > 1:
            # Simple pooling: average common numeric features across runs
            pooled: Dict[str, Any] = {}
            count = 0
            for fp in run_files:
                df = parse_dcgmi_profile(fp)
                feats = extractor.extract(df, context)
                # Accumulate numeric-only feature means
                for k, v in feats.items():
                    if isinstance(v, (int, float)):
                        pooled[k] = pooled.get(k, 0.0) + float(v)
            if run_files:
                count = len(run_files)
            for k in list(pooled.keys()):
                pooled[k] = pooled[k] / max(count, 1)
            return pooled

        # max-only or single file
        df = parse_dcgmi_profile(run_files[0])
        return extractor.extract(df, context)

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

        if not rows:
            # Ensure file exists even if empty
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text("")
            return output_file

        df = pd.DataFrame(rows)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.suffix.lower() in {".parquet", ".pq"}:
            df.to_parquet(output_file)
        else:
            df.to_csv(output_file, index=False)
        return output_file
