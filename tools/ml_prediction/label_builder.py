"""
Label Builder

Responsibilities:
- Generate ground-truth labels (EDP/ED²P optimal frequencies) per GPU–workload
- Reuse existing optimizer logic in tools/analysis/edp_optimizer.py
- Persist labels with the same performance_threshold that will be used at inference

Outputs:
- labels.json: list of entries {gpu, workload, performance_threshold, optimal_frequency_edp,
  optimal_frequency_ed2p, fastest_frequency, max_frequency, metrics...}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class LabelRecord:
    gpu: str
    workload: str
    performance_threshold: float
    optimal_frequency_edp_mhz: int
    optimal_frequency_ed2p_mhz: int
    fastest_frequency_mhz: int
    max_frequency_mhz: int
    # Optional metrics
    energy_savings_edp_percent: float | None = None
    energy_savings_ed2p_percent: float | None = None
    performance_vs_max_edp_percent: float | None = None
    performance_vs_max_ed2p_percent: float | None = None


def build_labels(
    results_base_dir: Path,
    performance_threshold: float,
    output_file: Path,
) -> List[LabelRecord]:
    """Run the EDP optimizer to build labels and save as JSON."""
    # Lazy import using importlib to avoid package path issues
    import importlib.util

    edp_path = (Path(__file__).resolve().parents[1] / "analysis" / "edp_optimizer.py").resolve()
    spec = importlib.util.spec_from_file_location("edp_optimizer", str(edp_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load optimizer module at {edp_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    optimizer = mod.EDPOptimizer(str(results_base_dir), performance_threshold)
    results = optimizer.analyze_all_results()

    records: List[LabelRecord] = []
    for r in results:
        # Map EDPOptimizer.OptimalResult (dataclass) to LabelRecord
        try:
            lr = LabelRecord(
                gpu=r.gpu,
                workload=r.workload,
                performance_threshold=performance_threshold,
                optimal_frequency_edp_mhz=int(r.optimal_frequency_edp),
                optimal_frequency_ed2p_mhz=int(r.optimal_frequency_ed2p),
                fastest_frequency_mhz=int(r.fastest_frequency),
                max_frequency_mhz=int(r.max_frequency),
                energy_savings_edp_percent=float(r.energy_savings_vs_max_edp),
                energy_savings_ed2p_percent=float(r.energy_savings_vs_max_ed2p),
                performance_vs_max_edp_percent=float(r.performance_vs_max_edp),
                performance_vs_max_ed2p_percent=float(r.performance_vs_max_ed2p),
            )
            records.append(lr)
        except Exception:
            # Skip malformed entries
            continue

    save_labels(records, output_file)
    return records


def save_labels(records: List[LabelRecord], output_file: Path) -> None:
    """Serialize label records to JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in records]
    output_file.write_text(json.dumps(data, indent=2))
