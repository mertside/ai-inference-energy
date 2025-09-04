"""
Label Builder (Scaffold)

Responsibilities:
- Generate ground-truth labels (EDP/ED²P optimal frequencies) per GPU–workload
- Reuse existing optimizer logic in tools/analysis/edp_optimizer.py
- Persist labels with the same performance_threshold that will be used at inference

Outputs:
- labels.json: list of entries {gpu, workload, performance_threshold, optimal_frequency_edp,
  optimal_frequency_ed2p, fastest_frequency, max_frequency, metrics...}

This is a scaffold. Implement functionality per the plan in phases.
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
    """Run the EDP optimizer to build labels and save as JSON.

    TODO:
    - Import and run tools/analysis/edp_optimizer.EDPOptimizer
    - Convert its OptimalResult entries to LabelRecord
    - Save to output_file as JSON
    """
    raise NotImplementedError("TODO: implement label builder using EDP optimizer")


def save_labels(records: List[LabelRecord], output_file: Path) -> None:
    """Serialize label records to JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in records]
    output_file.write_text(json.dumps(data, indent=2))
