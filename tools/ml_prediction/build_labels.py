#!/usr/bin/env python3
"""
Build labels.json using the EDP optimizer results.

Usage:
  python -m tools.ml_prediction.build_labels \
    --results-dir sample-collection-scripts \
    --performance-threshold 5.0 \
    --output tools/ml_prediction/labels.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Support both package and direct script execution
try:
    from .label_builder import build_labels  # type: ignore
except Exception:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.ml_prediction.label_builder import build_labels  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description="Build labels.json for ML training")
    parser.add_argument("--results-dir", required=True, help="Path to sample-collection results base dir")
    parser.add_argument("--performance-threshold", type=float, default=5.0, help="Performance threshold (%)")
    parser.add_argument("--output", default="tools/ml_prediction/labels.json", help="Output labels JSON file path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_path = Path(args.output).resolve()

    records = build_labels(results_dir, args.performance_threshold, out_path)
    print(f"Wrote {len(records)} labels to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
