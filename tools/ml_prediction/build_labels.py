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

from .label_builder import build_labels


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
