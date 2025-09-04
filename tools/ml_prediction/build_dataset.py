#!/usr/bin/env python3
"""
Build a training dataset CSV/Parquet from probe runs and labels.

Usage:
  python -m tools.ml_prediction.build_dataset \
    --results-dir sample-collection-scripts \
    --labels tools/ml_prediction/labels.json \
    --output tools/ml_prediction/datasets/max_only.csv \
    --policy max-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Support both package and direct script execution
try:
    from .dataset_builder import DatasetBuilder  # type: ignore
except Exception:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.ml_prediction.dataset_builder import DatasetBuilder  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ML dataset from results and labels")
    parser.add_argument("--results-dir", required=True, help="Path to sample-collection results base dir")
    parser.add_argument("--labels", required=True, help="Path to labels.json produced by build_labels")
    parser.add_argument("--output", required=True, help="Output CSV or Parquet path for dataset")
    parser.add_argument("--policy", choices=["max-only", "tri-point"], default="max-only", help="Probe policy")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    labels_path = Path(args.labels).resolve()
    out_path = Path(args.output).resolve()

    builder = DatasetBuilder(results_dir, labels_path)
    final = builder.build(out_path, policy=args.policy)
    print(f"Dataset saved to {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
