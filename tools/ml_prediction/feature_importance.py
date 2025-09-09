"""
Feature Importance Export Utilities

Responsibilities:
- Persist model-based feature importances to CSV and JSON
- Generate a simple horizontal bar plot (top-N) for quick inspection
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class FIContext:
    dataset: Optional[str] = None
    split: Optional[str] = None
    holdout: Optional[float] = None
    holdout_workloads: Optional[List[str]] = None
    holdout_gpus: Optional[List[str]] = None
    model: str = "RandomForest"
    tag: Optional[str] = None


def _sorted_items(importances: Dict[str, float]) -> List[tuple[str, float]]:
    return sorted(importances.items(), key=lambda x: x[1], reverse=True)


def save_feature_importances(
    importances: Dict[str, float],
    out_dir: Path,
    top_n: int = 25,
    context: Optional[FIContext] = None,
) -> Dict[str, Path]:
    """Save feature importances to CSV, JSON, and a bar plot.

    Returns a dict with paths for keys: csv, json, png, md (some may be missing).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Path] = {}

    items = _sorted_items(importances)
    # CSV
    csv_path = out_dir / "feature_importances.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("feature,importance,rank\n")
        for rank, (name, val) in enumerate(items, start=1):
            f.write(f"{name},{float(val):.10f},{rank}\n")
    files["csv"] = csv_path

    # JSON
    import json

    json_path = out_dir / "feature_importances.json"
    payload = {
        "importances": [{"feature": k, "importance": float(v)} for k, v in items],
        "context": asdict(context) if context else None,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    files["json"] = json_path

    # Markdown (top-N)
    md_path = out_dir / "feature_importances.md"
    with md_path.open("w", encoding="utf-8") as f:
        title = "Top Feature Importances"
        if context and context.tag:
            title += f" ({context.tag})"
        f.write(f"# {title}\n\n")
        if context:
            f.write("Context:\n\n")
            if context.dataset:
                f.write(f"- Dataset: `{context.dataset}`\n")
            if context.split:
                f.write(f"- Split: `{context.split}`\n")
            if context.holdout is not None:
                f.write(f"- Holdout: `{context.holdout}`\n")
            if context.holdout_workloads:
                f.write(f"- Holdout workloads: `{', '.join(context.holdout_workloads)}`\n")
            if context.holdout_gpus:
                f.write(f"- Holdout GPUs: `{', '.join(context.holdout_gpus)}`\n")
            f.write(f"- Model: `{context.model}`\n\n")
        f.write("| Rank | Feature | Importance |\n|---:|---|---:|\n")
        for rank, (name, val) in enumerate(items[:top_n], start=1):
            f.write(f"| {rank} | `{name}` | {float(val):.6f} |\n")
    files["md"] = md_path

    # Plot (top-N)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        top_items = items[:top_n]
        if top_items:
            names = [k for k, _ in top_items][::-1]  # reverse for horizontal plot
            vals = [float(v) for _, v in top_items][::-1]
            plt.figure(figsize=(8, max(3, 0.35 * len(names))))
            sns.set_style("whitegrid")
            ax = sns.barplot(x=vals, y=names, color="#348ABD")
            ax.set_xlabel("Importance (Gini)")
            ax.set_ylabel("Feature")
            title = "Feature Importances (Top {} )".format(len(names))
            if context and context.tag:
                title += f" — {context.tag}"
            ax.set_title(title)
            plt.tight_layout()
            png_path = out_dir / "feature_importances_top.png"
            plt.savefig(png_path, dpi=160)
            plt.close()
            files["png"] = png_path
    except Exception:
        # Plot is optional; ignore if backend/mpl not available
        pass

    return files


# Local import to avoid top-level import if unused
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .models.random_forest_predictor import RandomForestFrequencyPredictor  # noqa: F401
