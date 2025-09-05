# ML‑Based Frequency Prediction — Plan & Progress (v1.2)

This document summarizes the goal, current progress, key findings, and concrete next steps for the ML system that predicts EDP‑optimal GPU frequency from short profiling runs.

**Goal**
- Predict EDP‑optimal GPU frequency (and ED²P) using one or a few probe runs, aligning labels and methodology with the analysis tools (warm‑run averaging, outlier filtering, performance threshold).

**What We’ve Built**
- Labels: `build_labels.py` generates `labels.json` via `edp_optimizer.py` with consistent performance thresholding.
- Datasets: `build_dataset.py` supports `max-only`, `tri-point`, `all-freq` policies. Features include DCGMI stats (POWER/GPUTL/MCUTL/SMCLK/MMCLK/TMPTR/SMACT/DRAMA), ratios, trend slopes (POWER/TMPTR/GPUTL), HAL‑normalized clocks, context, and probe‑frequency features.
- Training: `train_baseline.py` trains an RF classifier with frequency snapping; prints quick metrics.
- Evaluation: `evaluate.py` supports random/workload/GPU splits and reports both frequency error and EDP gap (energy × time). Avoids split leakage.
- Refactors: shared run‑filename parser and unified timing loader reused across builder/extractor, plus `*.joblib` artifacts ignored.

**Results So Far**
- Random holdout (all‑freq): median 0.0 MHz, mean ~26.4 MHz; ~94.5% within 30/60 MHz.
- Workload holdout (e.g., whisper): raw MHz error large; EDP gap moderate (median ~10%, mean ~5%). The EDP curve is relatively flat near the optimum; MHz error can be high while energy remains close to optimal.
- GPU holdout (e.g., H100): raw MHz error moderate; EDP gap median near zero but mean high due to a few outliers → opportunity to improve generalization and reduce extreme misses.

**Methodology Notes**
- Energy estimate: prefers DCGMI `TOTEC` deltas only when positive; otherwise mean(POWER) × duration. Duration falls back to sample count × sampling interval when timing missing.
- Evaluation: if an exact (gpu, workload, freq) tuple is missing from the dataset maps, evaluator falls back to nearest available frequency; skips invalid/non‑finite EDP rows.

**Next Steps (Step‑By‑Step)**
1) Rebuild & Retrain (Completed)
   - Rebuild `all-freq` dataset; retrain RF; verify random split. Baseline: median 0.0 MHz, mean ~26.4 MHz.
2) Holdout Evaluation & EDP Gap
   - Run workload/GPU holdouts, track EDP gap distribution and worst cases; validate leakage controls.
3) Hyperparameter Tuning / Models
   - Sweep RF depth/leaf size; test XGBoost/LightGBM; select using EDP gap on holdouts.
4) Confidence‑Gated Few‑Shot
   - Implement tri‑point probing when predicted confidence is low; re‑predict with augmented features.
5) Diagnostics & Reporting
   - Per‑workload/GPU percentiles; feature importances; highlight worst‑case scenarios.
6) Tri‑Point Selection Improvements
   - Make tri‑point frequency anchors step‑aware using HAL frequencies for portability across GPUs.
7) Packaging
   - Optional API/CLI for one‑shot live probe; integrate confidence gating.

**Acceptance Targets**
- Random split (all‑freq): maintain >90% within 60 MHz; near‑zero median EDP gap.
- Workload holdout: median EDP gap ≤ 5–10%; mean ≤ 10–15%.
- GPU holdout: reduce outliers to bring mean EDP gap ≤ 20% while keeping median near zero.

**Updated TODO Checklist**
- [x] Labels CLI and JSON export
- [x] Dataset builder CLI with `max-only`, `tri-point`, `all-freq`; probe frequency features; DCGMI feature extraction
- [x] Baseline RF training with snapping and quick metrics
- [x] Evaluation script with random/workload/GPU splits and EDP gap (EDP = energy × time)
- [x] Feature enrichment (stats, trends, ratios, HAL‑normalized clocks)
- [x] DRY refactor: shared run‑filename parser + unified timing loader; ignore model artifacts
- [ ] Hyperparameter tuning / alternative models with holdout‑split tracking
- [ ] Confidence‑gated few‑shot inference (tri‑point)
- [ ] Per‑workload/GPU percentiles, worst‑case reporting, and feature importances
- [ ] API/CLI packaging for live use
