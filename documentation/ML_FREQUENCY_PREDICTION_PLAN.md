# ML‑Based Frequency Prediction — Plan & Progress (v1.2)

This document summarizes the goal, current progress, key findings, and concrete next steps for the ML system that predicts EDP‑optimal GPU frequency from short profiling runs.

**Goal**
- Predict EDP‑optimal GPU frequency (and ED²P) using one or a few probe runs, aligning labels and methodology with the analysis tools (warm‑run averaging, outlier filtering, performance threshold).

**What We’ve Built**
- Labels: `build_labels.py` generates `labels.json` via `edp_optimizer.py` with consistent performance thresholding.
- Datasets: `build_dataset.py` supports `max-only`, `tri-point`, `all-freq` policies. Features include DCGMI stats (POWER/GPUTL/MCUTL/SMCLK/MMCLK/TMPTR) + ratios + context + probe frequency features.
- Training: `train_baseline.py` trains an RF classifier with frequency snapping; prints quick metrics.
- Evaluation: `evaluate.py` supports random/workload/GPU splits and reports both frequency error and EDP gap (energy × time). Avoids split leakage.

**Results So Far**
- Random holdout (all‑freq): excellent (median 0 MHz, mean ~26 MHz; >90% within 60 MHz).
- Workload holdout (e.g., whisper): raw MHz error large; EDP gap moderate (median ~10%, mean ~5%). The EDP curve is relatively flat near the optimum; MHz error can be high while energy remains close to optimal.
- GPU holdout (e.g., H100): raw MHz error moderate; EDP gap median near zero but mean high due to a few outliers → opportunity to improve generalization and reduce extreme misses.

**Methodology Notes**
- Energy estimate: prefers DCGMI `TOTEC` deltas only when positive; otherwise mean(POWER) × duration. Duration falls back to sample count × sampling interval when timing missing.
- Evaluation: if an exact (gpu, workload, freq) tuple is missing from the dataset maps, evaluator falls back to nearest available frequency; skips invalid/non‑finite EDP rows.

**Next Steps (Step‑By‑Step)**
1) Feature Enrichment
   - Add normalized clocks/utilizations (e.g., SMCLK/max, MMCLK/max; normalize GPUTL/MCUTL as needed).
   - Add trend features (linear slopes) for POWER, TMPTR, GPUTL to capture stability/dynamics.
   - Include PSTATE/power‑limit features when available.
2) Rebuild & Retrain
   - Rebuild `all-freq` dataset; retrain RF; re‑evaluate on random and holdout splits.
3) Hyperparameter Tuning
   - Sweep RF depth/leaf size and/or test XGBoost/LightGBM; select by EDP gap on workload/GPU holdouts.
4) Confidence‑Gated Few‑Shot
   - Implement tri‑point probing at inference when confidence is low; re‑predict with augmented features.
5) Diagnostics & Reporting
   - Extend evaluator to print EDP gap percentiles per workload and per GPU; highlight worst‑case scenarios.
6) Packaging
   - Optionally add API/CLI for one‑shot prediction from a live probe; integrate confidence gating.

**Acceptance Targets**
- Random split (all‑freq): maintain >90% within 60 MHz; near‑zero median EDP gap.
- Workload holdout: median EDP gap ≤ 5–10%; mean ≤ 10–15%.
- GPU holdout: reduce outliers to bring mean EDP gap ≤ 20% while keeping median near zero.

**Updated TODO Checklist**
- [x] Labels CLI and JSON export
- [x] Dataset builder CLI with `max-only`, `tri-point`, `all-freq`; probe frequency features; DCGMI feature extraction
- [x] Baseline RF training with snapping and quick metrics
- [x] Evaluation script with random/workload/GPU splits and EDP gap (EDP = energy × time)
- [ ] Enrich features (normalized clocks/utilizations; trend features; PSTATE)
- [ ] Hyperparameter tuning / alternative models with holdout‑split tracking
- [ ] Confidence‑gated few‑shot inference (tri‑point)
- [ ] Per‑workload/GPU percentiles and worst‑case reporting
- [ ] API/CLI packaging for live use
