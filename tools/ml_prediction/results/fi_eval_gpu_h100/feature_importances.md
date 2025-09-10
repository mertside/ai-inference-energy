# Top Feature Importances (rf_eval_loo_gpu)

Context:

- Dataset: `/Users/MertSide/Developer/GitProjects/ai-inference-energy/tools/ml_prediction/datasets/all_freq.csv`
- Split: `gpu`
- Holdout GPUs: `H100`
- Model: `RandomForest`

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `workload` | 0.377041 |
| 2 | `duration_seconds` | 0.094880 |
| 3 | `energy_estimate_j` | 0.043914 |
| 4 | `smact_p95` | 0.029343 |
| 5 | `drama_mean` | 0.026755 |
| 6 | `smact_std` | 0.026464 |
| 7 | `smact_max` | 0.024407 |
| 8 | `mcutl_mean` | 0.023928 |
| 9 | `mcutl_max` | 0.016826 |
| 10 | `drama_std` | 0.016779 |
| 11 | `mcutl_std` | 0.015608 |
| 12 | `drama_max` | 0.015389 |
| 13 | `mem_to_gpu_ratio` | 0.015135 |
| 14 | `drama_p95` | 0.014952 |
| 15 | `mcutl_p95` | 0.014763 |
| 16 | `gputl_slope` | 0.014242 |
| 17 | `power_slope` | 0.013771 |
| 18 | `gputl_std` | 0.013476 |
| 19 | `gputl_mean` | 0.013324 |
| 20 | `gputl_max` | 0.012936 |
| 21 | `power_std` | 0.012642 |
| 22 | `smact_mean` | 0.012091 |
| 23 | `gputl_p95` | 0.011956 |
| 24 | `tmptr_slope` | 0.011571 |
| 25 | `power_p95` | 0.011411 |
| 26 | `power_max` | 0.009854 |
| 27 | `tmptr_min` | 0.009044 |
| 28 | `tmptr_max` | 0.008303 |
| 29 | `tmptr_mean` | 0.007788 |
| 30 | `gpu_type` | 0.007787 |
