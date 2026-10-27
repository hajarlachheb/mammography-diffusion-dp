# Verified aggregate results

These files are aggregates only: no images, no per-image predictions, no patient identifiers. All values were computed
from the per-image predictions of the 21 accepted runs by the code in `src/mammo_diffusion_dp/metrics/`. They were
independently re-verified before release; `scripts/compute_statistics.py` reproduces every value from the original
prediction files.

| File | Contents |
|---|---|
| `main_per_seed.csv` | 18 runs (ε ∈ {1, 6, 60} × Real / Real+CC-NET × 3 seeds). Metrics per test set, with a 95% patient-bootstrap CI per run |
| `main_summary.csv` | Mean, sample SD (ddof = 1), test-sampling CI of the seed mean, seed t95, achieved-ε range |
| `main_paired_differences.csv` | Real+CC-NET − Real-only, per seed and as the mean of the 3 seed-matched pairs |
| `m1_per_seed.csv`, `m1_summary.csv`, `m1_paired_differences.csv` | ε = 6: A (Real-only), M1 (matched control), C (Real+CC-NET). Contrasts C − M1, M1 − A, C − A |
| `run_acceptance.csv` | All 21 runs: selected epoch, ε at the selected checkpoint and for the full run, σ, q, number of acceptance checks passed, checkpoint SHA-256 |
| `image_quality_fid_frd.csv` | ImageNet FID and FRDv1: mean, SD and 2.5 / 97.5 percentiles over 20 subsample draws (n = 400 per side) |
| `osuala2024_reported_table1.csv` | Values reported by Osuala et al. (2024), Table 1, used for the comparison table |

## Column conventions

- **arm:** `real` = Real-only DP (A); `realsyn` = Real+CC-NET DP (B).
- **Metrics:**
  - `*_auroc`: malignant-positive AUROC.
  - `*_auprc_malignant`: AUPRC with malignant as positive.
  - `*_auprc_benign`: AUPRC with benign as positive (the convention of Osuala et al.).
- **`*_patient_ci95`, `*_test_sampling_ci95`, `test_sampling_ci95_lo/hi`:** 95% percentile interval over 2,000
  patient-cluster bootstrap resamples of the test set. Seeds: 20261026 (CBIS-DDSM) / 20261027 (BCDR). Identical draws
  are used for every model. **Test-set sampling uncertainty for fixed trained models only.**
- **`*_sd_ddof1`, `seed_sd_ddof1`, `*_seed_t95`, `n_seeds_positive`:** training-seed uncertainty over 3 independently
  trained runs. Seed t95 = mean ± t(0.975, 2)·SD/√3.
- **`epsilon_full_300_epoch_run`:** the budget to report.
- **`epsilon_at_selected_checkpoint`:** the ε spent up to the selected epoch.
- **BCDR patient identifiers:** filename-derived (127/394 confirmed), so BCDR CIs assume that grouping.
