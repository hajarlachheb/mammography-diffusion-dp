# Reproducibility

## What was checked before release

| Check | Result |
|---|---|
| `train_classifier.py`, `evaluate_classifier.py` | The reported training/evaluation protocol is unchanged; checkpoint loading was subsequently hardened with `weights_only=True` |
| `scripts/run_experiment.py` commands | Flags identical to the original Stage 1, Stage 2 and M1 launchers |
| `scripts/prepare_data.py build-folders` | Rebuilt manifest identical to the one used (5,208 entries: split, class, file, SHA-256, patient) |
| `scripts/prepare_data.py validate` | Passes on the original data. Zero patient overlap between CBIS train / val / test; zero SHA-256 overlap across all four splits |
| `scripts/prepare_data.py organize-mcgan` | Output byte- and name-identical to the MCGAN folder used for FID / FRDv1 |
| `scripts/calculate_privacy.py calibrate` | σ, q and steps match every recorded run (7 condition × ε combinations) |
| `scripts/compute_statistics.py` | All 1,332 per-seed, summary and paired values in `results/verified_summary/` were reproduced exactly from the original prediction files; those per-image predictions are not released |
| `scripts/reproduce_tables.py` | Table bodies identical to the manuscript tables |
| `scripts/reproduce_figures.py` | Figures pixel-identical to the manuscript figures (same library versions) |
| Tests | `pytest` passes, CPU only, artificial fixtures |

## Reproduction status

| Component | Status |
|---|---|
| Data preparation | Fully reproducible once the patch dataset has been obtained |
| A. Real-only DP, ε = 1 / 6 / 60 | Fully reproducible protocol (data + GPU). GPU kernels are not bitwise deterministic, so expect small numeric differences, not identical weights. Not re-run for this release |
| C. M1, ε = 6 | Same as A |
| B. Real+CC-NET DP | Protocol fully specified. **The exact synthetic pool is not distributed**, so results depend on the pool used |
| Published result tables, figures and confidence intervals | Reproducible from the supplied aggregate CSVs; this regenerates the published intervals but does not recompute the bootstrap |
| Bootstrap intervals | Not independently recomputable from this repository; the required per-image predictions are not released |
| Privacy calibration and replay | Fully reproducible on CPU. Replay needs the checkpoints, which are not distributed |
| FID / FRDv1 | Code ported from the scripts that produced the values (path handling only). Not re-executed after porting. Needs the CC-NET pool |
| CC-NET training | Partially reproducible: recorded arguments plus a later script revision (docs/experiments.md) |
| CC-NET sampling of the reported pool | Original settings unavailable |
| Weights: CC-NET, 21 classifiers | Not distributed; their SHA-256 hashes are listed in `results/verified_summary/run_acceptance.csv` |
| Matched Real+MCGAN DP comparison | Not performed |

## Unavailable original artifacts

1. **The December 2025 revision of the CC-NET training script.** It was overwritten by later revisions.
2. **The CC-NET training image list.** It was not saved, and the logged data path no longer resolves. It is inferred to
   be the 1,104 training images from:
   - file dates;
   - an optimiser-step-count bound (N ∈ [1,025, 1,280] if the loader was accelerate-prepared, as in both surviving
     revisions).
3. **The sampling command and settings** that generated the reported 3,000-image CC-NET pool.
4. **The training data and split of the MCGAN checkpoint** that produced the published MCGAN images. This is owned by
   the original authors; see docs/limitations.md.

The exact CC-NET synthetic pool used for condition B and its original sampling settings are also unavailable. The
Real+CC-NET results are therefore only partially reproducible: a newly trained and sampled pool will not be the
reported pool.

## Software

| Component | Version |
|---|---|
| Python | 3.12 |
| torch | 2.5.1 (CUDA 12.1 build) |
| torchvision | 0.20.1 |
| opacus | 1.5.4 |
| numpy | 2.4.4 |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| CC-NET | diffusers 0.30.3, transformers 4.45.2, accelerate 1.0.1 |
| FID | pytorch-fid 0.3.0 (run with torch 2.5.1). A small compatibility shim for scipy ≥ 1.16 is included in the code |
| FRDv1 | frd-score 1.0.1 |
