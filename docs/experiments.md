# Experiments

All 21 reported classifier runs use `configs/experiments/clean_protocol.yaml`. `scripts/run_experiment.py` turns that
file into the exact commands; their flags were checked to be identical to the original launchers.

## Conditions, run IDs and seeds

| Condition | ε | Run ID pattern (`{seed}` ∈ 42, 43, 44) |
|---|---|---|
| A. Real-only DP | 6 | `CLEAN_RICHARD_S1_real_swin_refmlp_head_dp_eps6_seed{seed}` |
| A. Real-only DP | 1 / 60 | `CLEAN_RICHARD_S2_real_swin_refmlp_head_dp_eps{1,60}_seed{seed}` |
| B. Real+CC-NET DP | 6 | `CLEAN_RICHARD_S1_realsyn_swin_refmlp_head_dp_eps6_seed{seed}` |
| B. Real+CC-NET DP | 1 / 60 | `CLEAN_RICHARD_S2_realsyn_swin_refmlp_head_dp_eps{1,60}_seed{seed}` |
| C. M1 matched control | 6 | `CLEAN_RICHARD_M1_real_bs34_swin_refmlp_head_dp_eps6_seed{seed}` |

**Seed handling.** The seed is passed as `--seed`. The trainer seeds `random`, NumPy and torch before building the
model, so the same seed is expected to give the same head initialisation across conditions. This was not verified from
saved tensors. Opacus's Poisson sampler then draws from
the global torch generator.

**Checking the runs.** `results/verified_summary/run_acceptance.csv` lists every run with:
- its selected epoch;
- ε at the selected checkpoint and over the full run;
- σ and q;
- the checkpoint SHA-256 (21 distinct values).

## Classifier recipe (all conditions)

**Model**
- `--model swin_t --pretrained` (torchvision ImageNet-1k weights).
- `--head_type ref_mlp`: Linear(768, 384) → LeakyReLU(0.2) → Linear(384, 2), no dropout. This is the head of the
  mammo_dp frozen-Swin configuration.
- `--freeze_backbone --unfreeze_last_n 1`: only the head trains, giving **296,066** trainable parameters. Every run
  was checked for this count and for a backbone bit-identical to the ImageNet weights.

**Optimisation**
- `--richard_finetuning` together with explicit flags sets AdamW, lr 1e-5, weight decay 1e-8, label smoothing 0.1,
  H/V flips, normalisation with mean = std = 0.5, 300 epochs and no early stopping.

**DP-SGD**
- `--enable_dp --epsilon {1,6,60} --delta 1e-4 --max_grad_norm 1.0 --dp_accountant rdp`.
- σ is calibrated by Opacus `make_private_with_epsilon` for the full 300 epochs; see `configs/privacy/noise_multipliers.yaml`.
- `--max_physical_batch_size 32` affects memory only; the logical batch sets q.

**Selection**
- `--select_by_auprc --dp_require_epsilon_near_target_for_selection`: keep the epoch with the highest validation AUPRC
  (benign as the positive class, as in mammo_dp) among epochs whose spent ε lies in [0.75, 1.05]·ε_target. Ties go to
  the lower validation loss.

### Documented code change: non-in-place LeakyReLU

The reference head uses `LeakyReLU(0.2)`. With the installed Opacus 1.5.4 / PyTorch 2.5.1, an **in-place** activation
inside the head fails under Opacus's per-sample-gradient hooks, which forbid in-place operations on their views. The
head is therefore built with `inplace=False`.

The forward values and gradients are identical to the in-place version. Only the error is avoided. The project
verified this before the runs (loss and gradients bit-identical), and `tests/test_classifier_head.py` checks
per-sample gradients through the head.

## M1: what it matches and what it does not

**Why batch 34 gives q = 1/33.** The trainer builds `DataLoader(batch_size=34, shuffle=True)` without `drop_last`.
Opacus therefore uses Poisson sampling with q = 1/len(loader) = 1/⌈1104/34⌉ = **1/33**. That gives the same q, the same
σ (2.34375) and the same 9,900 steps as B. `scripts/calculate_privacy.py calibrate` shows this.

**What M1 matches with B:**
- q, σ and the step count;
- the ε trajectory and selection window;
- the expected number of real records per step (33.45).

**What still differs:**

| Difference | Detail |
|---|---|
| Gradient normalisation | Opacus divides by int(N·q): 33 in M1 vs 124 in B. AdamW is only approximately invariant to this constant scale |
| Optimiser coupling | Adam normalises B's combined real + synthetic + noise update |
| Batch composition | About 73% of each B step is synthetic. Realised batch sizes are about 33 ± 6 in M1 vs 124 ± 11 in B |
| Synthetic-image correlations | The synthetic images come from one generator trained on the same real data, and have their own low-level statistics (native 256 px) |

**Interpretation.** B − M1 is the effect of adding the CC-NET records under matched sampling, noise and step count. It
is **not** a measurement of diagnostic image content.

## Statistics

`scripts/compute_statistics.py` uses `src/mammo_diffusion_dp/metrics/`.

**Metrics**
- AUROC, with malignant as the positive class.
- AUPRC with malignant as positive (`auprc_malignant`).
- AUPRC with benign as positive (`auprc_benign`). This is the convention of Osuala et al.; both directions are reported
  and never mixed.

**Test-sampling uncertainty (patient-bootstrap CI)**
- 95% percentile interval over B = 2,000 patient-cluster resamples of the test set.
- Seeds: 20261026 for CBIS-DDSM, 20261027 for BCDR.
- Draws with no positive or no negative image are redrawn.
- The same draws are applied to every model, so paired differences are paired draw by draw.
- This interval describes uncertainty from the finite test set **for fixed trained models**.

**Training-seed uncertainty**
- Sample SD (ddof = 1) over the 3 independently trained seeds.
- Seed t95 = mean ± t(0.975, 2)·SD/√3.
- k/3 = the number of seeds with a positive paired difference.

**Rules**
- The two kinds of uncertainty are reported separately and never merged.
- A bootstrap CI is not evidence of robustness across training runs.

**Check.** Run on the original prediction files, `compute_statistics.py` reproduces all 1,332 per-seed, summary and
paired values in `results/verified_summary/` exactly.

## CC-NET generator

**What CC-NET is here.** A class-conditional latent diffusion model obtained by fine-tuning Stable Diffusion 2.1-base
(UNet, with the VAE from the same source model) on the CBIS-DDSM training patches. It follows the training approach of
[RichardObi/ccnet](https://github.com/RichardObi/ccnet).

### Recorded training run (December 2025)

Source: the training log's argument dump (`configs/ccnet/ccnet_training_dec2025.yaml`).

| Setting | Value |
|---|---|
| Source model | `Manojb/stable-diffusion-2-1-base` |
| Image size | 256 |
| Epochs | 25 |
| Batch | 4 × gradient accumulation 64 |
| Learning rate | 1e-4, 500 warm-up steps |
| Noise schedule | DDPM, 1,000 timesteps, `scaled_linear` |
| Fine-tuning | yes (`fine_tune=True`) |
| VAE scale factor | 0.1 |
| Precision | fp16 |
| Seed | 2 |
| Gradient clipping | 1.0 |
| Selected checkpoint | Epoch 22 (lowest training loss 0.1068), the generator used for the reported pool |
| DP | **None** |

### Command with the included script

```bash
pip install -e ".[generation]"
python -m mammo_diffusion_dp.generation.train_ldm_ccnet --data_path $MDPD_DATA_ROOT/cbis-ddsm/train \
  --output_dir runs/ccnet --source_model Manojb/stable-diffusion-2-1-base --image_size 256 --n_epochs 25 \
  --batch_size 4 --gradient_accumulation_steps 64 --base_lr 1e-4 --lr_warmup_steps 500 --num_train_timesteps 1000 \
  --beta_schedule scaled_linear --fine_tune --scale_factor 0.1 --mixed_precision fp16 --seed 2 --clip_grad_norm_by 1.0
```

**Limitations of this command.**
- The included `train_ldm_ccnet.py` is a **later revision** (August 2026). The December 2025 revision that trained the
  reported generator was not preserved.
- The later revision accepts every recorded argument. It also adds options (e.g. `--conditioning_mode`, default
  `learned`, described in the code as the historical setting) whose equivalence with the December behaviour was not
  verified.
- The exact training image list was not saved. It is inferred to be the 1,104 CBIS-DDSM training images (see
  docs/limitations.md).

### Generation

```bash
python -m mammo_diffusion_dp.generation.generate_ldm_ccnet --model_path runs/ccnet/diffusion_best_model.pth \
  --output_dir $MDPD_SYNTHETIC_DIR --source_model Manojb/stable-diffusion-2-1-base --num_samples_per_class 1500
```

**This produces a class-balanced pool (1,500 + 1,500).** However, **the sampling settings used for the reported pool
(February 2026) were not recorded**:
- the scheduler, number of inference steps, guidance scale and seed are unknown;
- the script's defaults (DDIM, 100 steps, guidance 1.0, seed 42) are not evidence of what was used;
- the docstring example uses 50 steps.

A regenerated pool will therefore differ from the one behind the Real+CC-NET results.

## Image quality (FID, FRDv1)

```bash
pip install -e ".[quality]"
python scripts/prepare_data.py organize-mcgan --source <mammo_dp>/extension/synthetic_data/cbis-ddsm/c-dcgan --target data/mcgan
python -m mammo_diffusion_dp.evaluation.fid_imagenet --data_root $MDPD_DATA_ROOT --ccnet_dir $MDPD_SYNTHETIC_DIR \
  --mcgan_dir data/mcgan --out_dir quality/ --cache_dir quality/cache
python -m mammo_diffusion_dp.evaluation.frd_v1 --data_root $MDPD_DATA_ROOT --ccnet_dir $MDPD_SYNTHETIC_DIR \
  --mcgan_dir data/mcgan --out_dir quality/ --cache_dir quality/cache
```

**Controls run first.** Each script runs its controls before any comparison, and stops if they fail:
- FID(X, X) = 0;
- the feature-based FID equals pytorch-fid end to end;
- FRD(X, X) = −inf;
- the cached FRD equals the unpatched package.

**Ported code.** These modules are the scripts that produced the reported values, with only path handling changed.
They were not re-executed after porting.

**Reproducing the reported numbers needs:**
- the same image pools (the CC-NET pool is not distributed);
- identical file names. For MCGAN this is guaranteed by `organize-mcgan`.

## Historical experiments (not reproducibility targets)

These earlier experiments are not part of the clean evidence:
- the "E2" linear-head DP runs;
- the "E3–E6" synthetic-pretraining runs.

They used a different protocol: a single linear head, an image-level 20% validation carve that the generator had
seen, and seeds 42/123/456. See `experiments/synthetic_pretraining_historical/README.md`.
