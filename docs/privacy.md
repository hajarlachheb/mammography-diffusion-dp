# What the privacy guarantee covers

## Mechanism

**What is trained under DP.** Only the classifier head is trained, by DP-SGD (Opacus 1.5.4):
- per-sample gradient clipping to norm C = 1.0;
- Gaussian noise with multiplier σ;
- Poisson sampling with rate q = 1/⌈N/batch⌉;
- RDP accounting with δ = 10⁻⁴ over the full 300 epochs.

The calibrated σ for each condition, and how to recompute it, are in `configs/privacy/noise_multipliers.yaml` and
`scripts/calculate_privacy.py`.

**Report the full-run ε.** The full-run ε (0.999 / ≤ 6 / 59.99) is the budget to report. The ε at the selected
checkpoint is lower in some runs because selection may pick an earlier epoch (e.g. 4.56 at ε_target = 6).

## What (ε, δ) does cover

- **The records:** the 1,104 real CBIS-DDSM training images, at the **image level**. In Real+CC-NET runs, the 3,000
  synthetic images are additional records of the same mechanism.
- **The release:** the trained classifier parameters.

## What it does not cover

- **The generator.** CC-NET was trained **without** DP on CBIS-DDSM training images, inferred to be the same 1,104.
  The full generator + classifier pipeline is therefore **not** end-to-end differentially private.
- **Patients.** The images come from 578 patients, with up to 14 images each. By group privacy, the guarantee for a
  patient with k images is much weaker than ε, and it is vacuous for k = 14.
- **Model selection.** The selection procedure queries the 192-image validation set non-privately.
- **Hyperparameters.** They were taken from Osuala et al. and not tuned on private data.

## Synthetic images

- **They are not anonymous.** They come from a non-private generator trained on patient images.
- **Memorisation was not ruled out.** Within this project:
  - no exact duplicates of training images were found;
  - the highest nearest-neighbour correlation was 0.984;
  - the membership-inference probes that were tried had no power.
- **Correct wording:** "no evidence of copying", not "no memorisation".
- **Not released.** No synthetic images are distributed with this repository.
