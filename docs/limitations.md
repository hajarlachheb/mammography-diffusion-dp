# Limitations

## Evidence and claims

| Claim | Status |
|---|---|
| **Downstream CC-NET vs MCGAN** | **Not established.** The matched Real+MCGAN DP experiment was not run (see below). Our Real+CC-NET numbers can be set against Osuala et al.'s *reported* Real+MCGAN numbers only across studies, which is not a controlled comparison |
| **Image fidelity** | CC-NET has a lower ImageNet FID than MCGAN (66.0 vs 94.0). The FRDv1 difference (7.24 vs 7.73) is within subsampling variability. FID uses natural-image features, depends on n and resizing, and does not detect memorisation |
| **In-domain gains** | On CBIS-DDSM the Real+CC-NET gain is seed-sensitive at ε = 6 and 60 (2/3 seeds positive) |
| **Mechanism** | M1 removes only the sampling-rate / noise / step-count confound, at ε = 6 only. The residual gain is not proof that diagnostic image content causes it |
| **Statistical power** | 3 seeds per cell; CBIS-DDSM effects below about 0.1 AUROC cannot be resolved across seeds |

## MCGAN provenance

**Origin.** The 3,000 published MCGAN images are byte-identical to those in the mammo_dp repository. GAN inversion
with positive and negative controls attributes them to the medigan checkpoint `1750_ckpt_test_cbis_ddsm.pt`. Its
documentation describes it as trained on the CBIS-DDSM **test** split. The checkpoint documented as trained on the
training split does not reproduce them.

**What cannot be determined.** The weights alone cannot verify which images trained either checkpoint (there is no
discriminator, and a nearest-neighbour probe had no power). Two explanations remain:
- the images come from a test-trained generator; or
- the checkpoint labels changed when the second checkpoint was added.

**Consequences.**
- These images were used only for the image-quality comparison.
- No classifier was trained on them.
- MCGAN-vs-test distances are not reported.
- This is an open question for the original authors, not a finding about their study.

## Privacy

See docs/privacy.md:
- classifier-only, image-level DP;
- non-private generator;
- non-private model selection;
- weak patient-level guarantees.

## Data and domain

- **BCDR patient grouping.** BCDR patient identifiers are filename-derived and only partly confirmed (127/394).
  BCDR also contains 8 exact-duplicate image pairs, kept as upstream did.
- **Acquisition domain.** CBIS-DDSM and BCDR differ in acquisition and are separable from low-level image
  statistics. A domain-augmentation explanation of the larger external-cohort gain has not been excluded.
- **Resolution.** CC-NET images are 256 px; real and MCGAN patches are 128 px. All are resized to 224 for the
  classifier.

## Reproducibility

See docs/reproducibility.md. The CC-NET training-script revision, the training image list and the sampling settings
of the reported pool were not preserved.

## Scope

- Mass patches only; no full-field mammograms.
- Frozen ImageNet backbone with head-only training.
- ε ∈ {1, 6, 60}. Osuala et al. also report ε = 12 and 20, which were not run here.
- Clinical operating points (sensitivity / specificity) are not evaluated. AUROCs of 0.5–0.65 are far from clinical
  use.
