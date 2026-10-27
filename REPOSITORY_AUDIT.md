# Repository audit: scientific-software review

Date 2026-10-27. Reviewer stance: a skeptical scientific-software reviewer who checked each claim against the private
originals before writing it here.

## Verdict: **READY FOR PRIVATE CI STAGING**

**What READY FOR PRIVATE CI STAGING means here.** The candidate:
- contains no protected clinical data, secrets or machine-specific paths;
- installs, and its tests pass;
- matches the executed code;
- regenerated every included table and figure exactly in the audited environment.

The author confirmed A1–A4 and A6 on 2026-10-27. A5 is deferred until a manuscript citation exists. Public visibility
remains gated on passing CI, a final review of the GitHub-rendered files, and re-verification of the candidate.

## 1. Can another researcher install the code?

**Yes.** `pip install -e ".[dev]"` installs the `mammo_diffusion_dp` package; this was tested in a fresh virtual
environment. `requirements.txt` pins the versions used for the reported runs.

**Dependency notes**
- `timm` is optional and not needed.
- The original analysis environment lacked opacus's declared dependency `opt-einsum` (`pip check`). Opacus worked
  regardless; a fresh install from PyPI pulls it in.
- pytorch-fid and frd-score are optional extras (`[quality]`).
- CI installs CPU-only torch.

## 2. Does the code match the published methods?

**Yes, for the classifier experiments.**
- **Unchanged code.** `train_classifier.py` and `evaluate_classifier.py` are byte-identical to the versions that
  produced all 21 runs.
- **Commands.** `scripts/run_experiment.py` generates training commands whose flags are identical to the original
  Stage 1, Stage 2 and M1 launchers. This was checked flag by flag for each condition.
- **Privacy calibration.** `scripts/calculate_privacy.py calibrate` reproduces σ, q and the step count of every
  condition through the trainer's own Opacus code path.
- **The documented LeakyReLU change** (non-in-place) is the code that ran.

**Partially, for CC-NET.** The included training and generation scripts are a later revision (see §3).

## 3. Can the clean experiments be reproduced?

| Experiment | Assessment |
|---|---|
| Real-only (A) and M1 (C) | Yes, given the patch dataset and a GPU. Not re-run for this release; bitwise identity of weights is not expected on GPUs |
| Real+CC-NET (B) | The protocol is fully specified. The exact synthetic pool is **not distributed**, and the generation settings of that pool were not recorded, so numbers depend on the pool |
| Statistics | Yes. `compute_statistics.py` reproduced all **1,332** released values exactly from the original per-image predictions |
| Data preparation | Yes. The rebuilt training manifest is identical to the original (5,208 entries); split validation passes on the original data |
| FID / FRDv1 | Ported with path changes only, and **not re-executed after porting**. The MCGAN arrangement step is verified byte- and name-identical |

## 4. Are historical experiments properly separated?

**Yes.**
- The E2 (linear head) and E3–E6 (synthetic pretraining) families are excluded.
- No configuration, result or figure from them is included.
- `experiments/synthetic_pretraining_historical/README.md` explains why.
- `train_classifier.py` still contains the generic options those runs used. That is unavoidable if the executed file is
  kept unchanged, and is documented.

## 5. Do the results regenerate from the included summary files?

**Yes.**
- **Tables:** `scripts/reproduce_tables.py` output is identical to the manuscript tables (bodies compared line by line).
- **Figures:** `scripts/reproduce_figures.py` output is pixel-identical to the manuscript figures (0 differing pixels).
- **Consistency:** tests check that every summary mean, SD and paired difference follows from the per-seed rows.

## 6. Do the privacy claims match the implemented mechanisms?

**Yes.** The documentation states image-level, classifier-only DP. It states that the generator is non-private, that
selection is not covered, and that patient-level guarantees are weak. Synthetic images are not presented as anonymous.
These statements match the code: DP-SGD wraps only the trainer; the generator scripts have no DP.

## 7. Do the tests pass?

**Yes: 44 passed, 0 failed, about 46 s on CPU.** The tests use only artificial fixtures. They cover:
- split, overlap and label checks;
- head architecture and the 296,066 trainable parameters;
- Opacus per-sample gradients with the non-in-place head;
- the checkpoint round-trip through the evaluator;
- Poisson sampling rate, σ calibration and ε accounting;
- weighted metrics against sklearn, bootstrap properties and seed statistics;
- duplicate-seed detection;
- results consistency, table and figure regeneration;
- every documented CLI `--help`;
- absence of machine paths and secrets.

## 8. Is protected or unlicensed material present?

**No protected material.** There are no images, weights, per-image predictions, patient identifiers or logs (see
`SECURITY_AUDIT.md` and `DATA_REDISTRIBUTION_AUDIT.md`).

**Licensing.** The repository's original code is offered under Apache-2.0. That is compatible with the Apache-2.0
upstream projects (mammo_dp, ccnet) it follows. No third-party source files are vendored.

## 9. Is the repository suitable for public release?

**After the author actions below, yes.**

### Author decisions (A1–A6)

| # | Action |
|---|---|
| A1 | **Confirmed.** `CITATION.cff` and `NOTICE` identify Hajar Lachheb and the confirmed affiliation; the commit uses the GitHub noreply address. Set `date-released` when the repository is made public. |
| A2 | **Confirmed.** Apache-2.0 is approved; the author confirmed no CC-NET source was copied verbatim. |
| A3 | **Confirmed.** The author confirmed the CBIS-DDSM terms and the appropriateness of pointing to `mammo_dp/dataset16062024`. |
| A4 | **Confirmed.** Keep `Manojb/stable-diffusion-2-1-base`; its third-party mirror provenance is documented in `THIRD_PARTY_LICENSES.md`. |
| A5 | **Deferred.** Add manuscript details only when they exist; do not state acceptance or a DOI until confirmed. |
| A6 | **Confirmed.** The author reports that required institutional approval has been obtained. |

### Recommended, not blocking

- **R1, scope of the CC-NET scripts.** They contain options from later, unrelated full-field work (clean-background
  losses, mask conditioning). Their defaults are intended to be neutral, but default-equivalence with the December 2025
  revision is unverified. Consider documenting a minimal command set or trimming unused options in a clearly separate
  commit.
- **R2, FID / FRDv1.** Re-execute the ported scripts once on the original data to confirm byte-identical outputs. The
  ported code differs only in path handling; it was not re-run, to avoid a long CPU job on the shared server.
- **R3, release of artifacts.** Decide whether to release the CC-NET pool, the CC-NET weights or the classifier
  checkpoints. Releasing any of them needs an institutional review of privacy risk (non-DP generator trained on patient
  images) and of the license terms. Without the pool, Real+CC-NET remains only partially reproducible.
- **R4, real-data bootstrap check.** The bootstrap uses filename-derived BCDR patient IDs. A real-data check of the BCDR
  patient grouping (only 127/394 confirmed) would strengthen the BCDR CIs.
