# Third-party licenses

**What this repository does and does not distribute:**
- It distributes original code under Apache-2.0 (`LICENSE`, `NOTICE`).
- It does **not** vendor or copy third-party source files.
- It does **not** distribute model weights, datasets or images.

## Code this work builds on

| Project | License | Relationship | What is in this repository |
|---|---|---|---|
| [RichardObi/mammo_dp](https://github.com/RichardObi/mammo_dp) | Apache-2.0 (LICENSE file, verified 2026-10-27) | Reference study. Its classifier recipe and reference head are re-implemented in `train_classifier.py` | No copied files. MCGAN file names and hashes only (`configs/experiments/mcgan_pool_name_mapping.csv`) |
| [RichardObi/ccnet](https://github.com/RichardObi/ccnet) | Apache-2.0 (LICENSE file, verified 2026-10-27) | CC-NET training approach followed by `src/mammo_diffusion_dp/generation/` | No copied files beyond what the generation scripts state they are based on. Attribution in `NOTICE` |

**What the authors still need to confirm.** The generation scripts state they are "based on Richard's ccnet
codebase". The authors should confirm how much code, if any, was copied verbatim, and whether any upstream copyright
headers must be retained. Both upstream projects are Apache-2.0, which is compatible with releasing this repository
under Apache-2.0.

## Python dependencies (installed by the user, not redistributed)

| Package | License |
|---|---|
| PyTorch, torchvision | BSD-3-Clause |
| Opacus | Apache-2.0 |
| NumPy, SciPy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| matplotlib | Matplotlib License (PSF-style) |
| Pillow | MIT-CMU (HPND) |
| PyYAML | MIT |
| diffusers, transformers, accelerate (optional) | Apache-2.0 |
| pytorch-fid (optional) | Apache-2.0 |
| frd-score (optional) | Apache-2.0 (package metadata, verified) |
| pytest (dev) | MIT |

**Compatibility shim.** `fid_imagenet.py` contains a small shim that adapts pytorch-fid's call to `scipy.linalg.sqrtm`
to newer SciPy. It wraps, and does not copy, pytorch-fid code.

## Model weights downloaded at run time (not redistributed)

| Weights | Obtained via | Terms |
|---|---|---|
| Swin-T ImageNet-1k (`Swin_T_Weights.IMAGENET1K_V1`) | torchvision | torchvision model weights. Trained on ImageNet, whose terms restrict use to non-commercial research and education |
| Stable Diffusion 2.1-base (CC-NET initialisation), repository `Manojb/stable-diffusion-2-1-base` | Hugging Face | Stability AI SD 2.x weights are under the CreativeML Open RAIL++-M license (use restrictions apply). **Note:** this repository ID is a third-party mirror. Users may prefer the original publisher's repository, and the authors should confirm the mirror's provenance |
| InceptionV3 FID weights `pt_inception-2015-12-05` | pytorch-fid | Distributed with pytorch-fid (Apache-2.0); originally from TensorFlow (Apache-2.0) |
| FRDv1 feature extractor | frd-score / PyRadiomics | BSD-3-Clause (PyRadiomics) |

## Data

The datasets are not covered by this repository's license; see `DATA_REDISTRIBUTION_AUDIT.md`.
