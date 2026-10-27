# Preprocessing

## Patch extraction

**We did not extract the patches.** The 128×128 mass patches and their splits come from Osuala et al.'s
`dataset16062024/` (see docs/datasets.md). The original extraction pipeline is described in their paper and repository.

## Training folders

`scripts/prepare_data.py build-folders` creates **symlinks only**; no image is copied or modified.

| Folder | Contents |
|---|---|
| `real_train/` | the 1,104 CBIS-DDSM training images. Used by Real-only (A) and M1 (C) |
| `realsyn_train/` | the same 1,104 plus the 3,000 synthetic images, in one ImageFolder. Used by Real+CC-NET (B) |
| `manifest.csv` | split, source, class, link, target (relative), SHA-256 and patient ID for every entry |

**The validation set is used in place.** `cbis-ddsm/val` is never linked into a training folder.

**The builder aborts if:**
- a validation or test image (by path or SHA-256) appears in a training folder;
- a training patient appears in validation or test;
- any count differs from the expected one.

## Classifier input pipeline (`train_classifier.py`, `evaluate_classifier.py`)

**Loading.** `torchvision.datasets.ImageFolder` loads each PNG as RGB:
- grey patches are replicated to three channels;
- CC-NET files are already RGB with three identical channels.

**Training transform**, enabled by `--simple_augmentation --normalize_05` (the mammo_dp recipe):
1. `Resize((224, 224))`, squashing without keeping the aspect ratio;
2. random horizontal flip;
3. random vertical flip;
4. `ToTensor`;
5. normalise with mean = std = 0.5.

**Evaluation transform:** the same without the flips.

**One difference between the image sources.** Real patches and MCGAN are 128×128, while CC-NET is 256×256. All are
resized to 224, so CC-NET images are effectively downsampled while real patches are upsampled. This resolution and
sharpness difference is a possible confound (docs/limitations.md).

## Image-quality metrics

Both metrics compare every set at 128×128 grey. CC-NET is converted to channel 0 and resized to 128 with PIL bilinear
interpolation.

| Metric | Implementation | Settings | Subsampling |
|---|---|---|---|
| **FID** | pytorch-fid 0.3.0 | InceptionV3 pool3 (2048-d) | Equal-size subsamples, n = 400 per side, 20 seeded repeats |
| **FRDv1** | frd-score 1.0.1 | v1 radiomics extractor; the real set is the normalisation reference; reported as log d_F² | Same subsampling as FID |
