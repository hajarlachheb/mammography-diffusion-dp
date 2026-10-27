"""Arrange the published MCGAN images of Osuala et al. into ImageFolder class folders with the file names used here.

Source: the 3,000 PNGs in extension/synthetic_data/cbis-ddsm/c-dcgan/ of https://github.com/RichardObi/mammo_dp
(flat directory, names batch_<i>_<j>_{benign,malignant}.png).

The project's original reorganisation script (scripts/reorganize_synthetic_data.py) numbered files in file-system
order, which is not reproducible across machines. This version applies the recorded mapping
(configs/experiments/mcgan_pool_name_mapping.csv: upstream name -> class folder / local name, with git-blob SHA-1 and
SHA-256 of every file) and verifies every hash, so the folder is byte- and name-identical to the one used for the
reported FID/FRDv1 values. Files are copied, never moved.

These images are used only for the image-quality comparison. They were NOT used to train any classifier reported
here: the matched Real+MCGAN DP experiment was not run because the images' generator provenance is unresolved
(see docs/limitations.md).
"""
import argparse
import csv
import shutil
from pathlib import Path

from mammo_diffusion_dp.data.splits import sha256

DEFAULT_MAPPING = Path(__file__).resolve().parents[3] / "configs/experiments/mcgan_pool_name_mapping.csv"


def organize(source, target, mapping=DEFAULT_MAPPING):
    source, target = Path(source), Path(target)
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing {target}")
    rows = list(csv.DictReader(open(mapping)))
    for r in rows:
        src = source / r["upstream_name"]
        if not src.exists():
            raise SystemExit(f"missing upstream file {src}")
        if sha256(src) != r["sha256"]:
            raise SystemExit(f"hash mismatch for {src}: not the published image")
        dst = target / r["class_folder"] / r["local_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return {c: sum(r["class_folder"] == c for r in rows) for c in ("is_benign_false", "is_benign_true")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, help="mammo_dp/extension/synthetic_data/cbis-ddsm/c-dcgan")
    ap.add_argument("--target", required=True, help="output folder (must not exist)")
    ap.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    a = ap.parse_args(argv)
    print(organize(a.source, a.target, a.mapping))


if __name__ == "__main__":
    main()
