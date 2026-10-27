#!/usr/bin/env python3
"""Data preparation after obtaining the datasets from their official sources (see docs/datasets.md).

    python scripts/prepare_data.py validate      --data_root $MDPD_DATA_ROOT
    python scripts/prepare_data.py build-folders --data_root $MDPD_DATA_ROOT --synthetic_dir $MDPD_SYNTHETIC_DIR --out $MDPD_WORK_DIR/clean_training_folders
    python scripts/prepare_data.py organize-mcgan --source <mammo_dp>/extension/synthetic_data/cbis-ddsm/c-dcgan --target <dir>

validate       split sizes and class counts, label/folder consistency, patient overlap between CBIS train/val/test,
               and exact-duplicate (SHA-256) overlap between all splits.
build-folders  real_train (1,104) and realsyn_train (1,104 + 3,000) symlink folders plus manifest.csv.
organize-mcgan arrange the published MCGAN images (image-quality comparison only) with the recorded file names.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mammo_diffusion_dp.data import build_training_folders, organize_mcgan  # noqa: E402
from mammo_diffusion_dp.data.splits import EXPECTED_SYNTHETIC, class_counts, validate_data_root  # noqa: E402
from mammo_diffusion_dp.paths import resolve  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--data_root")
    v.add_argument("--synthetic_dir", help="also check the synthetic pool is 1,500 / 1,500")
    v.add_argument("--no_hashes", action="store_true", help="skip the SHA-256 overlap check")
    b = sub.add_parser("build-folders")
    b.add_argument("--data_root")
    b.add_argument("--synthetic_dir")
    b.add_argument("--out")
    m = sub.add_parser("organize-mcgan")
    m.add_argument("--source", required=True)
    m.add_argument("--target", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        res = validate_data_root(resolve("data_root", a.data_root), check_hashes=not a.no_hashes)
        if a.synthetic_dir:
            got = class_counts(a.synthetic_dir)
            res["synthetic_counts"] = got
            res["ok"] &= got == EXPECTED_SYNTHETIC
        print(json.dumps(res, indent=1))
        sys.exit(0 if res["ok"] else 1)
    if a.cmd == "build-folders":
        build_training_folders.main([x for k in ("data_root", "synthetic_dir", "out") if getattr(a, k)
                                     for x in (f"--{k}", getattr(a, k))])
    if a.cmd == "organize-mcgan":
        organize_mcgan.main(["--source", a.source, "--target", a.target])


if __name__ == "__main__":
    main()
