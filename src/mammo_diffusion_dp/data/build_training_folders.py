"""Build the classifier training folders used by every clean experiment (symlinks only; no image copied or modified).

  <out>/real_train/<class>/real__<name>             1,104 = official CBIS-DDSM train (Real-only and M1)
  <out>/realsyn_train/<class>/{real__,syn__}<name>  4,104 = the same 1,104 + 3,000 synthetic images (Real+CC-NET)
  <out>/manifest.csv                                split, source, class, link, target, sha256, patient_id

Validation uses <data_root>/cbis-ddsm/val directly (192 images, never linked into a training folder).

Adapted from the project script that built the folders used in the reported runs
(analysis/clean_reproduction_stage1_preflight/scripts/05_build_stage1_folders.py). Changes: the data root and the
synthetic directory are arguments instead of fixed server paths, paths in manifest.csv can be written relative to the
data root, and expected counts are checked through mammo_diffusion_dp.data.splits. The assertions are unchanged.
Refuses to run if <out> exists.
"""
import argparse
import csv
import os
from pathlib import Path

from torchvision import datasets

from mammo_diffusion_dp.data.splits import CLASS_TO_IDX, patient_id, sha256
from mammo_diffusion_dp.paths import resolve


def build(data_root, synthetic_dir, out, n_real=1104, n_syn=3000, n_val=192):
    data_root, synthetic_dir, out = Path(data_root), Path(synthetic_dir), Path(out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing {out}")
    real = datasets.ImageFolder(data_root / "cbis-ddsm/train")
    syn = datasets.ImageFolder(synthetic_dir)
    held = {k: datasets.ImageFolder(data_root / v) for k, v in
            {"val": "cbis-ddsm/val", "cbis_test": "cbis-ddsm/test", "bcdr_test": "bcdr/test"}.items()}
    assert real.class_to_idx == syn.class_to_idx == CLASS_TO_IDX and all(h.class_to_idx == CLASS_TO_IDX for h in held.values())
    assert len(real) == n_real and len(syn) == n_syn and len(held["val"]) == n_val, (len(real), len(syn), len(held["val"]))
    held_sha = {sha256(p) for h in held.values() for p, _ in h.samples}
    held_paths = {os.path.realpath(p) for h in held.values() for p, _ in h.samples}
    held_pat = {patient_id(p) for k in ("val", "cbis_test") for p, _ in held[k].samples}
    rows = []
    for folder, sources in (("real_train", [("real", real)]), ("realsyn_train", [("real", real), ("syn", syn)])):
        for src, ds in sources:
            for p, c in ds.samples:
                h = sha256(p)
                assert h not in held_sha and os.path.realpath(p) not in held_paths, f"held-out image in training data: {p}"
                if src == "real":
                    assert patient_id(p) and patient_id(p) not in held_pat, f"patient shared with val/test: {p}"
                cls = ds.classes[c]
                dst = out / folder / cls / f"{src}__{Path(p).name}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(os.path.abspath(p), dst)
                rows.append({"split": folder, "source": src, "class": cls, "link": str(dst.relative_to(out)),
                             "target": os.path.relpath(p, data_root) if src == "real" else os.path.relpath(p, synthetic_dir),
                             "sha256": h, "patient_id": patient_id(p) if src == "real" else ""})
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for folder, n in (("real_train", n_real), ("realsyn_train", n_real + n_syn)):
        ds = datasets.ImageFolder(out / folder)
        assert len(ds) == n and ds.class_to_idx == CLASS_TO_IDX, (folder, len(ds))
        counts[folder] = [ds.targets.count(i) for i in (0, 1)]
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_root", help="dataset16062024-style root (or $MDPD_DATA_ROOT)")
    ap.add_argument("--synthetic_dir", help="class-balanced synthetic pool (or $MDPD_SYNTHETIC_DIR)")
    ap.add_argument("--out", help="output folder (default: $MDPD_WORK_DIR/clean_training_folders)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else resolve("work_dir") / "clean_training_folders"
    counts = build(resolve("data_root", a.data_root), resolve("synthetic_dir", a.synthetic_dir), out)
    for k, v in counts.items():
        print(k, sum(v), "[malignant, benign] =", v)


if __name__ == "__main__":
    main()
