"""Frechet Radiomics Distance (ported for the public release from the project script C2_frd_v1.py; changes:
fixed paths replaced by command-line arguments via _configure(); computation, seeds and comparison order unchanged).

Original description: Frechet Radiomics Distance (FRDv1, frd_score 1.0.1) with controls, real-vs-real references and
subsampling uncertainty. Fresh computation; no earlier FRD output is read or reused.

Requires frd-score 1.0.1 (with pyradiomics); no torch needed.
Usage: python -m mammo_diffusion_dp.evaluation.frd_v1 --data_root ... --ccnet_dir ... --mcgan_dir ... --out_dir ... --cache_dir ...
(harmonised CC-NET copies and the feature cache go to --cache_dir)

Pipeline = the package's own: v1 YAML extractor (Original + LoG + Wavelet), full-image mask (corner pixel excluded,
package default), D1 z-score normalisation (D1 = the REAL reference set, always passed first), NaN-row drop,
float32 cast, NaN/Inf-column drop, Frechet distance, FRD = log(d_F^2) (package default convention).
Features are extracted ONCE per image with frd_score.frd.compute_features; for subsampled comparisons
frd_score.frd.compute_features is replaced by a lookup into that cache, so every other step is the package code.
Resolution: every set enters at 128x128 grey. Real CBIS-DDSM/BCDR and MCGAN are stored 128x128; CC-NET (256x256
RGB, three identical channels) is converted to channel 0 and resized to 128 with PIL bilinear (same as FID harm128).
Controls (must pass before any number is reported):
  C1  FRD(X, X) = -inf (d_F^2 = 0) for CBIS train.
  C2  cached/patched FRD equals an unpatched end-to-end frd_score.compute_frd on the same files
      (CBIS val vs CBIS test, all images) within 1e-4.
Comparisons: n = 400 per side, R = 20 seeded repeats, same list as C1_fid_imagenet.py; real halves patient-disjoint.
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import frd_score
from frd_score import frd

DATA = None  # set by _configure()
OUT = None
SCR = None
SETS = {"cbis_train": "cbis-ddsm/train", "cbis_val": "cbis-ddsm/val", "cbis_test": "cbis-ddsm/test",
        "bcdr_test": "bcdr/test", "ccnet": "synthetic_ldm_ccnet", "mcgan": "synthetic_mcgan"}
N, R, SEED, WORKERS = 400, 20, 20261030, 20


def files(rel):
    return sorted(p for c in ("is_benign_false", "is_benign_true") for p in (DATA / rel / c).glob("*.png"))


def pid(p):
    m = re.search(r"P_\d{5}", Path(p).name)
    return m.group(0) if m else None


def harmonise(paths, name):
    """Return 128x128 single-channel paths; files already in that form are used as stored."""
    out, d = [], SCR / f"g128_{name}"
    for p in paths:
        im = Image.open(p)
        if im.size == (128, 128) and im.mode == "L":
            out.append(str(p))
            continue
        a = np.asarray(im)
        if a.ndim == 3:
            assert np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2]), p
            a = a[..., 0]
        q = d / Path(p).parent.name / Path(p).name
        if not q.exists():
            q.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(a).resize((128, 128), Image.BILINEAR).save(q)
        out.append(str(q))
    return out


def main():
    SCR.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "FRD_V1_RESULTS.csv").exists(), "refusing to overwrite"
    src = {k: files(v) for k, v in SETS.items()}
    paths = {k: harmonise(v, k) for k, v in src.items()}
    ext = frd.get_feature_extractor(frd_version="v1")
    cache = SCR / "frd_v1_features.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        names = list(z["__names"])
        feats = {k: z[k] for k in SETS}
    else:
        feats, names = {}, None
        for k in SETS:
            arr, dicts, _ = frd.compute_features(paths[k], ext, num_workers=WORKERS, frd_version="v1")
            assert arr.shape[0] == len(paths[k]), (k, arr.shape)  # no image skipped
            nk = list(dicts[0].keys())
            assert names is None or nk == names
            names = nk
            feats[k] = arr
            print(k, arr.shape, flush=True)
        np.savez(cache, __names=np.array(names), **feats)
    row_of = {k: {p: i for i, p in enumerate(paths[k])} for k in SETS}
    where = {p: (k, i) for k in SETS for p, i in row_of[k].items()}

    real_compute_features = frd.compute_features

    def cached_compute_features(files, feature_extractor, masks=None, resize_size=None, verbose=False,
                                num_workers=None, frd_version="v1"):
        rows = [feats[where[str(f)][0]][where[str(f)][1]] for f in files]
        d = dict(zip(names, rows[0]))
        return np.asarray(rows), [d] * len(rows), [None] * len(rows)

    def frd_v1(a, b):
        frd.compute_features = cached_compute_features
        try:
            return float(frd_score.compute_frd([list(a), list(b)], frd_version="v1", num_workers=1))
        finally:
            frd.compute_features = real_compute_features

    # ---- controls
    ctrl = {"C1_frd_X_X_cbis_train": frd_v1(paths["cbis_train"], paths["cbis_train"])}
    ctrl["C1_pass"] = ctrl["C1_frd_X_X_cbis_train"] == float("-inf")
    ctrl["C2_cached"] = frd_v1(paths["cbis_val"], paths["cbis_test"])
    ctrl["C2_end_to_end_unpatched"] = float(frd_score.compute_frd([list(paths["cbis_val"]), list(paths["cbis_test"])],
                                                                  frd_version="v1", num_workers=WORKERS))
    ctrl["C2_pass"] = abs(ctrl["C2_cached"] - ctrl["C2_end_to_end_unpatched"]) < 1e-4
    print(ctrl, flush=True)
    assert ctrl["C1_pass"] and ctrl["C2_pass"], ctrl

    rng = np.random.default_rng(SEED)
    lab = {k: np.array(["is_benign_true" in str(p) for p in src[k]]) for k in SETS}  # True = benign

    def sub(k, mask=None):
        idx = np.flatnonzero(mask) if mask is not None else np.arange(len(paths[k]))
        return [paths[k][i] for i in rng.choice(idx, N, replace=False)]

    def halves_patient(k):
        ps = np.array([pid(p) for p in src[k]])
        up = rng.permutation(np.unique(ps))
        a = np.isin(ps, up[: len(up) // 2])
        return a, ~a

    def halves_image(k):
        a = np.zeros(len(paths[k]), bool)
        a[rng.permutation(len(a))[: len(a) // 2]] = True
        return a, ~a

    # (group, name, draw) with the real/reference set first (D1 normalisation base)
    COMPS = [
        ("real-vs-real", "CBIS train half A vs half B (patient-disjoint)", lambda: (lambda h: (sub("cbis_train", h[0]), sub("cbis_train", h[1])))(halves_patient("cbis_train"))),
        ("real-vs-real", "CBIS train vs CBIS test (patient-disjoint)", lambda: (sub("cbis_train"), sub("cbis_test"))),
        ("real-vs-real (cross-site)", "CBIS train vs BCDR test", lambda: (sub("cbis_train"), sub("bcdr_test"))),
        ("syn-vs-real", "CBIS train vs CC-NET (128)", lambda: (sub("cbis_train"), sub("ccnet"))),
        ("syn-vs-real", "CBIS train vs MCGAN", lambda: (sub("cbis_train"), sub("mcgan"))),
        ("syn-vs-real", "CBIS test vs CC-NET (128)", lambda: (sub("cbis_test"), sub("ccnet"))),
        ("syn-vs-real", "CBIS test vs MCGAN", lambda: (sub("cbis_test"), sub("mcgan"))),
        ("syn-vs-syn", "CC-NET half A vs half B", lambda: (lambda h: (sub("ccnet", h[0]), sub("ccnet", h[1])))(halves_image("ccnet"))),
        ("syn-vs-syn", "MCGAN half A vs half B", lambda: (lambda h: (sub("mcgan", h[0]), sub("mcgan", h[1])))(halves_image("mcgan"))),
        ("per-class syn-vs-real", "CBIS train benign vs CC-NET benign", lambda: (sub("cbis_train", lab["cbis_train"]), sub("ccnet", lab["ccnet"]))),
        ("per-class syn-vs-real", "CBIS train malignant vs CC-NET malignant", lambda: (sub("cbis_train", ~lab["cbis_train"]), sub("ccnet", ~lab["ccnet"]))),
        ("per-class syn-vs-real", "CBIS train benign vs MCGAN benign", lambda: (sub("cbis_train", lab["cbis_train"]), sub("mcgan", lab["mcgan"]))),
        ("per-class syn-vs-real", "CBIS train malignant vs MCGAN malignant", lambda: (sub("cbis_train", ~lab["cbis_train"]), sub("mcgan", ~lab["mcgan"]))),
        ("per-class real-vs-real", "CBIS train benign vs CBIS train malignant", lambda: (sub("cbis_train", lab["cbis_train"]), sub("cbis_train", ~lab["cbis_train"]))),
    ]
    rows = []
    for group, name, draw in COMPS:
        vals = np.array([frd_v1(*draw()) for _ in range(R)])
        assert np.isfinite(vals).all(), (name, vals)
        rows.append({"group": group, "comparison": name, "n_per_side": N, "repeats": R, "frd_v1_mean": vals.mean(),
                     "frd_v1_sd": vals.std(ddof=1), "frd_v1_p2.5": np.percentile(vals, 2.5),
                     "frd_v1_p97.5": np.percentile(vals, 97.5)})
        print(rows[-1], flush=True)
    with open(OUT / "FRD_V1_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({"controls": ctrl, "n_images": {k: len(v) for k, v in paths.items()}, "n_features_raw": len(names),
               "features_file": str(cache), "features_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
               "subsample_seed": SEED, "n_per_side": N, "repeats": R,
               "implementation": f"frd_score {frd_score.__version__}, frd_version=v1, norm_ref=d1 (real set first), "
                                 "zscore, FRD=log(d_F^2), full-image mask, all sets 128x128 grey"},
              open(OUT / "FRD_V1_SETTINGS_AND_CONTROLS.json", "w"), indent=1)


def _configure(argv=None):
    """Set the data/output locations (added for the public release; replaces fixed server paths)."""
    import argparse
    global DATA, OUT, SCR
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--data_root", required=True, help="dataset16062024-style root with cbis-ddsm/ and bcdr/")
    ap.add_argument("--ccnet_dir", required=True, help="CC-NET pool (is_benign_false/, is_benign_true/)")
    ap.add_argument("--mcgan_dir", required=True, help="MCGAN pool, e.g. mammo_dp extension/synthetic_data/cbis-ddsm/c-dcgan reorganised into class folders")
    ap.add_argument("--out_dir", required=True, help="where result CSV/JSON are written")
    ap.add_argument("--cache_dir", required=True, help="scratch directory for feature caches")
    a = ap.parse_args(argv)
    DATA, OUT, SCR = Path(a.data_root), Path(a.out_dir), Path(a.cache_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    SCR.mkdir(parents=True, exist_ok=True)
    SETS["ccnet"], SETS["mcgan"] = str(Path(a.ccnet_dir).resolve()), str(Path(a.mcgan_dir).resolve())


if __name__ == "__main__":
    _configure()
    main()
