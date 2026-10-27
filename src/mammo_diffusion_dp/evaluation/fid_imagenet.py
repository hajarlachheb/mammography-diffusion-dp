"""ImageNet-Inception FID (ported for the public release from the project script C1_fid_imagenet.py; changes:
fixed paths replaced by command-line arguments via _configure(); computation, seeds and comparison order unchanged).

Original description: ImageNet-Inception FID with verified implementation, real-vs-real references and subsampling uncertainty.

Requires pytorch-fid 0.3.0 (weights pt_inception-2015-12-05-6726825d.pth, downloaded by pytorch-fid on first use);
the reported values were computed with torch 2.5.1 on CPU.
Usage: python -m mammo_diffusion_dp.evaluation.fid_imagenet --data_root ... --ccnet_dir ... --mcgan_dir ... --out_dir ... --cache_dir ...
Feature extractor: pytorch-fid InceptionV3, pool3 (2048-d), FID-Inception weights; preprocessing exactly as
pytorch-fid's ImagePathDataset (PIL .convert('RGB'), ToTensor) + the model's internal bilinear resize to 299 and
[-1, 1] normalisation. CPU only (16 threads), so shared GPUs are not used.
Two input modes: 'native' (files as stored: real 128x128 L, CC-NET 256x256 RGB, MCGAN 128x128 L) and 'harm128'
(CC-NET first resized to 128x128 with PIL bilinear, so every set enters Inception from the same resolution).
FID is computed on equal-size subsamples (n = 400 per side; FID is biased by n, so only equal-n values are
comparable), R = 20 seeded repeats; real-vs-real halves are patient-disjoint (CBIS P_xxxxx ids). Reported: mean,
SD and 2.5/97.5 percentiles over repeats (subsampling uncertainty only), plus a full-set value.
Controls: FID(X, X) = 0 within 1e-6 * 2tr(Sigma) (relative; see note in main); the feature-based full-set FID equals pytorch_fid.fid_score.calculate_fid_given_paths
end-to-end on the same files (MCGAN vs CBIS train) within 1e-3.
Writes only under --out_dir (features go to --cache_dir; their sha256 is recorded).
"""
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_fid import fid_score
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
import torchvision.transforms as TF
import types
from pytorch_fid import fid_score as _fs
from scipy import linalg as _linalg


# Compatibility shim (math unchanged): pytorch-fid 0.3.0 calls linalg.sqrtm(A, disp=False) and unpacks
# (sqrtm, error estimate); scipy >= 1.16 removed `disp` and returns the matrix only. Callers of the shim get
# exactly scipy's sqrtm; the error estimate (unused by pytorch-fid beyond the unpack) is the Frobenius residual.
def _sqrtm(a, disp=True):
    s = _linalg.sqrtm(a)
    return s if disp else (s, float(np.linalg.norm(s @ s - a) / max(np.linalg.norm(a), 1e-300)))


_fs.linalg = types.SimpleNamespace(**{k: getattr(_linalg, k) for k in dir(_linalg) if not k.startswith("_")})
_fs.linalg.sqrtm = _sqrtm

torch.set_num_threads(8)  # shared machine: keep CPU use bounded (launch with OMP/OPENBLAS/MKL_NUM_THREADS=8)
DATA = None  # set by _configure()
OUT = None
SCR = None
SETS = {"cbis_train": "cbis-ddsm/train", "cbis_val": "cbis-ddsm/val", "cbis_test": "cbis-ddsm/test",
        "bcdr_test": "bcdr/test", "ccnet": "synthetic_ldm_ccnet", "mcgan": "synthetic_mcgan"}
N, R, SEED = 400, 20, 20261029


def files(rel, cls=None):
    cl = [cls] if cls else ["is_benign_false", "is_benign_true"]
    return sorted(p for c in cl for p in (DATA / rel / c).glob("*.png"))


def pid(p):
    m = re.search(r"P_\d{5}", p.name)
    return m.group(0) if m else None


@torch.no_grad()
def extract(paths, harm128=False):
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()
    out, tt = [], TF.ToTensor()
    for i in range(0, len(paths), 50):
        ims = []
        for p in paths[i:i + 50]:
            im = Image.open(p).convert("RGB")
            if harm128 and im.size != (128, 128):
                im = im.resize((128, 128), Image.BILINEAR)
            ims.append(tt(im))
        out.append(model(torch.stack(ims))[0].squeeze(-1).squeeze(-1).numpy())
    return np.concatenate(out).astype(np.float64)


def stats(f):
    return f.mean(0), np.cov(f, rowvar=False)


def fid_lowrank(a, b):
    """Exact FID for n_a, n_b << d. With centred features X_a, X_b, the non-zero eigenvalues of Sigma_a Sigma_b are
    the squared singular values of X_a X_b^T / sqrt((n_a-1)(n_b-1)), so tr sqrt(Sigma_a Sigma_b) is their sum.
    Avoids the 2048x2048 sqrtm (minutes per call on CPU); checked against pytorch-fid in main()."""
    xa, xb = a - a.mean(0), b - b.mean(0)
    sv = np.linalg.svd(xa @ xb.T, compute_uv=False) / np.sqrt((len(a) - 1) * (len(b) - 1))
    tr_a = (xa * xa).sum() / (len(a) - 1)
    tr_b = (xb * xb).sum() / (len(b) - 1)
    return float(((a.mean(0) - b.mean(0)) ** 2).sum() + tr_a + tr_b - 2 * sv.sum())


def fid(a, b):
    return float(calculate_frechet_distance(*stats(a), *stats(b)))


def main():
    feats, meta = {}, {}
    cache = SCR / "inception_pool3_features.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        feats = {k: z[k] for k in z.files if not k.endswith("__paths")}
        meta = {k[:-7]: list(z[k]) for k in z.files if k.endswith("__paths")}
    else:
        for k, rel in SETS.items():
            ps = files(rel)
            feats[k] = extract(ps)
            meta[k] = [str(p) for p in ps]
            print(k, feats[k].shape, flush=True)
        ps = files(SETS["ccnet"])
        feats["ccnet_harm128"] = extract(ps, harm128=True)
        meta["ccnet_harm128"] = [str(p) for p in ps]
        np.savez(cache, **feats, **{k + "__paths": np.array(v) for k, v in meta.items()})
    lab = {k: np.array(["is_benign_true" in p for p in v]) for k, v in meta.items()}  # True = benign

    # ---- controls
    ctrl = {"fid_X_X_cbis_train": fid(feats["cbis_train"], feats["cbis_train"])}
    with tempfile.TemporaryDirectory(dir=SCR) as td:
        for k in ("mcgan", "cbis_train"):
            d = Path(td) / k
            d.mkdir()
            for i, p in enumerate(meta[k]):
                os.symlink(p, d / f"{i:05d}.png")
        e2e = fid_score.calculate_fid_given_paths([str(Path(td) / "mcgan"), str(Path(td) / "cbis_train")], 50, "cpu", 2048, num_workers=4)
    ctrl["fid_full_mcgan_vs_cbis_train_featurebased"] = fid(feats["mcgan"], feats["cbis_train"])
    ctrl["fid_full_mcgan_vs_cbis_train_pytorch_fid_end_to_end"] = float(e2e)
    ctrl["end_to_end_match_1e-3"] = abs(ctrl["fid_full_mcgan_vs_cbis_train_featurebased"] - e2e) < 1e-3
    # Relative tolerance (changed 2026-10-27 after the first run): the absolute 1e-6 test failed with FID(X,X) =
    # -2.5e-5, which is 1.2e-7 of 2*tr(Sigma)=216 for a rank-1103/2048 covariance (sqrtm round-off). Raw value kept.
    ctrl["two_trace_sigma_cbis_train"] = float(2 * np.trace(np.cov(feats["cbis_train"], rowvar=False)))
    ctrl["fid_X_X_zero_rel1e-6"] = abs(ctrl["fid_X_X_cbis_train"]) < 1e-6 * ctrl["two_trace_sigma_cbis_train"]
    print(ctrl, flush=True)
    assert ctrl["fid_X_X_zero_rel1e-6"] and ctrl["end_to_end_match_1e-3"], ctrl
    assert not (OUT / "FID_IMAGENET_RESULTS.csv").exists(), "refusing to overwrite"

    # ---- comparisons on equal-n subsamples
    rng = np.random.default_rng(SEED)

    def sub(k, mask=None, n=N):
        idx = np.flatnonzero(mask) if mask is not None else np.arange(len(feats[k]))
        return feats[k][rng.choice(idx, n, replace=False)]

    def halves_patient(k):
        ps = np.array([pid(Path(p)) for p in meta[k]])
        up = rng.permutation(np.unique(ps))
        a = np.isin(ps, up[: len(up) // 2])
        return a, ~a

    def halves_image(k):
        perm = rng.permutation(len(feats[k]))
        a = np.zeros(len(perm), bool)
        a[perm[: len(perm) // 2]] = True
        return a, ~a

    COMPS = [
        ("real-vs-real", "CBIS train half A vs half B (patient-disjoint)", lambda: (lambda h: (sub("cbis_train", h[0]), sub("cbis_train", h[1])))(halves_patient("cbis_train"))),
        ("real-vs-real", "CBIS train vs CBIS test (patient-disjoint)", lambda: (sub("cbis_train"), sub("cbis_test"))),
        ("real-vs-real (cross-site)", "CBIS train vs BCDR test", lambda: (sub("cbis_train"), sub("bcdr_test"))),
        ("syn-vs-real", "CC-NET vs CBIS train (native)", lambda: (sub("ccnet"), sub("cbis_train"))),
        ("syn-vs-real", "CC-NET vs CBIS train (harm128)", lambda: (sub("ccnet_harm128"), sub("cbis_train"))),
        ("syn-vs-real", "MCGAN vs CBIS train", lambda: (sub("mcgan"), sub("cbis_train"))),
        ("syn-vs-real", "CC-NET vs CBIS test (native)", lambda: (sub("ccnet"), sub("cbis_test"))),
        ("syn-vs-real", "CC-NET vs CBIS test (harm128)", lambda: (sub("ccnet_harm128"), sub("cbis_test"))),
        ("syn-vs-real", "MCGAN vs CBIS test", lambda: (sub("mcgan"), sub("cbis_test"))),
        ("syn-vs-syn", "CC-NET half A vs half B (native)", lambda: (lambda h: (sub("ccnet", h[0]), sub("ccnet", h[1])))(halves_image("ccnet"))),
        ("syn-vs-syn", "MCGAN half A vs half B", lambda: (lambda h: (sub("mcgan", h[0]), sub("mcgan", h[1])))(halves_image("mcgan"))),
        ("syn-vs-syn", "CC-NET (harm128) vs MCGAN", lambda: (sub("ccnet_harm128"), sub("mcgan"))),
        ("per-class syn-vs-real", "CC-NET benign vs CBIS train benign (harm128)", lambda: (sub("ccnet_harm128", lab["ccnet_harm128"]), sub("cbis_train", lab["cbis_train"]))),
        ("per-class syn-vs-real", "CC-NET malignant vs CBIS train malignant (harm128)", lambda: (sub("ccnet_harm128", ~lab["ccnet_harm128"]), sub("cbis_train", ~lab["cbis_train"]))),
        ("per-class syn-vs-real", "MCGAN benign vs CBIS train benign", lambda: (sub("mcgan", lab["mcgan"]), sub("cbis_train", lab["cbis_train"]))),
        ("per-class syn-vs-real", "MCGAN malignant vs CBIS train malignant", lambda: (sub("mcgan", ~lab["mcgan"]), sub("cbis_train", ~lab["cbis_train"]))),
        ("per-class real-vs-real", "CBIS train benign vs CBIS train malignant", lambda: (sub("cbis_train", lab["cbis_train"]), sub("cbis_train", ~lab["cbis_train"]))),
    ]
    rows = []
    # Equivalence check (added 2026-10-27): on the first draw of 5 comparisons spanning every group, the low-rank
    # exact formula must equal pytorch-fid's calculate_frechet_distance within 1e-4 relative.
    CHECK = {0, 3, 9, 12, 16}
    ctrl["lowrank_vs_pytorch_fid"] = []
    for ci, (group, name, draw) in enumerate(COMPS):
        vals = []
        for r in range(R):
            a, b = draw()
            v = fid_lowrank(a, b)
            if r == 0 and ci in CHECK:
                ref = fid(a, b)
                ctrl["lowrank_vs_pytorch_fid"].append({"comparison": name, "lowrank": v, "pytorch_fid": ref,
                                                       "rel_diff": abs(v - ref) / abs(ref)})
                assert abs(v - ref) <= 1e-4 * abs(ref), ctrl["lowrank_vs_pytorch_fid"][-1]
            vals.append(v)
        vals = np.array(vals)
        rows.append({"group": group, "comparison": name, "n_per_side": N, "repeats": R, "fid_mean": vals.mean(),
                     "fid_sd": vals.std(ddof=1), "fid_p2.5": np.percentile(vals, 2.5), "fid_p97.5": np.percentile(vals, 97.5)})
        print(rows[-1], flush=True)
    full = {f"{a} vs {b} (all images)": fid(feats[a], feats[b]) for a, b in
            [("ccnet", "cbis_train"), ("ccnet_harm128", "cbis_train"), ("mcgan", "cbis_train"), ("cbis_test", "cbis_train"), ("bcdr_test", "cbis_train")]}
    with open(OUT / "FID_IMAGENET_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({"controls": ctrl, "full_set_fid_unequal_n": full, "n_images": {k: len(v) for k, v in meta.items()},
               "features_file": str(cache), "features_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
               "subsample_seed": SEED, "n_per_side": N, "repeats": R,
               "extractor": "pytorch-fid 0.3.0 InceptionV3 pool3 2048-d, pt_inception-2015-12-05-6726825d.pth",
               "torch": torch.__version__},
              open(OUT / "FID_IMAGENET_SETTINGS_AND_CONTROLS.json", "w"), indent=1,
              default=lambda v: v.item() if hasattr(v, "item") else str(v))  # numpy scalars


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
