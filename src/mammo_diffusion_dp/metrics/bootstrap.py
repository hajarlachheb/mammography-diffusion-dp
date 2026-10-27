"""Weighted AUROC / average precision and the patient-cluster bootstrap used for every reported CI.

w_auroc, w_ap and all_metrics are copied unchanged from the project helper used for the reported numbers
(analysis/midl_reviewer_followup_v1/scripts/03_bootstrap_ci.py). With unit weights they equal
sklearn.metrics.roc_auc_score and average_precision_score.

Bootstrap definition (test-set sampling uncertainty for a FIXED trained model):
  * patients are resampled with replacement and all images of a drawn patient are kept (multiplicity = weight);
  * a draw is rejected and redrawn if it contains no positive or no negative image;
  * B = 2,000 draws, numpy.random.default_rng(seed) with seed 20261026 for CBIS-DDSM and 20261027 for BCDR;
  * the same draws are applied to every model on a test set, so differences between models are paired.
This interval says nothing about variability across training seeds; that is reported separately
(see mammo_diffusion_dp.metrics.statistics).
"""
import numpy as np

B_DEFAULT = 2000
DRAW_SEEDS = {"cbis": 20261026, "bcdr": 20261027}


def w_auroc(s, y, w):
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    uniq, start = np.unique(s, return_index=True)
    wp = np.add.reduceat(w * y, start)
    wn = np.add.reduceat(w * (1 - y), start)
    cum_n_below = np.cumsum(wn) - wn
    num = (wp * (cum_n_below + 0.5 * wn)).sum()
    return num / (wp.sum() * wn.sum())


def w_ap(s, y, w):
    order = np.argsort(-s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    uniq_desc, start = np.unique(-s, return_index=True)
    tp = np.cumsum(np.add.reduceat(w * y, start))
    fp = np.cumsum(np.add.reduceat(w * (1 - y), start))
    prec = tp / np.maximum(tp + fp, 1e-300)
    rec = tp / tp[-1]
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def all_metrics(p, y, w):
    """(AUROC, malignant-positive AUPRC, benign-positive AUPRC) for malignant probability p and labels y (1=malignant)."""
    return (w_auroc(p, y, w), w_ap(p, y, w), w_ap(1 - p, 1 - y, w))


def cluster_weights(y, patients, n_draws=B_DEFAULT, seed=20261026):
    """Patient-cluster bootstrap weights (n_draws x n_images), generated exactly as for the reported results.
    y and patients must be in the same (name-sorted) image order for every model being compared."""
    y = np.asarray(y)
    upat, inv = np.unique(np.asarray(patients), return_inverse=True)
    rng = np.random.default_rng(seed)
    W = []
    while len(W) < n_draws:
        w = np.bincount(rng.integers(0, len(upat), len(upat)), minlength=len(upat))[inv].astype(float)
        if (w * y).sum() and (w * (1 - y)).sum():
            W.append(w)
    return np.array(W)


def percentile_ci(a, level=0.95):
    lo = (1 - level) / 2 * 100
    return float(np.percentile(a, lo)), float(np.percentile(a, 100 - lo))
