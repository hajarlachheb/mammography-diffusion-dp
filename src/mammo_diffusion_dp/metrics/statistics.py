"""Per-seed metrics, seed summaries and seed-paired differences from evaluate_classifier.py prediction files.

Ported from the project's acceptance/statistics scripts for the clean runs
(analysis/clean_reproduction_stage1_results/scripts/01_stage1_acceptance_and_stats.py and its Stage 2 / M1
derivatives); the computations are the same, run discovery and paths are configurable.

Two kinds of uncertainty, never merged:
  test-sampling   95% patient-cluster bootstrap CI (mammo_diffusion_dp.metrics.bootstrap), for fixed models.
                  For a condition mean, the bootstrap statistic is the mean over seeds within each draw.
  training-seed   sample SD over the n independently trained seeds (ddof=1) and the t-interval
                  mean +/- t(0.975, n-1) * SD / sqrt(n) ("seed t95").
"""
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

from mammo_diffusion_dp.data.splits import patient_id
from mammo_diffusion_dp.metrics.bootstrap import B_DEFAULT, DRAW_SEEDS, all_metrics, cluster_weights, percentile_ci

METRICS = ("auroc", "auprc_malignant", "auprc_benign")
PRED_FILES = {"cbis": "predictions_cbis_ddsm_test_metrics.csv", "bcdr": "predictions_bcdr_test_metrics.csv"}


def load_predictions(path):
    """Read an evaluator prediction CSV; returns (keys, y, p, patients) sorted by (class folder, file name)."""
    rows = list(csv.DictReader(open(path)))
    rows.sort(key=lambda r: Path(r["path"]).parts[-2:])
    keys = [Path(r["path"]).parts[-2:] for r in rows]
    y = np.array([int(r["malignant_label"]) for r in rows])
    p = np.array([float(r["malignant_probability"]) for r in rows])
    return keys, y, p, [patient_id(r["path"]) for r in rows]


def seed_interval(values):
    v = np.asarray(values, dtype=float)
    sd = v.std(ddof=1)
    h = stats.t.ppf(0.975, len(v) - 1) * sd / math.sqrt(len(v))
    return v.mean(), sd, (v.mean() - h, v.mean() + h)


def analyse(runs, n_draws=B_DEFAULT, draw_seeds=DRAW_SEEDS, contrasts=()):
    """runs: list of dicts {condition, seed, run_dir}. contrasts: list of (high_condition, low_condition).
    Returns (per_seed_rows, summary_rows, paired_rows)."""
    conds = list(dict.fromkeys(r["condition"] for r in runs))
    seeds = {c: sorted(int(r["seed"]) for r in runs if r["condition"] == c) for c in conds}
    boot = {}
    meta = {}
    for ds, fname in PRED_FILES.items():
        loaded = {(r["condition"], int(r["seed"])): load_predictions(Path(r["run_dir"]) / fname) for r in runs}
        ref_keys, ref_y, _, ref_pat = next(iter(loaded.values()))
        for k, (keys, y, _, _) in loaded.items():
            assert keys == ref_keys and (y == ref_y).all(), f"test list or labels differ for {k} on {ds}"
        W = cluster_weights(ref_y, ref_pat, n_draws, draw_seeds[ds])
        meta[ds] = {"n_images": len(ref_y), "n_patients": len(set(ref_pat))}
        for k, (_, y, p, _) in loaded.items():
            boot[(k[0], k[1], ds)] = {"point": np.array(all_metrics(p, y, np.ones_like(p))),
                                      "draws": np.array([all_metrics(p, y, w) for w in W])}
    per_seed, summary, paired = [], [], []
    for c in conds:
        for s in seeds[c]:
            row = {"condition": c, "seed": s}
            for ds in PRED_FILES:
                for i, m in enumerate(METRICS):
                    lo, hi = percentile_ci(boot[(c, s, ds)]["draws"][:, i])
                    row |= {f"{ds}_{m}": boot[(c, s, ds)]["point"][i], f"{ds}_{m}_patient_ci95": f"[{lo:.3f}, {hi:.3f}]"}
            per_seed.append(row)
        for ds in PRED_FILES:
            rec = {"condition": c, "dataset": ds, "n_independent_training_runs": len(seeds[c]), **meta[ds]}
            for i, m in enumerate(METRICS):
                mean, sd, (tlo, thi) = seed_interval([boot[(c, s, ds)]["point"][i] for s in seeds[c]])
                lo, hi = percentile_ci(np.mean([boot[(c, s, ds)]["draws"][:, i] for s in seeds[c]], axis=0))
                rec |= {f"{m}_mean": mean, f"{m}_sd_ddof1": sd, f"{m}_test_sampling_ci95": f"[{lo:.3f}, {hi:.3f}]",
                        f"{m}_seed_t95": f"[{tlo:.3f}, {thi:.3f}]"}
            summary.append(rec)
    for hi_c, lo_c in contrasts:
        common = sorted(set(seeds[hi_c]) & set(seeds[lo_c]))
        for ds in PRED_FILES:
            for i, m in enumerate(METRICS):
                diffs = np.array([boot[(hi_c, s, ds)]["point"][i] - boot[(lo_c, s, ds)]["point"][i] for s in common])
                draws = np.array([boot[(hi_c, s, ds)]["draws"][:, i] - boot[(lo_c, s, ds)]["draws"][:, i] for s in common])
                for s, d, dr in zip(common, diffs, draws):
                    lo, hi = percentile_ci(dr)
                    paired.append({"comparison": f"{hi_c} - {lo_c}", "dataset": ds, "metric": m, "level": f"seed {s}",
                                   "difference": d, "test_sampling_ci95_lo": lo, "test_sampling_ci95_hi": hi,
                                   "seed_sd_ddof1": "", "n_seeds_positive": "", "seed_t95": ""})
                lo, hi = percentile_ci(draws.mean(axis=0))
                mean, sd, (tlo, thi) = seed_interval(diffs)
                paired.append({"comparison": f"{hi_c} - {lo_c}", "dataset": ds, "metric": m,
                               "level": f"mean of {len(common)} seed-matched pairs", "difference": mean,
                               "test_sampling_ci95_lo": lo, "test_sampling_ci95_hi": hi, "seed_sd_ddof1": sd,
                               "n_seeds_positive": int((diffs > 0).sum()), "seed_t95": f"[{tlo:.3f}, {thi:.3f}]"})
    return per_seed, summary, paired


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def duplicate_runs(predictions, checkpoint_hashes=None, atol=0.0):
    """Flag 'seeds' that are not independent models: pairs with identical checkpoint hashes or identical per-image
    predictions (an earlier experiment family in this project reported three seeds that were one model).
    predictions: {run_name: 1-D array in a common image order}. Returns a list of (run_a, run_b, reason)."""
    names = sorted(predictions)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if checkpoint_hashes and checkpoint_hashes.get(a) and checkpoint_hashes.get(a) == checkpoint_hashes.get(b):
                out.append((a, b, "identical checkpoint sha256"))
            elif np.allclose(predictions[a], predictions[b], atol=atol, rtol=0):
                out.append((a, b, "identical per-image predictions"))
    return out
