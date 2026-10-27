"""Metric implementations, bootstrap definition, seed statistics and duplicate-seed detection (artificial data)."""
import csv

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from mammo_diffusion_dp.metrics import bootstrap, statistics


@pytest.fixture
def toy_scores():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.3 * y + rng.normal(0.35, 0.2, 300), 0, 1).round(2)  # rounding creates ties on purpose
    return y, p


def test_weighted_metrics_equal_sklearn_with_unit_weights(toy_scores):
    y, p = toy_scores
    a, apm, apb = bootstrap.all_metrics(p, y, np.ones_like(p))
    assert a == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert apm == pytest.approx(average_precision_score(y, p), abs=1e-12)
    assert apb == pytest.approx(average_precision_score(1 - y, 1 - p), abs=1e-12)


def test_weighted_metrics_equal_sklearn_sample_weight(toy_scores):
    y, p = toy_scores
    w = np.random.default_rng(2).integers(0, 4, len(y)).astype(float)
    assert bootstrap.w_auroc(p, y, w) == pytest.approx(roc_auc_score(y, p, sample_weight=w), abs=1e-12)
    assert bootstrap.w_ap(p, y, w) == pytest.approx(average_precision_score(y, p, sample_weight=w), abs=1e-12)


def test_cluster_weights_keep_patients_together_and_both_classes():
    y = np.array([1, 1, 0, 0, 1, 0])
    pats = np.array(["a", "a", "b", "b", "c", "d"])
    W = bootstrap.cluster_weights(y, pats, n_draws=200, seed=7)
    assert W.shape == (200, 6)
    assert (W[:, 0] == W[:, 1]).all() and (W[:, 2] == W[:, 3]).all()   # all images of a patient share a weight
    assert ((W * y).sum(1) > 0).all() and ((W * (1 - y)).sum(1) > 0).all()
    assert np.array_equal(W, bootstrap.cluster_weights(y, pats, n_draws=200, seed=7))  # deterministic


def test_seed_interval_uses_sample_sd():
    mean, sd, (lo, hi) = statistics.seed_interval([0.5, 0.6, 0.7])
    assert sd == pytest.approx(0.1)            # ddof = 1
    assert hi - mean == pytest.approx(4.302652729911275 * 0.1 / 3 ** 0.5)


def _write_preds(run_dir, y, p, names):
    run_dir.mkdir(parents=True)
    for fname in statistics.PRED_FILES.values():
        with open(run_dir / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "malignant_label", "malignant_probability", "predicted_class_name"])
            w.writeheader()
            for n, yy, pp in zip(names, y, p):
                cls = "is_benign_false" if yy else "is_benign_true"
                w.writerow({"path": f"/x/{cls}/{n}", "malignant_label": int(yy), "malignant_probability": pp,
                            "predicted_class_name": cls})


def test_analyse_end_to_end_on_artificial_runs(tmp_path, toy_scores):
    y, p = toy_scores
    names = [f"patientcbis-ddsm__P_9{i // 2:04d}_TEST{i}.png" for i in range(len(y))]  # 2 images per invented patient
    runs = []
    for cond, shift in (("low", 0.0), ("high", 0.1)):
        for s in (42, 43, 44):
            q = np.clip(p + shift * y + np.random.default_rng(s).normal(0, 0.01, len(p)), 0, 1)
            _write_preds(tmp_path / f"{cond}{s}", y, q, names)
            runs.append({"condition": cond, "seed": s, "run_dir": tmp_path / f"{cond}{s}"})
    per_seed, summary, paired = statistics.analyse(runs, n_draws=50, contrasts=[("high", "low")])
    assert len(per_seed) == 6 and len(summary) == 4
    mean_row = [r for r in paired if r["dataset"] == "cbis" and r["metric"] == "auroc" and r["level"].startswith("mean")][0]
    assert mean_row["difference"] > 0 and mean_row["n_seeds_positive"] == 3
    assert mean_row["test_sampling_ci95_lo"] <= mean_row["difference"] <= mean_row["test_sampling_ci95_hi"]


def test_duplicate_seed_detection():
    a = np.array([0.1, 0.9, 0.4])
    preds = {"seed42": a, "seed43": a.copy(), "seed44": a + 0.01}
    dup = statistics.duplicate_runs(preds)
    assert dup == [("seed42", "seed43", "identical per-image predictions")]
    assert statistics.duplicate_runs({"x": a, "y": a + 1}, {"x": "h", "y": "h"})[0][2] == "identical checkpoint sha256"
