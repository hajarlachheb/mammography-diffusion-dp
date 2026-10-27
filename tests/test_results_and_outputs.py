"""Internal consistency of the released aggregate results, and regeneration of the committed tables/figures."""
import csv
import statistics as st
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from conftest import REPO

V = REPO / "results/verified_summary"
M = ("auroc", "auprc_malignant", "auprc_benign")


def rd(name):
    return list(csv.DictReader(open(V / name)))


def test_all_21_runs_accepted_and_distinct():
    acc = rd("run_acceptance.csv")
    assert len(acc) == 21 and all(r["ACCEPTED"] == "True" for r in acc)
    assert all(r["n_acceptance_checks_passed"] == r["n_acceptance_checks"] for r in acc)
    assert len({r["checkpoint_sha256"] for r in acc}) == 21           # no duplicated "seed"
    for r in acc:
        target = float(r["stage"].split("eps")[1])
        assert float(r["epsilon_full_300_epoch_run"]) <= target
        assert 0.75 * target <= float(r["epsilon_at_selected_checkpoint"]) <= 1.05 * target


@pytest.mark.parametrize("per_seed,summary,key", [("main_per_seed.csv", "main_summary.csv", ("epsilon_target", "arm")),
                                                  ("m1_per_seed.csv", "m1_summary.csv", ("condition",))])
def test_summary_mean_and_sample_sd_follow_from_seeds(per_seed, summary, key):
    ps = rd(per_seed)
    for r in rd(summary):
        rows = [p for p in ps if all(p[k] == r[k] for k in key)]
        assert len(rows) == 3 == int(r["n_independent_training_runs"])
        for m in M:
            v = [float(p[f"{r['dataset']}_{m}"]) for p in rows]
            assert float(r[f"{m}_mean"]) == pytest.approx(st.mean(v), abs=1e-12)
            assert float(r[f"{m}_sd_ddof1"]) == pytest.approx(st.stdev(v), abs=1e-12)


def test_paired_differences_follow_from_seeds():
    ps = {(r["epsilon_target"], r["arm"], r["seed"]): r for r in rd("main_per_seed.csv")}
    for r in rd("main_paired_differences.csv"):
        e, ds, m = r["epsilon_target"], r["dataset"], r["metric"]
        d = [float(ps[(e, "realsyn", s)][f"{ds}_{m}"]) - float(ps[(e, "real", s)][f"{ds}_{m}"]) for s in ("42", "43", "44")]
        if r["level"].startswith("mean"):
            assert float(r["difference"]) == pytest.approx(st.mean(d), abs=1e-12)
            assert int(r["n_seeds_positive"]) == sum(x > 0 for x in d)
        else:
            assert float(r["difference"]) == pytest.approx(d[("42", "43", "44").index(r["level"].split()[1])], abs=1e-12)


def test_tables_regenerate_identically(tmp_path):
    subprocess.run([sys.executable, str(REPO / "scripts/reproduce_tables.py"), "--out_dir", str(tmp_path)], check=True)
    for f in (REPO / "results/publication_tables").glob("*.tex"):
        assert (tmp_path / f.name).read_text() == f.read_text(), f.name


def test_figures_regenerate_identically(tmp_path):
    subprocess.run([sys.executable, str(REPO / "scripts/reproduce_figures.py"), "--out_dir", str(tmp_path)], check=True)
    for f in (REPO / "figures/publication").glob("*.png"):
        a = np.asarray(Image.open(f).convert("RGB")).astype(int)
        b = np.asarray(Image.open(tmp_path / f.name).convert("RGB")).astype(int)
        assert a.shape == b.shape and np.abs(a - b).max() <= 2, f.name   # tolerance for font-rasteriser versions
