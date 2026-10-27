#!/usr/bin/env python3
"""Per-seed metrics, seed summaries, paired differences and patient-bootstrap CIs from your own evaluated runs.

    python scripts/compute_statistics.py --runs runs.csv --contrast real_ccnet:real_only --out_dir my_stats/

runs.csv has columns condition,seed,run_dir; each run_dir must contain the evaluator outputs
predictions_cbis_ddsm_test_metrics.csv and predictions_bcdr_test_metrics.csv (scripts/run_experiment.py).
With the same test images and the default bootstrap seeds, this reproduces the CIs in results/verified_summary/.
Definitions: docs/experiments.md#statistics.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mammo_diffusion_dp.metrics.statistics import analyse, write_csv  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--contrast", action="append", default=[], help="HIGH:LOW condition pair, repeatable")
    ap.add_argument("--n_draws", type=int, default=2000)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    runs = list(csv.DictReader(open(a.runs)))
    per_seed, summary, paired = analyse(runs, n_draws=a.n_draws, contrasts=[tuple(c.split(":")) for c in a.contrast])
    out = Path(a.out_dir)
    write_csv(out / "per_seed.csv", per_seed)
    write_csv(out / "summary.csv", summary)
    if paired:
        write_csv(out / "paired_differences.csv", paired)
    for r in summary:
        print(r["condition"], r["dataset"], f"AUROC {r['auroc_mean']:.3f} ± {r['auroc_sd_ddof1']:.3f}")


if __name__ == "__main__":
    main()
