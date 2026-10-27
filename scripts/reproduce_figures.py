#!/usr/bin/env python3
"""Regenerate the publication figures from results/verified_summary/ (no data or GPU needed).

    python scripts/reproduce_figures.py [--out_dir figures/publication]

Writes fig1_pipeline, fig3_privacy_utility_auroc, figS2_privacy_utility_auprc_malignant, figS3_m1_paired_differences
and figS4_m1_conditions (PDF + PNG). Figure 2 and Figure S1 of the manuscript show mammography patches and are not
part of this repository (see docs/datasets.md).
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from mammo_diffusion_dp.visualization import figures  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary_dir", default=str(REPO / "results/verified_summary"))
    ap.add_argument("--out_dir", default=str(REPO / "figures/publication"))
    a = ap.parse_args(argv)
    s, o = Path(a.summary_dir), Path(a.out_dir)
    figures.fig1_pipeline(o / "fig1_pipeline")
    figures.fig_privacy_utility(s / "main_per_seed.csv", s / "main_summary.csv", "auroc", o / "fig3_privacy_utility_auroc")
    figures.fig_privacy_utility(s / "main_per_seed.csv", s / "main_summary.csv", "auprc_malignant", o / "figS2_privacy_utility_auprc_malignant")
    figures.figS3_m1_paired(s / "m1_paired_differences.csv", o / "figS3_m1_paired_differences")
    figures.figS4_m1_conditions(s / "m1_per_seed.csv", s / "m1_summary.csv", s / "m1_paired_differences.csv", o / "figS4_m1_conditions")
    print("figures written to", o)


if __name__ == "__main__":
    main()
