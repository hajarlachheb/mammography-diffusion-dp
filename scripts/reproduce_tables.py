#!/usr/bin/env python3
"""Regenerate the LaTeX publication tables from results/verified_summary/ (no data or GPU needed).

    python scripts/reproduce_tables.py [--out_dir results/publication_tables]
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from mammo_diffusion_dp.visualization import tables  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary_dir", default=str(REPO / "results/verified_summary"))
    ap.add_argument("--out_dir", default=str(REPO / "results/publication_tables"))
    a = ap.parse_args(argv)
    s, out = Path(a.summary_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "table1_main_dp_sgd.tex": tables.table1_main(s / "main_summary.csv", s / "main_paired_differences.csv"),
        "table2_m1_ablation.tex": tables.table2_m1(s / "m1_summary.csv", s / "m1_paired_differences.csv"),
        "table3_image_quality.tex": tables.table3_quality(s / "image_quality_fid_frd.csv"),
        "tableS1_osuala_comparison.tex": tables.tableS1_osuala(s / "main_summary.csv", s / "osuala2024_reported_table1.csv"),
    }
    for name, text in files.items():
        (out / name).write_text(text)
        print("wrote", out / name)


if __name__ == "__main__":
    main()
