"""Publication figures from the aggregate CSVs in results/verified_summary/ (observed values only).

Ported from the project scripts that produced the manuscript figures (F1_pipeline_figure.py,
F3_privacy_utility_and_table1.py, F4_m1_paired_differences_supp.py, M1_02_comparison_figure_and_table.py). Drawing
code is unchanged; inputs are the public CSVs and every figure is drawn inside its own rc_context.

Figures that show mammography images (real CBIS-DDSM/BCDR, MCGAN or CC-NET patches) are NOT generated here and NOT
included in this repository; see docs/datasets.md.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY_FILL, DP_GREEN, M1_GREY = "#2a78d6", "#eb6834", "#f3f2ee", "#1f7a4d", "#6b6a65"
DS_NAME = {"cbis": "CBIS-DDSM test (in-domain)", "bcdr": "BCDR test (external)"}
EPS = (1, 6, 60)
BASE_RC = {"font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
           "font.family": "DejaVu Sans"}


def _rd(p):
    return list(csv.DictReader(open(p)))


def _save(fig, stem, bbox=None):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.pdf", bbox_inches=bbox)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches=bbox)
    plt.close(fig)


def fig1_pipeline(stem):
    """Schematic; every number is a fixed protocol setting (no data input)."""
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(7.0, 2.7))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 37)
        ax.axis("off")

        def box(x, y, w, h, title, body, edge, fill="white"):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fill, ec=edge, lw=1.1))
            ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=6.8, weight="bold", color=INK)
            ax.text(x + w / 2, y + h - 4.6, body, ha="center", va="top", fontsize=5.6, color=INK2, linespacing=1.35)

        def arrow(x0, y0, x1, y1, c=INK2, ls="-"):
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=c, lw=1.0, ls=ls))

        box(0.5, 21, 17.5, 13, "CBIS-DDSM train", "1,104 real patches\n578 patients\nprivate records", BLUE)
        box(0.5, 2.5, 17.5, 13, "CC-NET", "latent diffusion,\ntrained without DP\n3,000 synthetic patches", ORANGE)
        ax.add_patch(FancyBboxPatch((22.5, 0.8), 38.5, 35.2, boxstyle="round,pad=0.2,rounding_size=1.5", fc="none", ec=DP_GREEN, lw=1.2, ls=(0, (4, 2.5))))
        ax.text(23.8, 35.2, "(ε, δ)-DP guarantee covers this box only", fontsize=6.0, color=DP_GREEN, weight="bold", va="top")
        box(24.5, 15.6, 34.5, 17.0, "DP-SGD (Opacus, RDP)",
            "Poisson sampling, C = 1, δ = 10⁻⁴, 300 epochs\nReal-only: N = 1,104, bs 128, q = 1/9\nReal+CC-NET: N = 4,104, bs 128, q = 1/33\nM1 (ε = 6 only): N = 1,104, bs 34, q = 1/33\nσ calibrated per arm to ε ∈ {1, 6, 60}", DP_GREEN)
        box(24.5, 2.5, 34.5, 11.5, "Swin-T (ImageNet, frozen)", "head 768 → 384 → 2 (296,066 params)\n224 px, H/V flips, AdamW 1e-5", INK2)
        arrow(18.5, 27.5, 24.1, 25.5)
        arrow(18.5, 9, 24.1, 20.5, c=ORANGE, ls="--")
        ax.text(9.25, 18.2, "Real+CC-NET arm only", fontsize=5.6, color=ORANGE, ha="center", va="center")
        arrow(41.75, 15.3, 41.75, 14.4)
        box(64, 12.0, 16.5, 22.0, "Selection", "official val\n192 img, 102 pat.\npatient-disjoint\nmax val AUPRC\nin ε-gate\n[0.75, 1.05]·ε\nnot covered by DP", INK2, GREY_FILL)
        box(83.5, 12.0, 16, 22.0, "Evaluation", "CBIS-DDSM test\n402 img, 212 pat.\nBCDR test\n1,106 img, 394 pat.\n(external)", INK2, GREY_FILL)
        ax.text(91.5, 9.6, "AUROC, AUPRC\npatient-bootstrap CIs\n+ seed SD, 3 seeds", ha="center", va="top", fontsize=5.6, color=INK2)
        arrow(59.4, 24.6, 63.6, 24.6)
        arrow(80.9, 24.6, 83.1, 24.6)
        _save(fig, stem, bbox="tight")


def fig_privacy_utility(per_seed_csv, summary_csv, metric, stem):
    """metric: 'auroc' (main Fig. 3) or 'auprc_malignant' (supplementary Fig. S2)."""
    ARM_NAME = {"real": "Real-only DP", "realsyn": "Real+CC-NET DP"}
    COL, MARK = {"real": BLUE, "realsyn": ORANGE}, {"real": "o", "realsyn": "s"}
    ylab = {"auroc": "Malignant-positive AUROC", "auprc_malignant": "Malignant-positive AUPRC"}[metric]
    chance = {"auroc": {"cbis": 0.5, "bcdr": 0.5}, "auprc_malignant": {"cbis": 157 / 402, "bcdr": 486 / 1106}}[metric]
    S = {(int(r["epsilon_target"]), r["arm"], r["dataset"]): r for r in _rd(summary_csv)}
    seeds = {}
    for r in _rd(per_seed_csv):
        seeds.setdefault((int(r["epsilon_target"]), r["arm"]), []).append(r)
    with plt.rc_context(BASE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.75), sharey=True)
        for ax, ds in zip(axes, ("cbis", "bcdr")):
            for arm, dx, sx in (("real", 0.95, 0.86), ("realsyn", 1.05, 1.16)):
                xs = [e * dx for e in EPS]
                means = [float(S[(e, arm, ds)][f"{metric}_mean"]) for e in EPS]
                ax.plot(xs, means, color=COL[arm], lw=1.8, zorder=2)
                ax.scatter(xs, means, color=COL[arm], marker=MARK[arm], s=46, edgecolor="white", linewidth=1.2, zorder=4,
                           label=f"{ARM_NAME[arm]}: mean of 3 seeds")
                for e in EPS:
                    v = [float(r[f"{ds}_{metric}"]) for r in seeds[(e, arm)]]
                    ax.scatter([e * sx] * 3, v, facecolor="white", edgecolor=COL[arm], marker=MARK[arm], s=14, linewidth=0.9,
                               zorder=3, label=f"{ARM_NAME[arm]}: individual seed" if e == EPS[0] else None)
            ax.axhline(chance[ds], color=INK2, lw=0.7, ls=(0, (1, 2)))
            ax.text(16, chance[ds], "chance" if metric == "auroc" else "malignant prevalence", fontsize=6, color=INK2, va="bottom", ha="center")
            ax.set_xscale("log")
            ax.set_xticks(EPS, [str(e) for e in EPS])
            ax.minorticks_off()
            ax.set_xlim(0.75, 80)
            ax.set_xlabel("Privacy budget ε (δ = 10⁻⁴, full run)")
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_title(DS_NAME[ds], loc="left", fontsize=8.5)
        axes[0].set_ylabel(ylab)
        h, l = axes[0].get_legend_handles_labels()
        order = [l.index(x) for x in (f"{ARM_NAME['real']}: mean of 3 seeds", f"{ARM_NAME['realsyn']}: mean of 3 seeds",
                                      f"{ARM_NAME['real']}: individual seed", f"{ARM_NAME['realsyn']}: individual seed")]
        fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=2, frameon=False, fontsize=6.8,
                   bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        _save(fig, stem)


def figS3_m1_paired(paired_csv, stem):
    """Seed-paired AUROC differences at eps = 6: patient-bootstrap CI (thick) and seed t95 (thin), never merged."""
    contrasts = [("M1 - A", "M1 − A\nDP mechanics\nonly"), ("C - M1", "C − M1\nsynthetic records,\nmatched mechanics"),
                 ("C - A", "C − A\ntotal\n(Stage 1)")]
    COL = {"M1 - A": M1_GREY, "C - M1": ORANGE, "C - A": BLUE}
    rows = [r for r in _rd(paired_csv) if r["metric"] == "auroc"]
    get = lambda c, ds, lvl: next(r for r in rows if r["comparison"].startswith(c) and r["dataset"] == ds and r["level"].startswith(lvl))
    with plt.rc_context({**BASE_RC, "font.size": 7.5}):
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True)
        for ax, ds in zip(axes, ("cbis", "bcdr")):
            ax.axhline(0, color=INK2, lw=0.8)
            for i, (c, lab) in enumerate(contrasts):
                mean = get(c, ds, "mean")
                m = float(mean["difference"])
                blo, bhi = float(mean["test_sampling_ci95_lo"]), float(mean["test_sampling_ci95_hi"])
                tlo, thi = (float(x) for x in mean["seed_t95"].strip("[]").split(","))
                ax.plot([i - 0.09] * 2, [blo, bhi], color=COL[c], lw=4.0, alpha=0.45, solid_capstyle="butt", zorder=2)
                ax.errorbar(i + 0.09, m, yerr=[[m - tlo], [thi - m]], color=COL[c], lw=0.9, capsize=3, fmt="none", zorder=2)
                ax.scatter(i, m, color=COL[c], s=36, zorder=4, edgecolor="white", linewidth=0.8)
                for k, s in enumerate(("42", "43", "44")):
                    v = float(get(c, ds, f"seed {s}")["difference"])
                    ax.scatter(i + 0.24 + 0.05 * k, v, facecolor="white", edgecolor=COL[c], s=14, lw=0.9, zorder=3)
                ax.text(i, -0.118, f"{m:+.3f}\n{mean['n_seeds_positive']}/3 > 0", ha="center", va="bottom", fontsize=6.3, color=INK2)
            ax.set_xticks(range(3), [lab for _, lab in contrasts], fontsize=6.6)
            ax.set_xlim(-0.5, 2.6)
            ax.set_ylim(-0.12, 0.17)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_title(DS_NAME[ds], loc="left", fontsize=8)
        axes[0].set_ylabel("Seed-paired ΔAUROC (ε = 6)")
        handles = [Line2D([], [], color="#888", lw=4, alpha=0.45, label="95% patient-bootstrap CI (test sampling)"),
                   Line2D([], [], color="#888", lw=0.9, marker="_", ms=6, label="seed t95 (training seeds, df = 2)"),
                   Line2D([], [], color="#888", marker="o", ls="none", ms=5, label="mean of 3 seed pairs"),
                   Line2D([], [], color="#888", marker="o", ls="none", mfc="white", ms=4, label="individual seed pair")]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.01))
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        _save(fig, stem)


def figS4_m1_conditions(per_seed_csv, summary_csv, paired_csv, stem):
    ARMS = ("A", "M1", "C")
    XLAB = {"A": "A\nReal-only\nq = 1/9", "M1": "M1\nReal-only\nq = 1/33", "C": "C\nReal+CC-NET\nq = 1/33"}
    COL = {"A": BLUE, "M1": M1_GREY, "C": ORANGE}
    MARK = {"A": "o", "M1": "D", "C": "s"}
    ps = _rd(per_seed_csv)
    S = {(r["condition"], r["dataset"]): r for r in _rd(summary_csv)}
    P = {(r["comparison"][:6].strip(), r["dataset"], r["metric"]): r for r in _rd(paired_csv) if r["level"].startswith("mean")}
    with plt.rc_context(BASE_RC):
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
        for row, (metric, ylab) in enumerate((("auroc", "Malignant-positive AUROC"), ("auprc_malignant", "Malignant-positive AUPRC"))):
            for col, ds in enumerate(("cbis", "bcdr")):
                ax = axes[row, col]
                for s in ("42", "43", "44"):
                    ys = [float(next(r for r in ps if r["condition"] == a and r["seed"] == s)[f"{ds}_{metric}"]) for a in ARMS]
                    ax.plot(range(3), ys, color=GRID, lw=1.0, zorder=1)
                    for i, a in enumerate(ARMS):
                        ax.scatter(i, ys[i], facecolor="white", edgecolor=COL[a], marker=MARK[a], s=22, lw=1.0, zorder=3)
                for i, a in enumerate(ARMS):
                    m = float(S[(a, ds)][f"{metric}_mean"])
                    ax.plot([i - 0.22, i + 0.22], [m, m], color=COL[a], lw=2.4, solid_capstyle="round", zorder=4)
                c_m1 = P[("C - M1", ds, metric)]
                ax.text(0.98, 0.04, f"C − M1: {float(c_m1['difference']):+.3f} ({c_m1['n_seeds_positive']}/3 seeds > 0)",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK2)
                ax.set_xticks(range(3), [XLAB[a] for a in ARMS])
                ax.set_xlim(-0.5, 2.5)
                ax.grid(axis="y", color=GRID, lw=0.6)
                ax.set_axisbelow(True)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if row == 0:
                    ax.set_title(DS_NAME[ds], loc="left", fontsize=8.5)
                if col == 0:
                    ax.set_ylabel(ylab)
        fig.text(0.01, 0.005, "ε = 6 (δ = 10⁻⁴). Bar: mean of 3 independently trained seeds; hollow markers joined by grey lines: "
                 "the same seed across conditions.\nM1 and C share q = 1/33, σ = 2.344 and 9,900 steps; "
                 "M1 contains no synthetic records.", fontsize=6, color=INK2, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        _save(fig, stem)
