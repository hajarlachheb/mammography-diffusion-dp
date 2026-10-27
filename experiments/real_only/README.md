# A. Real-only DP: reproduction of Osuala et al.'s Real-only arm

**Setup.** 1,104 real CBIS-DDSM patches, batch 128, so q = 1/9 and 2,700 DP-SGD steps. ε ∈ {1, 6, 60}, seeds 42/43/44.

| ε | σ | Full-run ε |
|---|---|---|
| 1 | 20.3125 | 0.999 |
| 6 | 4.3359 | 5.998 |
| 60 | 0.9613 | 59.993 |

```bash
python scripts/run_experiment.py --condition real_only --epsilon 6 --all-seeds --execute
```

**Run IDs:** `CLEAN_RICHARD_S1_real_swin_refmlp_head_dp_eps6_seed{42,43,44}` and
`CLEAN_RICHARD_S2_real_swin_refmlp_head_dp_eps{1,60}_seed{42,43,44}`.

**Results:** `results/verified_summary/main_*.csv` (rows with `arm = real`). All six ε × test-set means agree with Osuala et
al.'s reported values within one SD (`results/publication_tables/tableS1_osuala_comparison.tex`).

**Role in the paper.** This is the reproduction baseline and the reference Real-only arm. M1 (`../m1_matched_dp`) does not
replace it.
