# B. Real+CC-NET DP

**Setup.** 1,104 real + 3,000 CC-NET patches (1,500 benign / 1,500 malignant) in one ImageFolder and one DP-SGD
mechanism. Batch 128, so q = 1/33 and 9,900 steps. ε ∈ {1, 6, 60}, seeds 42/43/44.

| ε | σ | Full-run ε |
|---|---|---|
| 1 | 10.625 | 0.999 |
| 6 | 2.3438 | 5.992 |
| 60 | 0.6831 | 59.993 |

```bash
python scripts/prepare_data.py build-folders      # needs $MDPD_SYNTHETIC_DIR (the 3,000-image pool)
python scripts/run_experiment.py --condition real_ccnet --epsilon 6 --all-seeds --execute
```

**Run IDs:** `CLEAN_RICHARD_S1_realsyn_swin_refmlp_head_dp_eps6_seed{42,43,44}` and
`CLEAN_RICHARD_S2_realsyn_swin_refmlp_head_dp_eps{1,60}_seed{42,43,44}`.

**The synthetic pool.** The exact pool used for the reported runs is not distributed, and its sampling settings were not
recorded (docs/experiments.md). Results with another CC-NET pool are expected to differ.
