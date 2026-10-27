# C. M1: matched DP-SGD optimisation control (ε = 6)

**Setup.** Real-only training (1,104 images) with batch 34. The trainer's Poisson rate is then q = 1/⌈1104/34⌉ = 1/33,
the calibrated σ is 2.34375, and there are 9,900 steps: the same sampling rate, noise multiplier, step count and ε
trajectory as Real+CC-NET at ε = 6. Seeds 42/43/44.

```bash
python scripts/calculate_privacy.py calibrate --n 1104 --batch_size 34 --epsilon 6   # q, sigma, steps (CPU)
python scripts/run_experiment.py --condition m1_matched --epsilon 6 --all-seeds --execute
```

**Run IDs:** `CLEAN_RICHARD_M1_real_bs34_swin_refmlp_head_dp_eps6_seed{42,43,44}`.

**Results:** `results/verified_summary/m1_*.csv`. The A and C rows there are the clean Stage 1 runs, evaluated on identical
bootstrap draws.

**What M1 is.** M1 is an *additional optimisation control*:
- **Not the reproduction baseline.** It uses a non-reference batch size.
- **Not a tuned baseline.** No batch-size search was run.
- **Not run at other budgets.** It exists only at ε = 6.

The remaining differences between M1 and Real+CC-NET (gradient normalisation 33 vs 124, Adam coupling, batch
composition, synthetic-image correlations) are listed in docs/experiments.md.
