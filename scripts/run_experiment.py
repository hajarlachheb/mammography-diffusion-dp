#!/usr/bin/env python3
"""Print (default) or execute the exact training + evaluation commands of the clean protocol.

    python scripts/run_experiment.py --condition real_only --epsilon 6 --seed 42            # print only
    python scripts/run_experiment.py --condition m1_matched --epsilon 6 --all-seeds --execute

Conditions (configs/experiments/clean_protocol.yaml): real_only (A), real_ccnet (B), m1_matched (C, eps 6 only).
Paths come from $MDPD_WORK_DIR, $MDPD_DATA_ROOT and $MDPD_RESULTS_DIR (or --work_dir/--data_root/--results_dir).
An existing run directory is never overwritten. Training requires a CUDA GPU; each run takes roughly 15-50 min on
one NVIDIA L40S.

Guard kept from the original launchers: train_classifier.py reads cwd-relative config/dp_config.yaml and
config/base_config.yaml when present, and a noise_multiplier there would replace the epsilon calibration. The
commands are therefore run from the repository root and refuse to start if either file exists.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from mammo_diffusion_dp.paths import resolve  # noqa: E402


def commands(cfg, condition, epsilon, seed, work_dir, data_root, results_dir, python=sys.executable):
    c = cfg["conditions"][condition]
    if epsilon not in c["epsilons"]:
        raise SystemExit(f"{condition} was run at epsilon {c['epsilons']} only")
    fill = lambda s: s.format(work_dir=work_dir, data_root=data_root, results_dir=results_dir)
    paths = {k: fill(v) for k, v in cfg["paths"].items()}
    run_id = c["run_id"][epsilon].format(seed=seed)
    out = Path(results_dir) / run_id
    train = [python, str(REPO / cfg["train_script"]), "--experiment_id", run_id,
             "--data_path", paths[c["train_folder"]], "--val_data_path", paths["val"], "--output_dir", str(out),
             *shlex.split(cfg["common_train_args"]), "--batch_size", str(c["batch_size"]), "--seed", str(seed),
             "--epsilon", str(epsilon)]
    evaluate = [python, str(REPO / cfg["eval_script"]), "--checkpoint", str(out / "best_model.pth"),
                "--test_data_cbis", paths["cbis_test"], "--test_data_bcdr", paths["bcdr_test"],
                "--output_file", str(out / "test_metrics.json"), *shlex.split(cfg["eval_args"])]
    return run_id, out, train, evaluate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(REPO / "configs/experiments/clean_protocol.yaml"))
    ap.add_argument("--condition", required=True, choices=["real_only", "real_ccnet", "m1_matched"])
    ap.add_argument("--epsilon", type=int, required=True, choices=[1, 6, 60])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=int)
    g.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--work_dir")
    ap.add_argument("--data_root")
    ap.add_argument("--results_dir")
    ap.add_argument("--execute", action="store_true", help="run the commands (default: print them)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    seeds = cfg["seeds"] if a.all_seeds else [a.seed]
    dirs = [resolve("work_dir", a.work_dir), resolve("data_root", a.data_root), resolve("results_dir", a.results_dir)]
    for c in ("config/dp_config.yaml", "config/base_config.yaml"):
        if (REPO / c).exists():
            raise SystemExit(f"{REPO / c} exists and could override the CLI DP settings; remove it first")
    for seed in seeds:
        run_id, out, train, evaluate = commands(cfg, a.condition, a.epsilon, seed, *dirs)
        print(f"# {run_id}\n{shlex.join(train)}\n{shlex.join(evaluate)}\n")
        if a.execute:
            if out.exists():
                raise SystemExit(f"refusing to overwrite existing run directory {out}")
            subprocess.run(train, check=True, cwd=REPO)
            subprocess.run(evaluate, check=True, cwd=REPO)


if __name__ == "__main__":
    main()
