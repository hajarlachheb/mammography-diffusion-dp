#!/usr/bin/env python3
"""Reproduce the DP-SGD calibration of every condition, or replay the privacy spent by a saved run.

    python scripts/calculate_privacy.py calibrate            # all conditions in configs/privacy/noise_multipliers.yaml
    python scripts/calculate_privacy.py calibrate --n 1104 --batch_size 34 --epsilon 6
    python scripts/calculate_privacy.py replay --checkpoint <run>/best_model.pth --n 1104 --batch_size 128

`calibrate` runs the trainer's own DataLoader -> Opacus make_private_with_epsilon path (CPU, seconds) and compares
the result with the recorded noise multipliers. `replay` recomputes epsilon at the selected checkpoint epoch and after
the full run from the checkpoint's stored noise multiplier (checkpoints are not distributed with this repository).
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from mammo_diffusion_dp.privacy.accounting import calibrate_like_trainer, replay_epsilon, steps_per_epoch  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--config", default=str(REPO / "configs/privacy/noise_multipliers.yaml"))
    c.add_argument("--n", type=int)
    c.add_argument("--batch_size", type=int)
    c.add_argument("--epsilon", type=float)
    r = sub.add_parser("replay")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--n", type=int, required=True, help="number of training records (1104 or 4104)")
    r.add_argument("--batch_size", type=int, required=True)
    r.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        if a.n:
            print(json.dumps(calibrate_like_trainer(a.n, a.batch_size, a.epsilon), indent=1))
            return
        cfg = yaml.safe_load(open(a.config))
        ok = True
        for name, cond in cfg["conditions"].items():
            for eps, sigma in cond["sigma"].items():
                res = calibrate_like_trainer(cond["N"], cond["batch_size"], float(eps), cfg["delta"], cfg["epochs"], cfg["max_grad_norm"])
                match = abs(res["noise_multiplier"] - sigma) < 1e-9 and abs(res["sample_rate"] - cond["sample_rate"]) < 1e-12
                ok &= match
                print(f"{name:12s} eps={eps:<3} q=1/{round(1 / res['sample_rate'])} steps={res['total_steps']} "
                      f"sigma={res['noise_multiplier']:.10g} eps_full_run={res['epsilon_full_run']:.5f} "
                      f"divisor={res['expected_batch_size_divisor']} {'MATCH' if match else 'MISMATCH'}")
        sys.exit(0 if ok else 1)
    import torch
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    spe = steps_per_epoch(a.n, a.batch_size)
    q = 1 / spe
    out = {"noise_multiplier": ck["noise_multiplier"], "sample_rate": q, "selected_epoch_0based": ck["epoch"],
           "epsilon_checkpoint_metadata": ck.get("epsilon"),
           "epsilon_replay_selected": replay_epsilon(ck["noise_multiplier"], q, (ck["epoch"] + 1) * spe, ck.get("delta", 1e-4)),
           "epsilon_replay_full_run": replay_epsilon(ck["noise_multiplier"], q, a.epochs * spe, ck.get("delta", 1e-4))}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
