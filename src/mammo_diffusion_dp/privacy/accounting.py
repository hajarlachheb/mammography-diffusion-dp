"""Privacy accounting helpers that mirror what train_classifier.py does.

The trainer builds a torch DataLoader(batch_size, shuffle=True) and calls Opacus
PrivacyEngine.make_private_with_epsilon(epochs, target_epsilon, target_delta, max_grad_norm) with the RDP accountant.
Opacus then uses Poisson sampling with sample rate q = 1 / len(DataLoader) = 1 / ceil(N / batch_size), takes
len(DataLoader) steps per epoch, calibrates sigma for the full run, and averages the noisy clipped-gradient sum by
expected_batch_size = int(N * q).

`calibrate_like_trainer` runs exactly that code path on a stand-in dataset of the same size (calibration depends only
on N, batch size, epochs, epsilon and delta). `replay_epsilon` recomputes the epsilon spent after a number of steps.
Adapted from the project's preflight scripts (S1_reverify_noise_multipliers.py, M1_00_preflight.py).
"""
import math

import torch
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from torch.utils.data import DataLoader, TensorDataset


def calibrate_like_trainer(n, batch_size, epsilon, delta=1e-4, epochs=300, max_grad_norm=1.0):
    torch.manual_seed(0)
    ds = TensorDataset(torch.zeros(n, 1), torch.zeros(n, dtype=torch.long))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0)
    model = torch.nn.Linear(1, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-8)
    pe = PrivacyEngine(accountant="rdp", secure_mode=False)
    _, opt, dpl = pe.make_private_with_epsilon(module=model, optimizer=opt, data_loader=dl, epochs=epochs,
                                               target_epsilon=epsilon, target_delta=delta, max_grad_norm=max_grad_norm)
    q = dpl.batch_sampler.sample_rate
    spe = len(dpl)
    return {"N": n, "batch_size": batch_size, "sampler": type(dpl.batch_sampler).__name__, "sample_rate": q,
            "steps_per_epoch": spe, "total_steps": epochs * spe, "noise_multiplier": opt.noise_multiplier,
            "expected_batch_size_divisor": opt.expected_batch_size, "expected_records_per_step": n * q,
            "epsilon_full_run": replay_epsilon(opt.noise_multiplier, q, epochs * spe, delta)}


def replay_epsilon(noise_multiplier, sample_rate, steps, delta=1e-4):
    acc = RDPAccountant()
    acc.history = [(noise_multiplier, sample_rate, steps)]
    return acc.get_epsilon(delta=delta)


def steps_per_epoch(n, batch_size):
    return math.ceil(n / batch_size)
