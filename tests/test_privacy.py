"""DP-SGD sampling and accounting through the trainer's DataLoader -> Opacus path (CPU, a few seconds)."""
import pytest
import yaml

from mammo_diffusion_dp.privacy.accounting import calibrate_like_trainer, replay_epsilon

from conftest import REPO


@pytest.mark.parametrize("n,bs,q_inv,divisor", [(1104, 128, 9, 122), (4104, 128, 33, 124), (1104, 34, 33, 33)])
def test_poisson_rate_is_one_over_loader_length(n, bs, q_inv, divisor):
    r = calibrate_like_trainer(n, bs, epsilon=6.0)
    assert r["sampler"] == "UniformWithReplacementSampler"
    assert r["sample_rate"] == pytest.approx(1 / q_inv, abs=1e-15)
    assert r["total_steps"] == 300 * q_inv
    assert r["expected_batch_size_divisor"] == divisor


def test_m1_matches_real_ccnet_mechanics():
    m1, b = calibrate_like_trainer(1104, 34, 6.0), calibrate_like_trainer(4104, 128, 6.0)
    assert m1["noise_multiplier"] == b["noise_multiplier"] == 2.34375
    assert m1["sample_rate"] == b["sample_rate"] and m1["total_steps"] == b["total_steps"] == 9900
    assert m1["expected_records_per_step"] == pytest.approx(1104 / 33)


def test_recorded_noise_multipliers_reach_target_epsilon():
    cfg = yaml.safe_load(open(REPO / "configs/privacy/noise_multipliers.yaml"))
    for cond in cfg["conditions"].values():
        steps = 300 * round(1 / cond["sample_rate"])
        for eps, sigma in cond["sigma"].items():
            e = replay_epsilon(sigma, cond["sample_rate"], steps, cfg["delta"])
            assert e <= eps and e == pytest.approx(cond["epsilon_full_run"][eps], abs=2e-5)


def test_epsilon_grows_with_steps():
    assert replay_epsilon(2.34375, 1 / 33, 3300) < replay_epsilon(2.34375, 1 / 33, 9900)
