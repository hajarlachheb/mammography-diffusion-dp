"""Configuration, documented command-line interfaces, and absence of machine-specific paths or secrets."""
import re
import subprocess
import sys

import pytest

from conftest import REPO

SCRIPTS = ["prepare_data.py", "run_experiment.py", "calculate_privacy.py", "compute_statistics.py",
           "reproduce_tables.py", "reproduce_figures.py"]
MODULES = ["mammo_diffusion_dp.classification.train_classifier", "mammo_diffusion_dp.classification.evaluate_classifier",
           "mammo_diffusion_dp.data.build_training_folders", "mammo_diffusion_dp.data.organize_mcgan"]
FORBIDDEN = re.compile(r"/media/|/home/[a-z]|/mnt/[a-z]|/dev/shm|\.venvs/|BEGIN [A-Z ]*PRIVATE KEY|"
                       r"ghp_[A-Za-z0-9]{20}|hf_[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16}", re.I)
TEXT = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".cff", ".csv", ".tex", ".json", ".cfg", ".example", ""}


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_help(script):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "usage" in r.stdout.lower()


@pytest.mark.parametrize("mod", MODULES)
def test_module_help(mod):
    r = subprocess.run([sys.executable, "-m", mod, "--help"], capture_output=True, text=True,
                       env={"PYTHONPATH": str(REPO / "src"), "PATH": ""})
    assert r.returncode == 0, r.stderr[-500:]


def test_run_experiment_uses_only_configured_paths(monkeypatch):
    env = {"MDPD_WORK_DIR": "/W", "MDPD_DATA_ROOT": "/D", "MDPD_RESULTS_DIR": "/R", "PATH": ""}
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_experiment.py"), "--condition", "m1_matched",
                        "--epsilon", "6", "--all-seeds"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("--batch_size 34") == 3 and "CLEAN_RICHARD_M1_real_bs34" in r.stdout
    assert "/W/clean_training_folders/real_train" in r.stdout and "/D/bcdr/test" in r.stdout


def test_run_experiment_refuses_unrun_combination():
    env = {"MDPD_WORK_DIR": "/W", "MDPD_DATA_ROOT": "/D", "MDPD_RESULTS_DIR": "/R", "PATH": ""}
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_experiment.py"), "--condition", "m1_matched",
                        "--epsilon", "1", "--seed", "42"], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "epsilon [6] only" in (r.stderr + r.stdout)


def test_missing_path_configuration_is_reported():
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_experiment.py"), "--condition", "real_only",
                        "--epsilon", "6", "--seed", "42"], capture_output=True, text=True, env={"PATH": ""})
    assert r.returncode != 0 and "MDPD_" in (r.stderr + r.stdout)


def test_no_machine_paths_or_secrets_in_repository():
    hits = []
    for p in REPO.rglob("*"):
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and p.suffix in TEXT:
            if p.name == "test_paths_and_cli.py":
                continue
            for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                if FORBIDDEN.search(line):
                    hits.append(f"{p.relative_to(REPO)}:{i}")
    assert not hits, hits


def test_no_large_or_binary_payloads():
    banned = {".pth", ".pt", ".ckpt", ".safetensors", ".npz", ".npy", ".dcm", ".nii", ".h5", ".pkl", ".zip"}
    for p in REPO.rglob("*"):
        if p.is_file() and ".git" not in p.parts:
            assert p.suffix.lower() not in banned, p
            assert p.stat().st_size < 5_000_000, p
