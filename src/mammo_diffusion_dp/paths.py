"""Path configuration. Nothing in this repository assumes a particular machine: every location comes from a
command-line argument or one of the environment variables below (see .env.example)."""
import os
from pathlib import Path

ENV = {
    "data_root": "MDPD_DATA_ROOT",          # contains cbis-ddsm/{train,val,test} and bcdr/test (dataset16062024 layout)
    "synthetic_dir": "MDPD_SYNTHETIC_DIR",  # class-balanced synthetic pool (is_benign_false/, is_benign_true/)
    "work_dir": "MDPD_WORK_DIR",            # derived training folders
    "results_dir": "MDPD_RESULTS_DIR",      # run directories (checkpoints, logs, predictions)
}


def resolve(name, value=None, required=True):
    """Return a Path from an explicit value or the matching environment variable."""
    value = value or os.environ.get(ENV[name])
    if not value:
        if required:
            raise SystemExit(f"{name} is not set: pass it explicitly or set ${ENV[name]}")
        return None
    return Path(value).expanduser()
