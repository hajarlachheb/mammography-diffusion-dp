"""Shared TEST FIXTURES. Everything generated here is artificial (random noise images, invented file names and
patient IDs); none of it is, or is derived from, scientific or clinical data."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _png(path, seed, size=16):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size, size), dtype=np.uint8), mode="L").save(path)


@pytest.fixture
def fake_split_tree(tmp_path):
    """A miniature dataset16062024-style tree with invented patients (TEST FIXTURE, not data)."""
    root = tmp_path / "fake_data"
    seed = 0
    for split, pats in (("cbis-ddsm/train", ["P_90001", "P_90002"]), ("cbis-ddsm/val", ["P_90003"]),
                        ("cbis-ddsm/test", ["P_90004"])):
        for pid in pats:
            for lab, cls in ((0, "is_benign_false"), (1, "is_benign_true")):
                seed += 1
                _png(root / split / cls / f"patientcbis-ddsm__{pid}_imageTEST{seed}_mass{seed}_is_benign_{lab}.png", seed)
    for k in (1, 2):
        for lab, cls in ((0, "is_benign_false"), (1, "is_benign_true")):
            seed += 1
            _png(root / "bcdr/test" / cls / f"patientbcdr__{800 + k}_imageTEST{seed}_is_benign_{lab}.png", seed)
    return root
