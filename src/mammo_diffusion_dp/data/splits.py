"""Split checks for the patch datasets (dataset16062024 layout of Osuala et al.).

Expected layout under the data root:
    cbis-ddsm/train/{is_benign_false,is_benign_true}/*.png   1,104 images (525 / 579)
    cbis-ddsm/val/...                                          192 images (102 / 90)
    cbis-ddsm/test/...                                         402 images (157 / 245)
    bcdr/test/...                                            1,106 images (486 / 620)
Class folders follow torchvision ImageFolder ordering: index 0 = is_benign_false (malignant),
index 1 = is_benign_true (benign).
"""
import hashlib
import re
from pathlib import Path

CLASS_TO_IDX = {"is_benign_false": 0, "is_benign_true": 1}
EXPECTED_COUNTS = {
    "cbis-ddsm/train": {"is_benign_false": 525, "is_benign_true": 579},
    "cbis-ddsm/val": {"is_benign_false": 102, "is_benign_true": 90},
    "cbis-ddsm/test": {"is_benign_false": 157, "is_benign_true": 245},
    "bcdr/test": {"is_benign_false": 486, "is_benign_true": 620},
}
EXPECTED_SYNTHETIC = {"is_benign_false": 1500, "is_benign_true": 1500}
_CBIS_PID = re.compile(r"P_\d{5}")
_BCDR_PID = re.compile(r"^patientbcdr__(\d+)_image")
_LABEL_IN_NAME = re.compile(r"_is_benign_([01])(?:\.png)?$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def patient_id(path):
    """Patient identifier as encoded in the public file names (CBIS 'P_xxxxx'; BCDR 'patientbcdr__<n>').
    BCDR identifiers are filename-derived and only partly confirmed against upstream metadata."""
    name = Path(path).name
    m = _CBIS_PID.search(name)
    if m:
        return m.group(0)
    m = _BCDR_PID.match(name)
    return f"BCDR_{m.group(1)}" if m else ""


def list_split(root):
    """Return {class_name: sorted list of png paths} for an ImageFolder-style directory."""
    root = Path(root)
    return {c: sorted(p for p in (root / c).glob("*.png")) for c in CLASS_TO_IDX if (root / c).is_dir()}


def class_counts(root):
    return {c: len(v) for c, v in list_split(root).items()}


def label_consistency(root):
    """Files whose name encodes '_is_benign_<0|1>' must sit in the matching class folder.
    Returns a list of mismatching paths (empty = consistent). Names without the suffix are skipped."""
    bad = []
    for cls, paths in list_split(root).items():
        for p in paths:
            m = _LABEL_IN_NAME.search(p.stem + ".png")
            if m and (m.group(1) == "1") != (cls == "is_benign_true"):
                bad.append(str(p))
    return bad


def patient_overlap(split_a, split_b):
    """Patient identifiers present in both splits (non-empty identifiers only)."""
    ids = lambda r: {patient_id(p) for v in list_split(r).values() for p in v} - {""}
    return ids(split_a) & ids(split_b)


def hash_overlap(split_a, split_b):
    h = lambda r: {sha256(p) for v in list_split(r).values() for p in v}
    return h(split_a) & h(split_b)


def validate_data_root(data_root, check_hashes=True):
    """Run all split checks; returns a dict of results and a boolean 'ok'."""
    data_root = Path(data_root)
    out = {"counts": {}, "label_mismatches": {}, "patient_overlap": {}, "hash_overlap": {}}
    ok = True
    for split, exp in EXPECTED_COUNTS.items():
        got = class_counts(data_root / split)
        out["counts"][split] = got
        ok &= got == exp
        mism = label_consistency(data_root / split)
        out["label_mismatches"][split] = len(mism)
        ok &= not mism
    for a, b in (("cbis-ddsm/train", "cbis-ddsm/val"), ("cbis-ddsm/train", "cbis-ddsm/test"), ("cbis-ddsm/val", "cbis-ddsm/test")):
        ov = patient_overlap(data_root / a, data_root / b)
        out["patient_overlap"][f"{a}~{b}"] = len(ov)
        ok &= not ov
    if check_hashes:
        splits = list(EXPECTED_COUNTS)
        for i, a in enumerate(splits):
            for b in splits[i + 1:]:
                ov = hash_overlap(data_root / a, data_root / b)
                out["hash_overlap"][f"{a}~{b}"] = len(ov)
                ok &= not ov
    out["ok"] = bool(ok)
    return out
