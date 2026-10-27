"""Dataset split validation, patient overlap, label consistency and duplicate detection (artificial fixtures)."""
import shutil

from mammo_diffusion_dp.data import splits


def test_patient_id_parsing():
    assert splits.patient_id("patientcbis-ddsm__P_99999_image1_mass2_is_benign_0.png") == "P_99999"
    assert splits.patient_id("patientbcdr__77_image3_is_benign_1.png") == "BCDR_77"
    assert splits.patient_id("syn_ldm_benign_0001.png") == ""


def test_clean_fixture_has_no_overlap(fake_split_tree):
    r = fake_split_tree
    assert not splits.patient_overlap(r / "cbis-ddsm/train", r / "cbis-ddsm/val")
    assert not splits.hash_overlap(r / "cbis-ddsm/train", r / "cbis-ddsm/test")
    assert splits.label_consistency(r / "cbis-ddsm/train") == []


def test_patient_overlap_detected(fake_split_tree):
    r = fake_split_tree
    src = next((r / "cbis-ddsm/train/is_benign_true").glob("*P_90001*"))
    shutil.copy(src, r / "cbis-ddsm/val/is_benign_true" / src.name.replace("TEST", "COPY"))
    assert splits.patient_overlap(r / "cbis-ddsm/train", r / "cbis-ddsm/val") == {"P_90001"}
    assert splits.hash_overlap(r / "cbis-ddsm/train", r / "cbis-ddsm/val")  # identical bytes also caught


def test_label_folder_mismatch_detected(fake_split_tree):
    r = fake_split_tree
    src = next((r / "cbis-ddsm/train/is_benign_false").glob("*.png"))
    shutil.move(src, r / "cbis-ddsm/train/is_benign_true" / src.name)
    assert len(splits.label_consistency(r / "cbis-ddsm/train")) == 1


def test_validate_rejects_wrong_counts(fake_split_tree):
    res = splits.validate_data_root(fake_split_tree)
    assert res["ok"] is False  # the fixture is tiny, so the published counts must not match
    assert res["counts"]["cbis-ddsm/train"] == {"is_benign_false": 2, "is_benign_true": 2}


def test_expected_counts_are_the_published_ones():
    total = {k: sum(v.values()) for k, v in splits.EXPECTED_COUNTS.items()}
    assert total == {"cbis-ddsm/train": 1104, "cbis-ddsm/val": 192, "cbis-ddsm/test": 402, "bcdr/test": 1106}
    assert splits.CLASS_TO_IDX == {"is_benign_false": 0, "is_benign_true": 1}
