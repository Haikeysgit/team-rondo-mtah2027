import dataclasses

import pandas as pd
import pytest

from rondo.post import finalize, validate_official, write_submission


def _all(label: str, n: int = 199319) -> pd.DataFrame:
    return pd.DataFrame({"frame": range(n), "label": label})


def test_finalize_rules(full_meta):
    out = finalize(_all("H"), full_meta)
    assert out.loc[0, "label"] == "X" and out.loc[100000, "label"] == "X"
    assert (out.loc[110808:113741, "label"] == "D").all()
    assert out.loc[11185, "label"] == "H"
    assert out.loc[113742, "label"] == "H"
    assert len(out) == 199319


def test_finalize_accepts_subset_and_fills_x(full_meta):
    first = pd.DataFrame({"frame": range(11185, 90652), "label": "A"})
    second = pd.DataFrame({"frame": range(110808, 190198), "label": "H"})
    out = finalize(pd.concat([first, second], ignore_index=True), full_meta)
    assert len(out) == 199319 and out.loc[5, "label"] == "X" and out.loc[190198, "label"] == "X"
    assert out.loc[20000, "label"] == "A"


def test_finalize_raises_on_missing_graded_frame(full_meta):
    lab = _all("H").drop(index=[20000, 20001])
    with pytest.raises(ValueError, match="2 graded"):
        finalize(lab, full_meta)


def test_finalize_does_not_force_d_for_other_sizes(small_meta):
    out = finalize(pd.DataFrame({"frame": range(10), "label": "H"}), small_meta)
    assert (out["label"] == "H").all()


def test_finalize_ignores_frames_outside_blocks(small_meta):
    meta = dataclasses.replace(small_meta, half_blocks=[(1, 2, 5)])
    out = finalize(pd.DataFrame({"frame": range(2, 6), "label": "A"}), meta)
    assert out["label"].tolist() == ["X", "X", "A", "A", "A", "A", "X", "X", "X", "X"]


def test_written_file_passes_official_validator_and_is_lf(tmp_path, full_meta):
    lab = finalize(_all("A"), full_meta)
    p = tmp_path / "sub" / "team-rondo.csv"
    digest = write_submission(lab, p)
    assert validate_official(p) is None
    data = p.read_bytes()
    assert b"\r\n" not in data and data.startswith(b"frame,label\n0,X\n")
    assert digest == write_submission(lab, tmp_path / "again.csv")
    assert len(digest) == 64


def test_validate_official_returns_reason_not_exception(tmp_path, full_meta):
    lab = finalize(_all("A"), full_meta).iloc[:-1]
    short = tmp_path / "short.csv"
    write_submission(lab, short)
    assert "row count" in validate_official(short)
    garbage = tmp_path / "garbage.csv"
    garbage.write_bytes(b"\x00\xffnot,a\ncsv\n")
    assert isinstance(validate_official(garbage), str)
    assert isinstance(validate_official(tmp_path / "missing.csv"), str)
