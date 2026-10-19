import pandas as pd

from rondo import kit
from rondo.contracts import labels_to_codes
from rondo.viz.errors import error_moments, format_timestamp


def _labels(*runs: tuple[str, int]) -> pd.Series:
    """A full-length label series from ``[(label, length), ...]`` runs, e.g. ``[("H", 50)]``."""
    values = [label for label, length in runs for _ in range(length)]
    return pd.Series(values, index=pd.RangeIndex(len(values), name="frame"), name="label")


def test_missed_when_truth_switches_but_prediction_never_does():
    truth = _labels(("H", 100), ("A", 100))
    pred = _labels(("H", 200))

    out = error_moments(pred, truth)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["status"] == "missed"
    assert row["frame"] == 100
    assert row["gt_from"] == "H"
    assert row["gt_to"] == "A"
    assert row["pred_from"] is None
    assert row["pred_to"] is None
    assert row["latency"] is None


def test_spurious_excursion_gives_two_rows():
    truth = _labels(("H", 200))
    pred = _labels(("H", 85), ("A", 30), ("H", 85))

    out = error_moments(pred, truth)

    assert len(out) == 2
    assert list(out["status"]) == ["spurious", "spurious"]
    assert list(out["frame"]) == [85, 115]
    assert out.iloc[0]["pred_from"] == "H"
    assert out.iloc[0]["pred_to"] == "A"
    assert out.iloc[1]["pred_from"] == "A"
    assert out.iloc[1]["pred_to"] == "H"
    assert out["gt_from"].isna().all()
    assert out["latency"].isna().all()


def test_late_when_prediction_matches_but_well_past_the_tight_tolerance():
    truth = _labels(("H", 100), ("A", 100))
    pred = _labels(("H", 120), ("A", 80))

    out = error_moments(pred, truth)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["status"] == "late"
    assert row["frame"] == 100  # the ground-truth transition's frame
    assert row["latency"] == 20


def test_close_match_is_not_reported_as_an_error():
    truth = _labels(("H", 100), ("A", 100))
    pred = _labels(("H", 105), ("A", 95))  # 5 frames late: inside LATE_TOLERANCE

    out = error_moments(pred, truth)

    assert out.empty


def test_counts_match_the_official_evaluator():
    truth = _labels(("H", 50), ("A", 50), ("D", 50), ("H", 150))
    pred = _labels(("H", 50), ("A", 150), ("D", 20), ("H", 80))

    out = error_moments(pred, truth)

    official = kit.pm.evaluate(labels_to_codes(truth), labels_to_codes(pred))["transitions"][25]
    assert (out["status"] == "missed").sum() == official["n_missed"]
    assert (out["status"] == "spurious").sum() == official["n_spurious"]


def test_rows_sorted_by_frame():
    truth = _labels(("H", 200))
    pred = _labels(("H", 85), ("A", 30), ("H", 85))

    out = error_moments(pred, truth)

    assert list(out["frame"]) == sorted(out["frame"])


def test_tol_is_configurable():
    truth = _labels(("H", 100), ("A", 100))
    pred = _labels(("H", 120), ("A", 80))  # 20 frames late

    # wide enough to pair the two transitions, so it is one combined "late" row
    assert error_moments(pred, truth, tol=30).iloc[0]["status"] == "late"
    # too tight to pair them at all: the truth's transition is missed and the prediction's is
    # spurious, rather than one combined late row
    out = error_moments(pred, truth, tol=5)
    assert set(out["status"]) == {"missed", "spurious"}


def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(25) == "00:01"  # 25 fps
    assert format_timestamp(25 * 75) == "01:15"
