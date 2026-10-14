import dataclasses

import numpy as np
import pandas as pd
import pytest

from rondo.contracts import (
    ContractError,
    MatchData,
    labels_to_codes,
    validate_emissions,
    validate_frames,
    validate_labels,
    validate_meta,
    validate_players,
    validate_signals,
    validate_tags,
)


def test_emissions_must_sum_to_one(small_meta):
    bad = pd.DataFrame({"frame": [0], "p_H": [0.5], "p_A": [0.5], "p_D": [0.5]})
    with pytest.raises(ContractError, match="sum"):
        validate_emissions(bad, small_meta)


def test_contract_error_is_value_error():
    assert issubclass(ContractError, ValueError)


def test_good_tables_pass(
    small_meta, good_frames, good_players, good_tags, good_emissions, good_labels
):
    assert validate_meta(small_meta) is None
    assert validate_frames(good_frames, small_meta) is None
    assert validate_players(good_players, small_meta) is None
    assert validate_tags(good_tags, small_meta) is None
    assert validate_emissions(good_emissions, small_meta) is None
    assert validate_labels(good_labels, small_meta, allow_x=True) is None
    assert validate_labels(good_labels, small_meta, allow_x=False) is None
    sig = pd.DataFrame({"frame": good_frames["frame"], "speed_h": 1.0})
    assert validate_signals(sig, small_meta) is None


def test_match_data_holds_tables(small_meta, good_frames, good_players, good_tags):
    m = MatchData(good_frames, good_players, good_tags, small_meta)
    assert m.meta is small_meta
    assert dataclasses.is_dataclass(m)


def _run(validator, case, meta, table, *args):
    mutate, match = case
    bad = table.copy()
    result = mutate(bad)
    bad = bad if result is None else result
    with pytest.raises(ContractError, match=match):
        validator(bad, meta, *args)


def _set(col, value):
    def f(df):
        df[col] = value

    return f


def _drop(col):
    return lambda df: df.drop(columns=col)


def _cell(col, i, value):
    def f(df):
        df.loc[i, col] = value

    return f


META_CASES = {
    "zero frames": ({"n_frames": 0}, "n_frames"),
    "bad fps": ({"fps": 0}, "fps"),
    "bad pitch": ({"pitch": (105.3, -1.0)}, "pitch"),
    "block out of range": ({"half_blocks": [(1, 0, 10)]}, "half_blocks"),
    "block reversed": ({"half_blocks": [(1, 5, 2)]}, "half_blocks"),
    "bad half number": ({"half_blocks": [(3, 0, 9)]}, "half"),
    "overlap": ({"half_blocks": [(1, 0, 5), (2, 5, 9)]}, "overlap"),
    "empty name": ({"home_name": ""}, "home_name"),
}


@pytest.mark.parametrize("changes,match", META_CASES.values(), ids=META_CASES.keys())
def test_validate_meta_rejects(small_meta, changes, match):
    with pytest.raises(ContractError, match=match):
        validate_meta(dataclasses.replace(small_meta, **changes))


FRAMES_CASES = {
    "missing column": (_drop("ball_x"), "ball_x"),
    "frame dtype": (lambda d: d.assign(frame=d["frame"].astype(float)), "frame.*int64"),
    "duplicate frame": (_cell("frame", 3, 2), "duplicate"),
    "unsorted frame": (lambda d: d.iloc[::-1].reset_index(drop=True), "sorted"),
    "frame out of range": (_cell("frame", 9, 10), "range"),
    "too few rows": (lambda d: d.iloc[:5], "rows"),
    "half dtype": (lambda d: d.assign(half=d["half"].astype("int64")), "half.*Int8"),
    "half value": (_cell("half", 2, 3), "half"),
    "half missing inside block": (_cell("half", 2, pd.NA), "half"),
    "graded wrong": (_cell("graded", 2, False), "graded"),
    "graded dtype": (lambda d: d.assign(graded=d["graded"].astype(int)), "graded.*bool"),
    "ball dtype": (lambda d: d.assign(ball_x=d["ball_x"].astype("float32")), "ball_x.*float64"),
    "ball inf": (_cell("ball_y", 4, np.inf), "ball_y.*finite"),
}


@pytest.mark.parametrize("case", FRAMES_CASES.values(), ids=FRAMES_CASES.keys())
def test_validate_frames_rejects(small_meta, good_frames, case):
    _run(validate_frames, case, small_meta, good_frames)


def test_validate_frames_allows_nan_ball_and_extra_columns(small_meta, good_frames):
    good_frames.loc[3, "ball_x"] = np.nan
    good_frames["ball_status"] = "alive"
    validate_frames(good_frames, small_meta)


def test_validate_frames_half_outside_block(good_frames):
    from rondo.contracts import Meta

    meta = Meta(10, 25, (105.3, 68.0), [(1, 2, 5)], "H", "A")
    with pytest.raises(ContractError, match="half"):
        validate_frames(good_frames, meta)
    ok = good_frames.copy()
    ok["half"] = pd.array([pd.NA, pd.NA, 1, 1, 1, 1, pd.NA, pd.NA, pd.NA, pd.NA], dtype="Int8")
    ok["graded"] = [False, False, True, True, True, True, False, False, False, False]
    validate_frames(ok, meta)


PLAYERS_CASES = {
    "missing column": (_drop("speed"), "speed"),
    "slot dtype": (lambda d: d.assign(slot=d["slot"].astype("int64")), "slot.*int16"),
    "team value": (_cell("team", 1, "Z"), "team"),
    "frame out of range": (_cell("frame", 19, 10), "range"),
    "frames decrease": (lambda d: d.iloc[::-1].reset_index(drop=True), "sorted"),
    "duplicate slot": (lambda d: d.assign(slot=np.int16(0)), "duplicate"),
    "x inf": (_cell("x", 0, np.inf), "x.*finite"),
    "y inf": (_cell("y", 0, -np.inf), "y.*finite"),
    "speed negative": (_cell("speed", 0, -1.0), "speed.*negative"),
    "speed dtype": (lambda d: d.assign(speed=d["speed"].astype("float32")), "speed.*float64"),
}


@pytest.mark.parametrize("case", PLAYERS_CASES.values(), ids=PLAYERS_CASES.keys())
def test_validate_players_rejects(small_meta, good_players, case):
    _run(validate_players, case, small_meta, good_players)


def test_validate_players_allows_nan(small_meta, good_players):
    good_players.loc[0, ["x", "y", "speed"]] = np.nan
    validate_players(good_players, small_meta)


TAGS_CASES = {
    "missing column": (_drop("code"), "code"),
    "start dtype": (lambda d: d.assign(start_frame=d["start_frame"].astype(float)), "start_frame"),
    "empty window": (_set("end_frame", np.int64(2)), "start_frame.*end_frame"),
    "reversed window": (_set("start_frame", np.int64(7)), "start_frame.*end_frame"),
    "negative start": (_set("start_frame", np.int64(-1)), "start_frame"),
    "end beyond range": (_set("end_frame", np.int64(11)), "end_frame"),
    "lower case code": (_set("code", "set pieces"), "code.*upper"),
    "team value": (_set("team", "Z"), "team"),
    "half value": (_set("half", pd.array([3], dtype="Int8")), "half"),
    "half missing": (_set("half", pd.array([pd.NA], dtype="Int8")), "half"),
    "half dtype": (_set("half", np.array([1], dtype="int64")), "half.*Int8"),
    "type not text": (_set("type", 5), "type"),
}


@pytest.mark.parametrize("case", TAGS_CASES.values(), ids=TAGS_CASES.keys())
def test_validate_tags_rejects(small_meta, good_tags, case):
    _run(validate_tags, case, small_meta, good_tags)


def test_validate_tags_allows_none_team_and_type_and_empty(small_meta, good_tags):
    good_tags["team"] = [None]
    good_tags["type"] = [None]
    validate_tags(good_tags, small_meta)
    validate_tags(good_tags.iloc[:0], small_meta)


def _signals(frames):
    return pd.DataFrame({"frame": frames["frame"], "a": 1.0, "b": 2.0})


SIGNAL_CASES = {
    "no frame column": (_drop("frame"), "frame"),
    "frame mismatch": (_cell("frame", 4, 40), "frame"),
    "frame short": (lambda d: d.iloc[:9], "frame"),
    "frame unsorted": (lambda d: d.iloc[::-1].reset_index(drop=True), "frame"),
    "duplicate columns": (lambda d: d.rename(columns={"b": "a"}), "duplicate column"),
}


@pytest.mark.parametrize("case", SIGNAL_CASES.values(), ids=SIGNAL_CASES.keys())
def test_validate_signals_rejects(small_meta, good_frames, case):
    _run(validate_signals, case, small_meta, _signals(good_frames))


EMISSION_CASES = {
    "missing column": (_drop("p_D"), "p_D"),
    "sum not one": (_set("p_D", 0.5), "sum"),
    "negative": (lambda d: d.assign(p_H=-0.1, p_A=0.6, p_D=0.5), "p_H.*\\[0, 1\\]"),
    "above one": (lambda d: d.assign(p_H=1.2, p_A=-0.1, p_D=-0.1), "\\[0, 1\\]"),
    "nan": (_cell("p_A", 2, np.nan), "p_A.*finite"),
    "dtype": (lambda d: d.assign(p_H=d["p_H"].astype("float32")), "p_H.*float64"),
    "duplicate frame": (_cell("frame", 3, 2), "duplicate"),
    "frame out of range": (_cell("frame", 9, 99), "range"),
}


@pytest.mark.parametrize("case", EMISSION_CASES.values(), ids=EMISSION_CASES.keys())
def test_validate_emissions_rejects(small_meta, good_emissions, case):
    _run(validate_emissions, case, small_meta, good_emissions)


def test_emissions_tolerance(small_meta, good_emissions):
    good_emissions["p_D"] += 5e-7
    validate_emissions(good_emissions, small_meta)


LABEL_CASES = {
    "missing column": (_drop("label"), "label"),
    "bad value": (_cell("label", 3, "Z"), "label"),
    "lower case": (_cell("label", 3, "h"), "label"),
    "frame duplicate": (_cell("frame", 3, 2), "duplicate"),
    "frame dtype": (lambda d: d.assign(frame=d["frame"].astype(float)), "frame.*int64"),
}


@pytest.mark.parametrize("case", LABEL_CASES.values(), ids=LABEL_CASES.keys())
def test_validate_labels_rejects(small_meta, good_labels, case):
    _run(validate_labels, case, small_meta, good_labels, True)


def test_validate_labels_x_only_when_allowed(small_meta, good_labels):
    good_labels.loc[3, "label"] = "X"
    validate_labels(good_labels, small_meta, allow_x=True)
    with pytest.raises(ContractError, match="label.*X"):
        validate_labels(good_labels, small_meta, allow_x=False)


def test_validate_labels_submission_needs_every_frame(small_meta, good_labels):
    with pytest.raises(ContractError, match="rows"):
        validate_labels(good_labels.iloc[:5], small_meta, allow_x=True)
    validate_labels(good_labels.iloc[:5], small_meta, allow_x=False)


def test_labels_to_codes():
    codes = labels_to_codes(pd.Series(["H", "A", "D", "X", "H"]))
    assert codes.dtype == np.int8
    assert codes.tolist() == [0, 1, 2, 3, 0]
    with pytest.raises(ContractError, match="label"):
        labels_to_codes(pd.Series(["H", "Q"]))
