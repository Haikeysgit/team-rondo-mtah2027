import numpy as np
import pandas as pd
import pytest

from rondo import pipeline, registry
from rondo.contracts import MatchData, Meta, validate_signals
from rondo.signals import tags as T
from tests.synthetic import Step, make_match

N = 2000


def make_tags(rows):
    """rows: (start, end, code, team)."""
    return pd.DataFrame(
        {
            "start_frame": pd.Series([r[0] for r in rows], dtype="int64"),
            "end_frame": pd.Series([r[1] for r in rows], dtype="int64"),
            "code": pd.Series([r[2] for r in rows], dtype="str"),
            "team": pd.Series([r[3] for r in rows], dtype="str"),
            "type": pd.Series([None] * len(rows), dtype="str"),
            "half": pd.array([1] * len(rows), dtype="Int8"),
        }
    )


def match_with(rows, n=N):
    meta = Meta(n, 25, (105.3, 68.0), [(1, 0, n - 1)], "FC Barcelona", "Athletic Club")
    frames = pd.DataFrame(
        {
            "frame": np.arange(n, dtype="int64"),
            "half": pd.array([1] * n, dtype="Int8"),
            "graded": np.ones(n, dtype=bool),
            "ball_x": np.full(n, np.nan),
            "ball_y": np.full(n, np.nan),
        }
    )
    return MatchData(frames, pd.DataFrame(), make_tags(rows), meta)


def restarts(rows, n=N):
    m = match_with(rows, n)
    out = T.tag_restarts(m)
    validate_signals(out, m.meta)
    return out


def prior(rows, n=N):
    m = match_with(rows, n)
    out = T.tag_team_prior(m)
    validate_signals(out, m.meta)
    return out


def test_restart_window_and_team_boundaries():
    out = restarts([(1000, 1125, "SET PIECES", "A")])
    w, t = out["tag_restart_window"], out["tag_restart_team"]
    for f in (1000, 1124):
        assert w[f] and t[f] == "A"
    for f in (999, 1125):
        assert not w[f] and pd.isna(t[f])
    assert w.sum() == 125
    assert w.dtype == bool and out["frame"].dtype == "int64"
    assert pd.api.types.is_string_dtype(t.dtype)


def test_restart_null_team_window_without_team():
    out = restarts([(1000, 1100, "SET PIECES", None)])
    assert out["tag_restart_window"][1000:1100].all()
    assert out["tag_restart_team"].isna().all()


def test_restart_window_clipped_at_edges():
    out = restarts([(0, 10, "SET PIECES", "H"), (N - 10, N, "SET PIECES", "A")])
    assert out["tag_restart_window"][:10].all() and out["tag_restart_window"][N - 10 :].all()
    assert out["tag_restart_team"][0] == "H" and out["tag_restart_team"][N - 1] == "A"
    assert out["tag_restart_window"].sum() == 20
    assert len(out) == N


def test_restart_ignores_other_codes_and_is_case_insensitive():
    out = restarts([(10, 20, "BUILD UP", "H"), (30, 40, " set pieces ", "A")])
    assert not out["tag_restart_window"][10:20].any()
    assert out["tag_restart_window"][30:40].all()
    assert (out["tag_restart_team"][30:40] == "A").all()


def test_overlapping_restarts_shortest_wins_then_later_start():
    out = restarts([(100, 300, "SET PIECES", "H"), (150, 200, "SET PIECES", "A")])
    t = out["tag_restart_team"]
    assert t[120] == "H" and t[160] == "A" and t[250] == "H"
    tie = restarts([(100, 200, "SET PIECES", "H"), (110, 210, "SET PIECES", "A")])
    assert tie["tag_restart_team"][105] == "H"
    assert tie["tag_restart_team"][150] == "A"  # equal length: later start wins
    row_order = restarts([(100, 200, "SET PIECES", "H"), (100, 200, "SET PIECES", "A")])
    assert row_order["tag_restart_team"][150] == "A"  # same start: later row wins


def test_null_team_restart_never_overrides_team():
    out = restarts([(100, 300, "SET PIECES", "H"), (150, 200, "SET PIECES", None)])
    assert out["tag_restart_team"][160] == "H" and out["tag_restart_window"][160]


def test_empty_tags():
    for fn in (restarts, prior):
        out = fn([])
        assert len(out) == N
        assert not out.drop(columns="frame").select_dtypes("bool").any().any()
    assert restarts([])["tag_restart_team"].isna().all()
    assert prior([])["tag_team_prior"].isna().all()


@pytest.mark.parametrize("code", sorted(T.POSSESSION_SIDE))
@pytest.mark.parametrize("team", ["H", "A"])
def test_prior_every_code(code, team):
    side = T.POSSESSION_SIDE[code]
    out = prior([(10, 20, code, team)])["tag_team_prior"]
    other = "A" if team == "H" else "H"
    expected = {"same": team, "other": other, "none": None}[side]
    if expected is None:
        assert out.isna().all()
    else:
        assert (out[10:20] == expected).all() and out.drop(range(10, 20)).isna().all()


def test_prior_examples_from_brief():
    assert prior([(0, 10, "BUILD UP", "H")])["tag_team_prior"][5] == "H"
    assert prior([(0, 10, "DEFENDING IN MIDDLE THIRD", "H")])["tag_team_prior"][5] == "A"
    assert prior([(0, 10, "DEFENSIVE TRANSITION", "A")])["tag_team_prior"][5] == "H"
    assert prior([(0, 10, "GOALS", "H")])["tag_team_prior"].isna().all()
    assert prior([(0, 10, "MYSTERY", "H")])["tag_team_prior"].isna().all()
    lower = prior([(0, 10, " defending in defensive third ", "H")])["tag_team_prior"]
    assert (lower[:10] == "A").all()


def test_prior_null_team_and_pandas_na():
    assert prior([(0, 10, "BUILD UP", None)])["tag_team_prior"].isna().all()


def test_prior_overlap_shortest_then_later_start_and_skips_none():
    out = prior([(0, 100, "BUILD UP", "H"), (20, 40, "DEFENDING IN MIDDLE THIRD", "H")])
    p = out["tag_team_prior"]
    assert p[10] == "H" and p[30] == "A" and p[60] == "H"
    # a shorter GOALS (none) or unknown tag does not hide the covering phase
    hidden = prior([(0, 100, "BUILD UP", "H"), (20, 40, "GOALS", "A"), (50, 60, "ZZZ", "A")])
    assert (hidden["tag_team_prior"][:100] == "H").all()
    tie = prior([(0, 20, "BUILD UP", "H"), (10, 30, "BUILD UP", "A")])["tag_team_prior"]
    assert tie[5] == "H" and tie[15] == "A"


def test_deterministic():
    rows = [(5, 90, "BUILD UP", "H"), (40, 70, "LONG BALL", "A"), (60, 80, "SET PIECES", "A")]
    a, b = prior(rows), prior(rows)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(restarts(rows), restarts(rows))


def test_unknown_tag_codes():
    t = make_tags([(0, 1, "BUILD UP", "H"), (0, 1, "GOAL", "H"), (0, 1, "  zzz ", "H")])
    assert T.unknown_tag_codes(t) == ["GOAL", "ZZZ"]
    assert T.unknown_tag_codes(make_tags([])) == []


def test_registered_via_pipeline():
    pipeline.load_components()
    assert registry.SIGNALS["tag_restarts"] is T.tag_restarts
    assert registry.SIGNALS["tag_team_prior"] is T.tag_team_prior


def test_synthetic_restart_inside_generator_window():
    steps = [
        Step("dribble", "H", 3),
        Step("out", "H", 3),
        Step("restart", "A", 2),
        Step("dribble", "A", 2),
    ]
    m, _ = make_match(steps)
    out = T.tag_restarts(m)
    validate_signals(out, m.meta)
    tag = m.tags.iloc[0]
    inside = out.iloc[int(tag.start_frame) : int(tag.end_frame)]
    assert inside["tag_restart_window"].all() and (inside["tag_restart_team"] == "A").all()
    assert out["tag_restart_window"].sum() == tag.end_frame - tag.start_frame
