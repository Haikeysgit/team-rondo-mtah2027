import numpy as np
import pandas as pd
import pytest

from rondo import pipeline, registry
from rondo.contracts import MatchData, Meta, validate_signals
from rondo.signals import ball as B
from tests.synthetic import Step, make_match

N = 200
PITCH = (105.3, 68.0)


def _meta(n=N) -> Meta:
    return Meta(n, 25, PITCH, [(1, 0, n - 1)], "FC Barcelona", "Athletic Club")


def _frames(n, ball):
    """``ball``: list of ``(x, y)`` or ``None`` per frame, normalised 0..1."""
    bx = np.array([np.nan if b is None else b[0] for b in ball], dtype="float64")
    by = np.array([np.nan if b is None else b[1] for b in ball], dtype="float64")
    return pd.DataFrame(
        {
            "frame": np.arange(n, dtype="int64"),
            "half": pd.array([1] * n, dtype="Int8"),
            "graded": np.ones(n, dtype=bool),
            "ball_x": bx,
            "ball_y": by,
        }
    )


def _players(rows):
    """``rows``: list of ``(frame, slot, team, x, y, speed)``."""
    return pd.DataFrame(
        {
            "frame": pd.Series([r[0] for r in rows], dtype="int64"),
            "slot": pd.Series([r[1] for r in rows], dtype="int16"),
            "team": pd.Series([r[2] for r in rows], dtype="str"),
            "x": pd.Series([r[3] for r in rows], dtype="float64"),
            "y": pd.Series([r[4] for r in rows], dtype="float64"),
            "speed": pd.Series([r[5] for r in rows], dtype="float64"),
        }
    )


_EMPTY_TAGS = pd.DataFrame(columns=["start_frame", "end_frame", "code", "team", "type", "half"])


def match_with(ball, players_rows, n=N) -> MatchData:
    return MatchData(_frames(n, ball), _players(players_rows), _EMPTY_TAGS, _meta(n))


# --------------------------------------------------------------------------- ball_nearest


def test_nearest_team_and_distance_during_dribble():
    steps = [Step("dribble", "H", 2)]
    m, _ = make_match(steps, n_frames=500, start=50)
    out = B.ball_nearest(m)
    validate_signals(out, m.meta)
    mid = 50 + 25  # well inside the dribble, away from either edge
    assert out["nearest_team"][mid] == "H"
    assert out["nearest_dist_m"][mid] < 1.0


def test_distance_uses_metres_scale():
    w, _ = PITCH
    m = match_with(
        ball=[(0.5, 0.5)],
        players_rows=[(0, 0, "H", 0.6, 0.5, 0.0), (0, 11, "A", 0.5, 0.5, 0.0)],
        n=1,
    )
    out = B.ball_nearest(m)
    validate_signals(out, m.meta)
    assert out["nearest_team"][0] == "A"
    assert out["nearest_dist_m"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["second_team_dist_m"][0] == pytest.approx(0.1 * w, abs=1e-6)  # 10.53 m


def test_nearest_null_when_ball_missing():
    m = match_with(
        ball=[None],
        players_rows=[(0, 0, "H", 0.5, 0.5, 0.0), (0, 11, "A", 0.6, 0.5, 0.0)],
        n=1,
    )
    out = B.ball_nearest(m)
    validate_signals(out, m.meta)
    assert pd.isna(out["nearest_team"][0])
    assert pd.isna(out["nearest_dist_m"][0])
    assert pd.isna(out["second_team_dist_m"][0])


def test_nearest_one_team_missing_still_gives_the_other():
    m = match_with(ball=[(0.5, 0.5)], players_rows=[(0, 0, "H", 0.5, 0.5, 0.0)], n=1)
    out = B.ball_nearest(m)
    assert out["nearest_team"][0] == "H"
    assert out["nearest_dist_m"][0] == pytest.approx(0.0, abs=1e-9)
    assert pd.isna(out["second_team_dist_m"][0])


def test_nearest_no_players_at_all():
    m = match_with(ball=[(0.5, 0.5)], players_rows=[], n=1)
    out = B.ball_nearest(m)
    validate_signals(out, m.meta)
    assert pd.isna(out["nearest_team"][0])
    assert pd.isna(out["nearest_dist_m"][0])


# --------------------------------------------------------------------------- ball_motion


def test_pass_speed_is_fast():
    steps = [Step("dribble", "H", 1), Step("pass", "H", 2)]
    m, _ = make_match(steps, n_frames=500, start=50)
    out = B.ball_motion(m)
    validate_signals(out, m.meta)
    pass_start, pass_end = 50 + 25, 50 + 25 + 50
    assert out["ball_speed_ms"][pass_start:pass_end].max() > 8


def test_missing_run_increases_and_resets():
    steps = [Step("dribble", "H", 1), Step("missing", "H", 1), Step("dribble", "H", 1)]
    m, _ = make_match(steps, n_frames=500, start=50)
    out = B.ball_motion(m)
    validate_signals(out, m.meta)
    miss_start = 50 + 25
    run = out["ball_missing_run"]
    assert run[miss_start] == 1
    assert run[miss_start + 10] == 11
    assert run[miss_start - 1] == 0  # last dribble frame, not missing
    after = miss_start + 25
    assert run[after] == 0  # possession resumes, ball visible again


def test_nearest_distances_are_nan_through_a_missing_run():
    steps = [Step("dribble", "H", 1), Step("missing", "H", 1)]
    m, _ = make_match(steps, n_frames=500, start=50)
    out = B.ball_nearest(m)
    validate_signals(out, m.meta)
    miss_start = 50 + 25
    assert out["nearest_dist_m"][miss_start : miss_start + 20].isna().all()
    assert out["nearest_team"][miss_start : miss_start + 20].isna().all()


def test_motion_speed_null_at_window_edges():
    m = match_with(ball=[(0.5, 0.5), (0.5, 0.5)], players_rows=[], n=2)
    out = B.ball_motion(m)
    validate_signals(out, m.meta)
    assert out["ball_speed_ms"].isna().all()  # neither frame has a full +-2 window


# --------------------------------------------------------------------------- ball_bounds


def test_out_leaves_bounds():
    steps = [Step("dribble", "H", 1), Step("out", "H", 2)]
    m, _ = make_match(steps, n_frames=500, start=50)
    out = B.ball_bounds(m)
    validate_signals(out, m.meta)
    last = 50 + 25 + 50 - 1
    assert out["ball_in_bounds"][last] == False


def test_bounds_margin_and_missing():
    w, _ = PITCH
    margin_x = 0.5 / w
    m = match_with(
        ball=[(0.5, 0.5), (1 + margin_x - 1e-6, 0.5), (1 + margin_x + 1e-6, 0.5), None],
        players_rows=[],
        n=4,
    )
    out = B.ball_bounds(m)
    validate_signals(out, m.meta)
    assert out["ball_in_bounds"][0] == True
    assert out["ball_in_bounds"][1] == True
    assert out["ball_in_bounds"][2] == False
    assert pd.isna(out["ball_in_bounds"][3])


# --------------------------------------------------------------------------- team_motion


def test_team_motion_all_slow_gives_mean_and_zero_counts():
    rows = [(0, s, "H" if s < 11 else "A", 0.5, 0.5, 0.2) for s in range(22)]
    m = match_with(ball=[(0.5, 0.5)], players_rows=rows, n=1)
    out = B.team_motion(m)
    validate_signals(out, m.meta)
    assert out["team_moving_H"][0] == 0
    assert out["team_moving_A"][0] == 0
    assert out["team_speed_mean_m_s"][0] == pytest.approx(0.2)


def test_team_motion_counts_fast_players_per_team():
    rows = [(0, s, "H", 0.5, 0.5, 3.0) for s in range(4)]
    m = match_with(ball=[(0.5, 0.5)], players_rows=rows, n=1)
    out = B.team_motion(m)
    validate_signals(out, m.meta)
    assert out["team_moving_H"][0] == 4
    assert out["team_moving_A"][0] == 0


def test_team_motion_no_tracked_players_gives_nan_mean():
    m = match_with(ball=[(0.5, 0.5)], players_rows=[], n=1)
    out = B.team_motion(m)
    validate_signals(out, m.meta)
    assert out["team_moving_H"][0] == 0
    assert out["team_moving_A"][0] == 0
    assert pd.isna(out["team_speed_mean_m_s"][0])


def test_team_motion_works_without_the_ball():
    rows = [(0, 0, "H", 0.5, 0.5, 3.0)]
    m = match_with(ball=[None], players_rows=rows, n=1)
    out = B.team_motion(m)
    validate_signals(out, m.meta)
    assert out["team_moving_H"][0] == 1


# --------------------------------------------------------------------------- shared


def test_registered_via_pipeline():
    pipeline.load_components()
    assert registry.SIGNALS["ball_nearest"] is B.ball_nearest
    assert registry.SIGNALS["ball_motion"] is B.ball_motion
    assert registry.SIGNALS["ball_bounds"] is B.ball_bounds
    assert registry.SIGNALS["team_motion"] is B.team_motion


def test_deterministic():
    steps = [Step("dribble", "H", 1), Step("pass", "H", 1), Step("out", "H", 1)]
    m, _ = make_match(steps, n_frames=300, start=20)
    for fn in (B.ball_nearest, B.ball_motion, B.ball_bounds, B.team_motion):
        a, b = fn(m), fn(m)
        pd.testing.assert_frame_equal(a, b)
