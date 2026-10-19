"""Ball and player signals: nearest player, ball motion, pitch bounds, and team activity.

These are the main evidence for possession and for dead ball, especially on the ~41% of graded
frames where the ball itself is not tracked: ``team_motion`` works from player speed alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rondo.contracts import MatchData
from rondo.registry import register_signal

MOVING_SPEED_MS = 1.5  # m/s threshold for "moving" in team_motion
OUT_OF_BOUNDS_MARGIN_M = 0.5  # metres of slack outside [0, 1] before ball_in_bounds is False
MOTION_HALF_WINDOW = 2  # frames either side of the centred difference for ball_speed_ms


def _frame(match: MatchData) -> pd.DataFrame:
    return pd.DataFrame({"frame": np.arange(match.meta.n_frames, dtype="int64")})


def _full_frames(match: MatchData) -> pd.DataFrame:
    """``match.frames`` reindexed to every frame 0..n-1 (a no-op once C1 is satisfied)."""
    n = match.meta.n_frames
    return match.frames.set_index("frame")[["ball_x", "ball_y"]].reindex(np.arange(n))


@register_signal("ball_nearest")
def ball_nearest(match: MatchData) -> pd.DataFrame:
    """Which team is nearest the ball, its distance, and the other team's nearest distance.

    All three are null when the ball is missing or when neither team has a tracked player that
    frame; a team missing on its own still yields the other team's distance.
    """
    w, h = match.meta.pitch
    n = match.meta.n_frames
    out = _frame(match)

    if match.players.empty:
        out["nearest_team"] = pd.Series([None] * n, dtype="str")
        out["nearest_dist_m"] = np.nan
        out["second_team_dist_m"] = np.nan
        return out

    frames = match.frames[["frame", "ball_x", "ball_y"]]
    joined = match.players.merge(frames, on="frame", how="left")
    dx = (joined["x"] - joined["ball_x"]) * w
    dy = (joined["y"] - joined["ball_y"]) * h
    joined = joined.assign(dist_m=np.hypot(dx, dy))
    per_team = joined.groupby(["frame", "team"])["dist_m"].min().unstack("team")
    per_team = per_team.reindex(index=np.arange(n), columns=["H", "A"])

    dist_h = per_team["H"].to_numpy()
    dist_a = per_team["A"].to_numpy()
    h_valid, a_valid = ~np.isnan(dist_h), ~np.isnan(dist_a)
    both_null = ~h_valid & ~a_valid
    h_nearer = np.where(~h_valid, False, np.where(~a_valid, True, dist_h <= dist_a))

    nearest_dist = np.where(h_nearer, dist_h, dist_a)
    second_dist = np.where(h_nearer, dist_a, dist_h)
    team = np.where(both_null, None, np.where(h_nearer, "H", "A"))

    out["nearest_team"] = pd.Series(team, dtype="str")
    out["nearest_dist_m"] = nearest_dist
    out["second_team_dist_m"] = second_dist
    return out


@register_signal("ball_motion")
def ball_motion(match: MatchData) -> pd.DataFrame:
    """Ball speed (centred difference over ``MOTION_HALF_WINDOW`` frames either side, m/s) and
    the number of consecutive missing frames up to and including this one."""
    w, h = match.meta.pitch
    frames = _full_frames(match)

    x_m = pd.Series(frames["ball_x"].to_numpy() * w)
    y_m = pd.Series(frames["ball_y"].to_numpy() * h)
    k = MOTION_HALF_WINDOW
    dx = (x_m.shift(-k) - x_m.shift(k)).to_numpy()
    dy = (y_m.shift(-k) - y_m.shift(k)).to_numpy()
    speed = np.hypot(dx, dy) * match.meta.fps / (2 * k)

    missing = frames[["ball_x", "ball_y"]].isna().any(axis=1).to_numpy()
    run_group = np.cumsum(~missing)
    run = pd.Series(missing.astype("int64")).groupby(run_group).cumsum().to_numpy()

    out = _frame(match)
    out["ball_speed_ms"] = speed
    out["ball_missing_run"] = run
    return out


@register_signal("ball_bounds")
def ball_bounds(match: MatchData) -> pd.DataFrame:
    """Whether the ball is on the pitch, allowing ``OUT_OF_BOUNDS_MARGIN_M`` of slack; null when
    the ball is missing."""
    w, h = match.meta.pitch
    frames = _full_frames(match)
    bx, by = frames["ball_x"].to_numpy(), frames["ball_y"].to_numpy()
    margin_x, margin_y = OUT_OF_BOUNDS_MARGIN_M / w, OUT_OF_BOUNDS_MARGIN_M / h
    in_bounds = (bx >= -margin_x) & (bx <= 1 + margin_x) & (by >= -margin_y) & (by <= 1 + margin_y)

    values = pd.Series(in_bounds, dtype="boolean")
    values[np.isnan(bx) | np.isnan(by)] = pd.NA

    out = _frame(match)
    out["ball_in_bounds"] = values.to_numpy()
    return out


@register_signal("team_motion")
def team_motion(match: MatchData) -> pd.DataFrame:
    """How many of each team's tracked players are moving faster than ``MOVING_SPEED_MS``, and
    the mean tracked speed of all players that frame (null when nobody is tracked). Does not need
    the ball, so it is the main dead-ball evidence on frames where the ball is not tracked."""
    n = match.meta.n_frames
    out = _frame(match)
    players = match.players

    if players.empty:
        out["team_moving_H"] = 0
        out["team_moving_A"] = 0
        out["team_speed_mean_m_s"] = np.nan
        return out

    moving = players.loc[players["speed"] > MOVING_SPEED_MS]
    counts = (
        moving.groupby(["frame", "team"])
        .size()
        .unstack("team")
        .reindex(index=np.arange(n), columns=["H", "A"])
        .fillna(0)
        .astype("int64")
    )
    mean_speed = players.groupby("frame")["speed"].mean().reindex(np.arange(n))

    out["team_moving_H"] = counts["H"].to_numpy()
    out["team_moving_A"] = counts["A"].to_numpy()
    out["team_speed_mean_m_s"] = mean_speed.to_numpy()
    return out
