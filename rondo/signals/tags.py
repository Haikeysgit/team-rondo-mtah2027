"""Signals from the Smart Tagging phases: restart windows and the team in possession prior."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from rondo.contracts import MatchData
from rondo.registry import register_signal

# The tag's team is the team performing the event, which is not always the team with the ball.
POSSESSION_SIDE: dict[str, Literal["same", "other", "none"]] = {
    "BUILD UP": "same",
    "PROGRESSION": "same",
    "CREATING CHANCES": "same",
    "ATTACKING TRANSITION": "same",
    "LONG BALL": "same",
    "BALL IN FINAL THIRD": "same",
    "BALL IN THE BOX": "same",
    "PLAYERS IN THE BOX": "same",
    "SET PIECES": "same",
    "DEFENDING IN ATTACKING THIRD": "other",
    "DEFENDING IN MIDDLE THIRD": "other",
    "DEFENDING IN DEFENSIVE THIRD": "other",
    "DEFENSIVE TRANSITION": "other",
    "GOALS": "none",
}

RESTART_CODE = "SET PIECES"
_TEAM_CODE = {"H": 1, "A": 2}
_TEAM_NAME = np.array([None, "H", "A"], dtype=object)


def _normalise(codes: pd.Series) -> pd.Series:
    return codes.astype("object").map(lambda c: str(c).strip().casefold())


_SIDE_BY_KEY = {code.casefold(): side for code, side in POSSESSION_SIDE.items()}


def unknown_tag_codes(tags: pd.DataFrame) -> list[str]:
    """Sorted unique tag codes (stripped, upper case) that ``POSSESSION_SIDE`` does not list."""
    keys = {str(c).strip().casefold() for c in tags["code"].dropna()}
    return sorted(k.upper() for k in keys if k not in _SIDE_BY_KEY)


def _paint(n: int, start, end, length, team) -> np.ndarray:
    """Team code (0 = none) per frame; the shortest covering interval wins, then the later start,
    then the later row. Intervals are painted from lowest to highest priority."""
    out = np.zeros(n, dtype=np.int8)
    order = np.lexsort((np.arange(len(start)), start, -length))  # length desc, start asc, row asc
    for i in order:
        if team[i]:
            out[start[i] : end[i]] = team[i]
    return out


def _intervals(match: MatchData, rows: pd.DataFrame):
    n = match.meta.n_frames
    start = np.clip(rows["start_frame"].to_numpy(dtype=np.int64), 0, n)
    end = np.clip(rows["end_frame"].to_numpy(dtype=np.int64), 0, n)
    length = (rows["end_frame"] - rows["start_frame"]).to_numpy(dtype=np.int64)
    return start, end, length


def _teams(rows: pd.DataFrame) -> np.ndarray:
    return np.array(
        [_TEAM_CODE.get(t, 0) if isinstance(t, str) else 0 for t in rows["team"]], dtype=np.int8
    )


def _as_string(codes: np.ndarray) -> pd.Series:
    values = np.where(codes > 0, _TEAM_NAME[codes], np.nan).astype(object)
    return pd.Series(values, dtype="str")


def _frame(match: MatchData) -> pd.DataFrame:
    return pd.DataFrame({"frame": np.arange(match.meta.n_frames, dtype="int64")})


@register_signal("tag_restarts")
def tag_restarts(match: MatchData) -> pd.DataFrame:
    """``tag_restart_window`` over each ``SET PIECES`` window; ``tag_restart_team`` its team."""
    n = match.meta.n_frames
    tags = match.tags
    rows = tags[_normalise(tags["code"]) == RESTART_CODE.casefold()].reset_index(drop=True)
    start, end, length = _intervals(match, rows)
    window = _paint(n, start, end, length, np.ones(len(rows), dtype=np.int8)) > 0
    team = _paint(n, start, end, length, _teams(rows))
    out = _frame(match)
    out["tag_restart_window"] = window
    out["tag_restart_team"] = _as_string(team)
    return out


@register_signal("tag_team_prior")
def tag_team_prior(match: MatchData) -> pd.DataFrame:
    """Team in possession implied by the shortest covering phase, null where none says."""
    n = match.meta.n_frames
    tags = match.tags
    sides = _normalise(tags["code"]).map(_SIDE_BY_KEY).to_numpy(dtype=object)
    team = _teams(tags)
    flipped = np.where(team == 1, 2, np.where(team == 2, 1, 0)).astype(np.int8)
    possessor = np.where(sides == "same", team, np.where(sides == "other", flipped, 0))
    start, end, length = _intervals(match, tags)
    out = _frame(match)
    out["tag_team_prior"] = _as_string(_paint(n, start, end, length, possessor.astype(np.int8)))
    return out
