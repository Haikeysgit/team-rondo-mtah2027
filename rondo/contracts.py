"""Formats the modules pass to each other (docs/CONTRACTS.md) and the checks that enforce them.

Every ``validate_*`` returns ``None`` when the table is valid and raises ``ContractError`` naming
the first violation otherwise. Extra columns are always allowed; missing ones are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from rondo import kit

__all__ = [
    "ContractError",
    "MatchData",
    "Meta",
    "labels_to_codes",
    "validate_emissions",
    "validate_frames",
    "validate_labels",
    "validate_meta",
    "validate_players",
    "validate_signals",
    "validate_tags",
]

PROB_TOLERANCE = 1e-6
TEAMS = ("H", "A")


class ContractError(ValueError):
    """A table does not follow its contract."""


@dataclass(frozen=True)
class Meta:
    """Match constants. ``half_blocks`` are ``(half, first_frame, last_frame)``, inclusive."""

    n_frames: int
    fps: int
    pitch: tuple[float, float]
    half_blocks: list[tuple[int, int, int]]
    home_name: str
    away_name: str


@dataclass
class MatchData:
    """C1: ``frames`` (one row per frame), ``players`` (per tracked object per frame), ``tags``."""

    frames: pd.DataFrame
    players: pd.DataFrame
    tags: pd.DataFrame
    meta: Meta


# --------------------------------------------------------------------------- helpers


def _fail(message: str):
    raise ContractError(message)


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        _fail(f"{name} is missing required column(s): {', '.join(missing)}")


def _check_dtype(df: pd.DataFrame, column: str, expected: str, name: str) -> None:
    """``expected`` is a dtype string, or ``str`` / ``str?`` (text, the latter allowing nulls)."""
    series = df[column]
    if expected in ("str", "str?"):
        nullable = expected == "str?"
        if nullable and series.isna().all():
            return
        is_text = pd.api.types.is_string_dtype(series.dtype)
        if not (is_text or series.dtype == object):
            _fail(f"{name} column '{column}' must be text, got dtype {series.dtype}")
        if not nullable and series.isna().any():
            _fail(f"{name} column '{column}' must not contain nulls")
        if series.dtype == object:
            bad = [v for v in series if not isinstance(v, str) and not (nullable and pd.isna(v))]
            if bad:
                _fail(f"{name} column '{column}' must be text, found {type(bad[0]).__name__}")
        return
    if str(series.dtype) != expected:
        _fail(f"{name} column '{column}' must have dtype {expected}, got {series.dtype}")


def _check_columns(df: pd.DataFrame, spec: dict[str, str], name: str) -> None:
    _require_columns(df, list(spec), name)
    for column, expected in spec.items():
        _check_dtype(df, column, expected, name)


def _check_finite_or_nan(df: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        if np.isinf(df[column].to_numpy()).any():
            _fail(f"{name} column '{column}' must be finite or NaN (found infinity)")


def _check_frame_column(
    frames: np.ndarray, n_frames: int, name: str, *, every_frame: bool = False
) -> None:
    """Frames are within ``[0, n_frames)``, unique and sorted; optionally exactly ``0..n-1``."""
    if frames.size and (frames.min() < 0 or frames.max() >= n_frames):
        _fail(f"{name} column 'frame' out of range 0..{n_frames - 1}")
    steps = np.diff(frames)
    if (steps == 0).any():
        _fail(f"{name} column 'frame' has duplicate values")
    if (steps < 0).any():
        _fail(f"{name} column 'frame' must be sorted ascending")
    if every_frame and not np.array_equal(frames, np.arange(n_frames)):
        _fail(
            f"{name} column 'frame' must be every frame 0..{n_frames - 1} "
            f"(got {frames.size} rows, one per frame required)"
        )


def _check_values(series: pd.Series, allowed: set[str], name: str, column: str) -> None:
    bad = ~series.isin(allowed)
    if bad.any():
        _fail(
            f"{name} column '{column}' has invalid value {series[bad].iloc[0]!r}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )


def _block_mask(meta: Meta) -> tuple[np.ndarray, np.ndarray]:
    """(half number per frame, 0 outside the blocks; inside-a-block mask)."""
    half = np.zeros(meta.n_frames, dtype=np.int8)
    for number, first, last in meta.half_blocks:
        half[first : last + 1] = number
    return half, half > 0


# --------------------------------------------------------------------------- C1


def validate_meta(meta: Meta) -> None:
    if not isinstance(meta.n_frames, Integral) or isinstance(meta.n_frames, bool):
        _fail("meta n_frames must be an integer")
    if meta.n_frames <= 0:
        _fail("meta n_frames must be positive")
    if not isinstance(meta.fps, Integral) or meta.fps <= 0:
        _fail("meta fps must be a positive integer")
    pitch = meta.pitch
    if len(pitch) != 2 or not all(isinstance(v, Real) and v > 0 for v in pitch):
        _fail("meta pitch must be two positive lengths (length, width) in metres")
    previous_last = -1
    seen_halves: set[int] = set()
    for block in meta.half_blocks:
        if len(block) != 3:
            _fail(f"meta half_blocks entry {block!r} must be (half, first, last)")
        half, first, last = block
        if half not in (1, 2):
            _fail(f"meta half_blocks has invalid half number {half!r} (must be 1 or 2)")
        if half in seen_halves:
            _fail(f"meta half_blocks lists half {half} twice")
        seen_halves.add(half)
        if not (0 <= first <= last < meta.n_frames):
            _fail(f"meta half_blocks entry {block!r} must satisfy 0 <= first <= last < n_frames")
        if first <= previous_last:
            _fail(f"meta half_blocks entry {block!r} overlaps or precedes the previous block")
        previous_last = last
    for field in ("home_name", "away_name"):
        value = getattr(meta, field)
        if not isinstance(value, str) or not value:
            _fail(f"meta {field} must be a non-empty string")


_FRAMES_SPEC = {
    "frame": "int64",
    "half": "Int8",
    "graded": "bool",
    "ball_x": "float64",
    "ball_y": "float64",
}


def validate_frames(df: pd.DataFrame, meta: Meta) -> None:
    _check_columns(df, _FRAMES_SPEC, "frames")
    _check_frame_column(df["frame"].to_numpy(), meta.n_frames, "frames", every_frame=True)
    expected_half, inside = _block_mask(meta)
    if not np.array_equal(df["half"].fillna(0).to_numpy(dtype=np.int8), expected_half):
        _fail("frames column 'half' must be the block's half (1 or 2) inside half blocks, else NA")
    if not np.array_equal(df["graded"].to_numpy(), inside):
        _fail("frames column 'graded' must be True exactly inside the half blocks")
    _check_finite_or_nan(df, ["ball_x", "ball_y"], "frames")


_PLAYERS_SPEC = {
    "frame": "int64",
    "slot": "int16",
    "team": "str",
    "x": "float64",
    "y": "float64",
    "speed": "float64",
}


def validate_players(df: pd.DataFrame, meta: Meta) -> None:
    _check_columns(df, _PLAYERS_SPEC, "players")
    _check_values(df["team"], set(TEAMS), "players", "team")
    frames = df["frame"].to_numpy()
    if frames.size and (frames.min() < 0 or frames.max() >= meta.n_frames):
        _fail(f"players column 'frame' out of range 0..{meta.n_frames - 1}")
    if (np.diff(frames) < 0).any():
        _fail("players column 'frame' must be sorted ascending")
    if df.duplicated(["frame", "slot"]).any():
        _fail("players has duplicate (frame, slot) rows")
    _check_finite_or_nan(df, ["x", "y", "speed"], "players")
    if (df["speed"].to_numpy() < 0).any():
        _fail("players column 'speed' must not be negative")


_TAGS_SPEC = {
    "start_frame": "int64",
    "end_frame": "int64",
    "code": "str",
    "team": "str?",
    "type": "str?",
    "half": "Int8",
}


def validate_tags(df: pd.DataFrame, meta: Meta) -> None:
    _check_columns(df, _TAGS_SPEC, "tags")
    start, end = df["start_frame"], df["end_frame"]
    if (start >= end).any():
        _fail("tags start_frame must be before end_frame (end is exclusive)")
    if (start < 0).any():
        _fail("tags column 'start_frame' must not be negative")
    if (end > meta.n_frames).any():
        _fail(f"tags column 'end_frame' must not exceed n_frames ({meta.n_frames})")
    if (df["code"] != df["code"].str.upper()).any() or (df["code"] == "").any():
        _fail("tags column 'code' must be non-empty and upper case")
    _check_values(df["team"].dropna(), set(TEAMS), "tags", "team")
    if df["half"].isna().any() or not df["half"].isin([1, 2]).all():
        _fail("tags column 'half' must be 1 or 2")


# --------------------------------------------------------------------------- C2


def validate_signals(df: pd.DataFrame, meta: Meta) -> None:
    """One row per frame: ``frame`` equals the frames' ``frame`` column; column names unique."""
    duplicated = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicated:
        _fail(f"signals has duplicate column name(s): {', '.join(map(str, duplicated))}")
    _require_columns(df, ["frame"], "signals")
    _check_dtype(df, "frame", "int64", "signals")
    _check_frame_column(df["frame"].to_numpy(), meta.n_frames, "signals", every_frame=True)


# --------------------------------------------------------------------------- C3, C4, C5


def validate_emissions(df: pd.DataFrame, meta: Meta) -> None:
    """``frame, p_H, p_A, p_D``; frames unique, sorted, in range (not necessarily all)."""
    prob_columns = ["p_H", "p_A", "p_D"]
    _check_columns(df, {"frame": "int64", **{c: "float64" for c in prob_columns}}, "emissions")
    _check_frame_column(df["frame"].to_numpy(), meta.n_frames, "emissions")
    probs = df[prob_columns].to_numpy()
    for i, column in enumerate(prob_columns):
        if not np.isfinite(probs[:, i]).all():
            _fail(f"emissions column '{column}' must be finite")
        if ((probs[:, i] < 0) | (probs[:, i] > 1)).any():
            _fail(f"emissions column '{column}' must be in [0, 1]")
    off = np.abs(probs.sum(axis=1) - 1.0) > PROB_TOLERANCE
    if off.any():
        row = int(np.argmax(off))
        _fail(
            f"emissions rows must sum to 1 (frame {df['frame'].iloc[row]} sums to "
            f"{probs[row].sum():.6f})"
        )


def validate_labels(df: pd.DataFrame, meta: Meta, allow_x: bool) -> None:
    """``frame, label``. Decoder output: ``H/A/D``. Submission-like (``allow_x``): ``H/A/D/X``
    and one row for every frame."""
    _check_columns(df, {"frame": "int64", "label": "str"}, "labels")
    _check_frame_column(df["frame"].to_numpy(), meta.n_frames, "labels", every_frame=allow_x)
    allowed = {"H", "A", "D", "X"} if allow_x else {"H", "A", "D"}
    bad = ~df["label"].isin(allowed)
    if bad.any():
        value = df["label"][bad].iloc[0]
        extra = " (X is only allowed in a submission)" if value == "X" else ""
        _fail(
            f"labels column 'label' has invalid value {value!r}; allowed: "
            f"{', '.join(sorted(allowed))}{extra}"
        )


def labels_to_codes(labels: pd.Series) -> np.ndarray:
    """Labels to int8 codes in the official order (``H=0, A=1, D=2, X=3``)."""
    index = {label: code for code, label in enumerate(kit.pm.LABELS)}
    codes = labels.map(index)
    if codes.isna().any():
        bad = labels[codes.isna()].iloc[0]
        _fail(f"label {bad!r} is not one of {', '.join(kit.pm.LABELS)}")
    return codes.to_numpy(dtype=np.int8)
