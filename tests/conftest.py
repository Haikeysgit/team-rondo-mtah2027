"""Shared fixtures: a tiny valid match whose tables each test may mutate freely."""

import numpy as np
import pandas as pd
import pytest

from rondo import registry
from rondo.contracts import MatchData, Meta
from tests.io.fixtures import write_bundle


@pytest.fixture
def small_meta() -> Meta:
    return Meta(
        n_frames=10,
        fps=25,
        pitch=(105.3, 68.0),
        half_blocks=[(1, 0, 9)],
        home_name="FC Barcelona",
        away_name="Athletic Club",
    )


@pytest.fixture
def good_frames(small_meta) -> pd.DataFrame:
    n = small_meta.n_frames
    return pd.DataFrame(
        {
            "frame": np.arange(n, dtype="int64"),
            "half": pd.array([1] * n, dtype="Int8"),
            "graded": np.ones(n, dtype=bool),
            "ball_x": np.full(n, 0.5),
            "ball_y": np.full(n, 0.5),
        }
    )


@pytest.fixture
def good_players(small_meta) -> pd.DataFrame:
    n = small_meta.n_frames
    return pd.DataFrame(
        {
            "frame": np.repeat(np.arange(n, dtype="int64"), 2),
            "slot": np.tile(np.array([0, 11], dtype="int16"), n),
            "team": np.tile(["H", "A"], n),
            "x": np.full(2 * n, 0.4),
            "y": np.full(2 * n, 0.5),
            "speed": np.full(2 * n, 1.0),
        }
    )


@pytest.fixture
def good_tags() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_frame": np.array([2], dtype="int64"),
            "end_frame": np.array([6], dtype="int64"),
            "code": ["SET PIECES"],
            "team": ["H"],
            "type": ["Throw-in"],
            "half": pd.array([1], dtype="Int8"),
        }
    )


@pytest.fixture
def good_emissions(small_meta) -> pd.DataFrame:
    n = small_meta.n_frames
    return pd.DataFrame(
        {
            "frame": np.arange(n, dtype="int64"),
            "p_H": np.full(n, 0.5),
            "p_A": np.full(n, 0.3),
            "p_D": np.full(n, 0.2),
        }
    )


@pytest.fixture
def good_labels(small_meta) -> pd.DataFrame:
    n = small_meta.n_frames
    return pd.DataFrame({"frame": np.arange(n, dtype="int64"), "label": ["H"] * n})


# --------------------------------------------------------------------------- Task 6 fixtures
# Stub pipeline components, registered once under names that cannot clash with the real ones.


@pytest.fixture
def full_meta() -> Meta:
    return Meta(
        n_frames=199319,
        fps=25,
        pitch=(105.3, 68.0),
        half_blocks=[(1, 11185, 90651), (2, 110808, 190197)],
        home_name="FC Barcelona",
        away_name="Athletic Club",
    )


def stub_nearest(match: MatchData) -> pd.DataFrame:
    """Signal: which team has the player nearest the ball, and whether the ball is off the pitch."""
    frames = match.frames[["frame", "ball_x", "ball_y"]]
    joined = match.players.merge(frames, on="frame")
    joined["d"] = np.hypot(joined["x"] - joined["ball_x"], joined["y"] - joined["ball_y"])
    best = joined.groupby(["frame", "team"])["d"].min().unstack("team")
    best = best.reindex(index=frames["frame"], columns=["H", "A"])
    out = pd.DataFrame({"frame": frames["frame"].to_numpy()})
    out["stub_h_near"] = (best["H"].fillna(np.inf) <= best["A"].fillna(np.inf)).to_numpy()
    ball = frames[["ball_x", "ball_y"]]
    out["stub_outside"] = (ball.lt(0) | ball.gt(1)).any(axis=1).to_numpy()
    return out


class StubEmission:
    def __init__(self, confidence: float = 0.8):
        self.confidence = confidence
        self.fitted_on = None

    def fit(self, signals, labels):
        self.fitted_on = len(labels)
        return self

    def predict(self, signals: pd.DataFrame) -> pd.DataFrame:
        c, rest = self.confidence, (1.0 - self.confidence) / 2
        home = signals["stub_h_near"].to_numpy()
        dead = signals["stub_outside"].to_numpy()
        p_h = np.where(dead, rest, np.where(home, c, rest))
        p_a = np.where(dead, rest, np.where(home, rest, c))
        p_d = np.where(dead, c, rest)
        return pd.DataFrame(
            {"frame": signals["frame"].to_numpy(), "p_H": p_h, "p_A": p_a, "p_D": p_d}
        )


def stub_argmax(emissions: pd.DataFrame, tags: pd.DataFrame, meta: Meta, params: dict):
    probs = emissions[["p_H", "p_A", "p_D"]].to_numpy()
    labels = np.array(["H", "A", "D"])[probs.argmax(axis=1)]
    return pd.DataFrame({"frame": emissions["frame"].to_numpy(), "label": labels})


def _register_once(register, name, obj) -> None:
    try:
        register(name)(obj)
    except ValueError:  # already registered by an earlier collection
        pass


_register_once(registry.register_signal, "stub_nearest", stub_nearest)
_register_once(registry.register_emission, "stub_emission", StubEmission)
_register_once(registry.register_decoder, "stub_argmax", stub_argmax)


@pytest.fixture
def synthetic_config() -> dict:
    return {
        "signals": ["stub_nearest"],
        "emission": "stub_emission",
        "decoder": "stub_argmax",
        "emission_params": {},
        "decoder_params": {},
    }


@pytest.fixture
def tmp_bundle(tmp_path):
    return write_bundle(tmp_path)
