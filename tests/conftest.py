"""Shared fixtures: a tiny valid match whose tables each test may mutate freely."""

import numpy as np
import pandas as pd
import pytest

from rondo.contracts import Meta


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
