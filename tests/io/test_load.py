import importlib.metadata
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from rondo.contracts import MatchData, validate_frames, validate_players, validate_tags
from rondo.io import load, load_match
from tests.io.fixtures import TAGS_XML, write_bundle


def test_load_match_validates(tmp_path):
    bundle = write_bundle(tmp_path, n=50)
    match = load_match(bundle, cache=False, expected_frames=50)
    assert isinstance(match, MatchData)
    assert match.meta.n_frames == 50 and match.meta.fps == 25
    assert match.meta.pitch == (105.3, 68.0) and match.meta.half_blocks == []
    assert (match.meta.home_name, match.meta.away_name) == ("FC Barcelona", "Athletic Club")
    assert not match.frames["graded"].any() and match.frames["half"].isna().all()
    assert len(match.tags) == 3 and len(match.players) == 50 * 2 - 10


def test_load_match_fills_half_and_graded(tmp_path):
    bundle = write_bundle(tmp_path, n=11200)
    match = load_match(bundle, cache=False, expected_frames=11200)
    assert match.meta.half_blocks == [(1, 11185, 11199)]
    frames = match.frames
    assert frames["graded"].sum() == 15
    assert frames.loc[11185, "half"] == 1 and pd.isna(frames.loc[11184, "half"])
    validate_frames(frames, match.meta)
    validate_players(match.players, match.meta)
    validate_tags(match.tags, match.meta)


def test_load_match_default_demands_official_count(tmp_path):
    bundle = write_bundle(tmp_path, n=50)
    with pytest.raises(ValueError):
        load_match(bundle, cache=False)


def test_missing_files_listed(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        load_match(tmp_path, cache=False, expected_frames=50)
    assert "smart_tagging" in str(err.value) and str(tmp_path) in str(err.value)


def test_cache_reused_then_invalidated(tmp_path):
    bundle = write_bundle(tmp_path, n=50)
    load_match(bundle, expected_frames=50)
    cache_file = bundle / "cache" / "match.pkl"
    assert cache_file.exists()

    # Tamper with the cached payload: a cache hit must return it unchanged.
    with open(cache_file, "rb") as f:
        key, match = pickle.load(f)
    match.meta = type(match.meta)(**{**match.meta.__dict__, "away_name": "CACHED"})
    with open(cache_file, "wb") as f:
        pickle.dump((key, match), f)
    assert load_match(bundle, expected_frames=50).meta.away_name == "CACHED"
    assert load_match(bundle, cache=False, expected_frames=50).meta.away_name == "Athletic Club"

    # Changing an input file invalidates it.
    tags_path = bundle / "tactical" / "BAR-ATH_tactical_smart_tagging.xml"
    tags_path.write_text(TAGS_XML.replace("Goal", "Foul"), encoding="utf-8")
    os.utime(tags_path, ns=(1, 1))
    reloaded = load_match(bundle, expected_frames=50)
    assert reloaded.meta.away_name == "Athletic Club"
    assert "FOUL" in reloaded.tags["code"].tolist()


def _tamper(cache_file):
    with open(cache_file, "rb") as f:
        key, match = pickle.load(f)
    match.meta = type(match.meta)(**{**match.meta.__dict__, "away_name": "CACHED"})
    with open(cache_file, "wb") as f:
        pickle.dump((key, match), f)


def test_cache_key_changes_with_the_version_stamp(tmp_path, monkeypatch):
    bundle = write_bundle(tmp_path, n=50)
    load_match(bundle, expected_frames=50)
    _tamper(bundle / "cache" / "match.pkl")
    assert load_match(bundle, expected_frames=50).meta.away_name == "CACHED"
    monkeypatch.setattr(load, "CACHE_FORMAT", load.CACHE_FORMAT + 1)  # e.g. a code change
    assert load_match(bundle, expected_frames=50).meta.away_name == "Athletic Club"


def test_cache_stamp_names_the_code_and_library_versions():
    stamp = load.cache_stamp()
    assert stamp == (
        load.CACHE_FORMAT,
        importlib.metadata.version("rondo"),
        pd.__version__,
        np.__version__,
    )


def test_no_cache_neither_reads_nor_writes(tmp_path):
    bundle = write_bundle(tmp_path, n=50)
    load_match(bundle, cache=False, expected_frames=50)
    assert not (bundle / "cache").exists()


def test_refresh_ignores_the_cache_and_rewrites_it(tmp_path):
    bundle = write_bundle(tmp_path, n=50)
    load_match(bundle, expected_frames=50)
    _tamper(bundle / "cache" / "match.pkl")
    assert load_match(bundle, refresh=True, expected_frames=50).meta.away_name == "Athletic Club"
    assert load_match(bundle, expected_frames=50).meta.away_name == "Athletic Club"
