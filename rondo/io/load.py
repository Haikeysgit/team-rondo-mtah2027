"""Assemble the C1 ``MatchData`` from the bundle, and summarise the bundle with aggregates only."""

from __future__ import annotations

import pickle
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from rondo import config, kit
from rondo.contracts import (
    MatchData,
    Meta,
    validate_frames,
    validate_meta,
    validate_players,
    validate_tags,
)
from rondo.io.atd import read_atd_meta, read_atd_raw
from rondo.io.tags import read_tags

GAME = "BAR-ATH"
_CACHE_ASSETS = ("atd_raw", "atd_meta", "smart_tagging")


def _official_frames() -> int:
    return int(kit.games.GAMES[GAME]["n_video_frames"])


def _paths(data_dir: Path | None, assets: tuple[str, ...]) -> tuple[Path, dict[str, Path]]:
    data_dir = Path(data_dir) if data_dir is not None else config.data_dir()
    all_paths = kit.games.asset_paths(GAME, data_dir=str(data_dir))
    paths = {name: Path(all_paths[name]) for name in assets}
    missing = [p for p in paths.values() if not p.is_file()]
    if missing:
        listing = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(f"Missing file(s) in data dir {data_dir}:\n{listing}")
    return data_dir, paths


def _clipped_blocks(n_frames: int) -> list[tuple[int, int, int]]:
    """The official half blocks, cut to the frames present (identity for the full match)."""
    blocks = []
    for half, first, last in kit.games.half_video_blocks(GAME):
        if first < n_frames:
            blocks.append((int(half), int(first), int(min(last, n_frames - 1))))
    return blocks


def _cache_key(paths: dict[str, Path], n_frames: int) -> tuple:
    stats = [(paths[a].stat().st_size, paths[a].stat().st_mtime_ns) for a in _CACHE_ASSETS]
    return (n_frames, *stats)


def _build(paths: dict[str, Path], n_frames: int) -> MatchData:
    slot_team, home, away = read_atd_meta(paths["atd_meta"])
    frames, players, report = read_atd_raw(paths["atd_raw"], slot_team, n_frames)
    if not _aligned(report, n_frames):
        raise ValueError(
            f"Raw tracking file is not aligned to {n_frames} frames: "
            f"{report['n_records']} records, ids {report['first_id']}..{report['last_id']}, "
            f"contiguous={report['contiguous']}, skipped lines={report['skipped_lines']}"
        )
    meta = Meta(
        n_frames=n_frames,
        fps=int(kit.games.GAMES[GAME]["fps"]),
        pitch=tuple(float(v) for v in kit.games.GAMES[GAME]["pitch"]),
        half_blocks=_clipped_blocks(n_frames),
        home_name=home,
        away_name=away,
    )
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in meta.half_blocks:
        half[first : last + 1] = number
    frames["half"] = pd.array(np.where(half > 0, half, pd.NA), dtype="Int8")
    frames["graded"] = half > 0
    frames = frames[["frame", "half", "graded", "ball_x", "ball_y"]]
    tags = read_tags(paths["smart_tagging"], home)
    validate_meta(meta)
    validate_frames(frames, meta)
    validate_players(players, meta)
    validate_tags(tags, meta)
    return MatchData(frames=frames, players=players, tags=tags, meta=meta)


def _read_cache(cache_file: Path, key: tuple) -> MatchData | None:
    """The cached match if the file exists, is readable and was built from the same inputs."""
    if not cache_file.is_file():
        return None
    try:
        with open(cache_file, "rb") as handle:
            cached_key, cached = pickle.load(handle)
    except (OSError, EOFError, ValueError, AttributeError, ImportError, pickle.UnpicklingError):
        return None
    return cached if cached_key == key else None


def load_match(
    data_dir: Path | None = None, cache: bool = True, expected_frames: int | None = None
) -> MatchData:
    """Load the bundle into C1 and validate every table.

    ``expected_frames`` is the number of video frames (default: the official 199,319); tests pass
    a smaller number for a tiny fixture bundle. With ``cache`` the result is pickled to
    ``<data_dir>/cache/match.pkl`` and reused while the three source files keep the same size and
    modification time.
    """
    n_frames = _official_frames() if expected_frames is None else int(expected_frames)
    data_dir, paths = _paths(data_dir, _CACHE_ASSETS)
    key = _cache_key(paths, n_frames)
    cache_file = data_dir / "cache" / "match.pkl"
    if cache:
        cached = _read_cache(cache_file, key)
        if cached is not None:
            return cached
    match = _build(paths, n_frames)
    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as handle:
            pickle.dump((key, match), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return match


def _aligned(report: dict, expected: int) -> bool:
    return bool(
        report["n_records"] == expected and report["contiguous"] and report["first_id"] == 1
    )


def check_bundle(data_dir: Path | None = None, *, expected_records: int | None = None) -> dict:
    """Aggregate sanity report on the bundle (plain numbers only, never rows).

    Keys: ``n_records``, ``aligned`` (record count equals ``expected_records`` -- default the
    official 199,319 --, ids contiguous, first id 1), ``ball_present_graded_pct`` (None when no
    graded frame is present), ``n_tags``, ``homography_rows``, ``homography_valid_pct``,
    ``ball_fields`` (comma fields in the ball group; more than 3 hints at a ball-status field).
    """
    n_frames = _official_frames() if expected_records is None else int(expected_records)
    _, paths = _paths(data_dir, ("atd_meta", "atd_raw", "smart_tagging", "homography"))
    slot_team, _, _ = read_atd_meta(paths["atd_meta"])
    frames, _, report = read_atd_raw(paths["atd_raw"], slot_team, n_frames)

    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in _clipped_blocks(n_frames):
        graded[first : last + 1] = True
    ball_pct = None
    if graded.any():
        present = frames["ball_x"].notna().to_numpy()[graded]
        ball_pct = float(100.0 * present.mean())

    homography = pd.read_csv(paths["homography"], usecols=["valid"])
    valid_pct = float(100.0 * homography["valid"].mean()) if len(homography) else None

    n_tags = sum(1 for _ in ElementTree.parse(paths["smart_tagging"]).getroot().iter("instance"))
    return {
        "n_records": int(report["n_records"]),
        "aligned": _aligned(report, n_frames),
        "ball_present_graded_pct": ball_pct,
        "n_tags": int(n_tags),
        "homography_rows": len(homography),
        "homography_valid_pct": valid_pct,
        "ball_fields": int(report["ball_fields"]),
    }
