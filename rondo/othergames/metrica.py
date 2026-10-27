"""Metrica's public sample game 3: another match to validate on, with an answer key from its events.

The official rules forbid hand labels of the provided match for any purpose, validation included,
and allow labelled data from other matches. Metrica publishes three anonymised matches with
tracking and human-recorded event data (``github.com/metrica-sports/sample-data``). Game 3 uses
the same FIFA EPTS tracking format as the provided match, so it runs through the same pipeline.

**The answer key is built by code from the events, following our possession rules:**

- A team has the ball from the start of its first possession event after the other team's
  (``PASS``, ``CARRY``, ``RECOVERY``, ``SET PIECE``, ``SHOT``, ``BALL LOST``, ``BALL OUT``):
  the spell starts where control began. ``CHALLENGE`` events change nothing.
- **Control, as in our possession rules:** a spell won from the other team in play counts only
  if it lasts ``MIN_CONTROL_FRAMES`` (2 s) or holds ``MIN_PASSES`` (2) passes; otherwise it stays
  with the team that had the ball. The annotators log every recovery, even a one-touch one, while
  the official answers count a change only under control. A spell starting from a dead ball (a
  restart) always counts.
- The ball is dead from where it leaves the pitch (``BALL OUT``, a ``SHOT`` that goes ``OUT`` or
  is a ``GOAL``, at the event's end frame), and from a stoppage (``FAULT RECEIVED``, a
  ``CHALLENGE`` with ``FAULT`` and no ``ADVANTAGE``, a ``BALL LOST`` or ``PASS`` for ``OFFSIDE``,
  ``HAND BALL`` or ``REFEREE HIT``, at the end frame), until the next ``SET PIECE`` or, failing
  one, the next possession event.
- ``X`` (not scored) before each half's first event and after its last.

Frames: the sample files count from 1, rows here from 0 (``row = frame - 1``), as for the
provided match. The sample games have no Smart Tagging, so restart anchors play no part here.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from rondo.contracts import (
    MatchData,
    Meta,
    validate_frames,
    validate_meta,
    validate_players,
    validate_tags,
)

GAME_FOLDER = Path("data") / "Sample_Game_3"
METADATA, TRACKING, EVENTS = (
    "Sample_Game_3_metadata.xml",
    "Sample_Game_3_tracking.txt",
    "Sample_Game_3_events.json",
)
SPEED_HALF_WINDOW = 2  # frames either side of the centred difference for player speed
POSSESSION_EVENTS = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST", "BALL OUT"}
STOPPAGE_SUBTYPES = {"OFFSIDE", "HAND BALL", "REFEREE HIT"}
MIN_CONTROL_FRAMES = 50  # 2 s, as ``RulesParams.min_control_frames``
MIN_PASSES = 2


def game_folder(root: Path) -> Path:
    """The game-3 folder inside a checkout of the sample-data repository."""
    return Path(root) / GAME_FOLDER


# -- tracking -------------------------------------------------------------------------------------


def _meta_xml(path: Path) -> dict:
    root = ElementTree.parse(path).getroot()
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    teams = list(root.iter("Team"))
    team_label = {teams[0].get("id"): "H", teams[1].get("id"): "A"}
    player_team = {p.get("id"): team_label[p.get("teamId")] for p in root.iter("Player")}
    slots = {pid: i for i, pid in enumerate(player_team)}  # one slot per person, stable
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    segments = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        order = [channel_player[ref] for ref in refs[::2]]  # x then y for each position
        segments.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    halves = [
        (1, int(params["first_half_start"]) - 1, int(params["first_half_end"]) - 1),
        (2, int(params["second_half_start"]) - 1, int(params["second_half_end"]) - 1),
    ]
    field = next(root.iter("FieldSize"))
    return {
        "fps": int(root.findtext(".//FrameRate")),
        "pitch": (float(field.findtext("Width")), float(field.findtext("Height"))),
        "home_id": teams[0].get("id"),
        "home": (teams[0].findtext("Name") or "").strip(),
        "away": (teams[1].findtext("Name") or "").strip(),
        "player_team": player_team,
        "slots": slots,
        "segments": sorted(segments),
        "halves": halves,
    }


def _read_tracking(path: Path, info: dict, n_frames: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Ball positions per row and player positions per (slot, row); NaN where not tracked."""
    n_slots = len(info["slots"])
    px = np.full((n_slots, n_frames), np.nan)
    py = np.full((n_slots, n_frames), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    segments = info["segments"]
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            frame_id, groups, ball_group = line.split(":")
            row = int(frame_id) - 1
            if not 0 <= row < n_frames:
                continue
            order = next(o for start, end, o in segments if start <= row + 1 <= end)
            for player, group in zip(order, groups.split(";"), strict=False):
                if group and group[0] != "N":
                    x, y = group.split(",")[:2]
                    slot = info["slots"][player]
                    px[slot, row], py[slot, row] = float(x), float(y)
            if ball_group and ball_group[0] != "N":
                bx, by = ball_group.split(",")[:2]
                ball[row] = float(bx), float(by)
    frames = pd.DataFrame(
        {"frame": np.arange(n_frames, dtype="int64"), "ball_x": ball[:, 0], "ball_y": ball[:, 1]}
    )
    return frames, np.stack([px, py])


def _players(positions: np.ndarray, info: dict, fps: int) -> pd.DataFrame:
    """Long table ``frame, slot, team, x, y, speed``; speed from a centred difference in metres."""
    px, py = positions
    width, height = info["pitch"]
    k = SPEED_HALF_WINDOW
    dx = np.full_like(px, np.nan)
    dy = np.full_like(py, np.nan)
    dx[:, k:-k] = (px[:, 2 * k :] - px[:, : -2 * k]) * width
    dy[:, k:-k] = (py[:, 2 * k :] - py[:, : -2 * k]) * height
    speed = np.hypot(dx, dy) * fps / (2 * k)
    slot_team = np.array([info["player_team"][p] for p in info["slots"]])
    slot, frame = np.nonzero(~np.isnan(px))
    order = np.lexsort((slot, frame))
    slot, frame = slot[order], frame[order]
    return pd.DataFrame(
        {
            "frame": frame.astype("int64"),
            "slot": slot.astype("int16"),
            "team": pd.Series(slot_team[slot], dtype="str"),
            "x": px[slot, frame],
            "y": py[slot, frame],
            "speed": np.nan_to_num(speed[slot, frame], nan=0.0),
        }
    )


def _no_tags() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_frame": pd.Series([], dtype="int64"),
            "end_frame": pd.Series([], dtype="int64"),
            "code": pd.Series([], dtype="str"),
            "team": pd.Series([], dtype="str"),
            "type": pd.Series([], dtype="str"),
            "half": pd.array([], dtype="Int8"),
        }
    )


def load_sample_game(root: Path) -> MatchData:
    """Game 3 as a ``MatchData``, the same shape as the provided match, checked against C1."""
    folder = game_folder(root)
    info = _meta_xml(folder / METADATA)
    n_frames = info["halves"][-1][2] + 1
    meta = Meta(
        n_frames=n_frames,
        fps=info["fps"],
        pitch=info["pitch"],
        half_blocks=info["halves"],
        home_name=info["home"],
        away_name=info["away"],
    )
    frames, positions = _read_tracking(folder / TRACKING, info, n_frames)
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in meta.half_blocks:
        half[first : last + 1] = number
    frames["half"] = pd.array(np.where(half > 0, half, pd.NA), dtype="Int8")
    frames["graded"] = half > 0
    frames = frames[["frame", "half", "graded", "ball_x", "ball_y"]]
    players = _players(positions, info, meta.fps)
    tags = _no_tags()
    validate_meta(meta)
    validate_frames(frames, meta)
    validate_players(players, meta)
    validate_tags(tags, meta)
    return MatchData(frames=frames, players=players, tags=tags, meta=meta)


# -- the answer key -------------------------------------------------------------------------------


def home_team_id(root: Path) -> str:
    """The id of the home team (the first ``<Team>`` in the metadata, as for the provided match)."""
    return _meta_xml(game_folder(root) / METADATA)["home_id"]


def read_events(path: Path, home_id: str) -> pd.DataFrame:
    """``start, end`` (rows), ``kind``, ``subtypes`` (a set), ``team`` (``H``/``A``), ``half``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
    rows = []
    for e in raw:
        subtypes = e.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        rows.append(
            {
                "start": int(e["start"]["frame"]) - 1,
                "end": int((e.get("end") or e["start"])["frame"]) - 1,
                "kind": e["type"]["name"],
                "subtypes": frozenset(s["name"] for s in subtypes),
                "team": "H" if e["team"]["id"] == home_id else "A",
                "half": int(e["period"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["start", "end"], kind="stable", ignore_index=True)


def _dead_from(e) -> int | None:
    """The row from which an event makes the ball dead, or ``None``."""
    kind, sub = e.kind, e.subtypes
    if kind == "BALL OUT":
        return e.end
    if kind == "SHOT" and ({"OUT", "GOAL"} & sub):
        return e.end
    if kind == "FAULT RECEIVED":
        return e.start
    if kind == "CHALLENGE" and "FAULT" in sub and "ADVANTAGE" not in sub:
        return e.start
    if kind in {"BALL LOST", "PASS"} and (STOPPAGE_SUBTYPES & sub):
        return e.end
    return None


def possession_from_events(events: pd.DataFrame, meta: Meta) -> pd.Series:
    """Full-length ``H/A/D/X`` labels from the events, by the rules in the module docstring."""
    labels = np.full(meta.n_frames, "X", dtype=object)
    for half, first, last in meta.half_blocks:
        block = events[events["half"] == half]
        if block.empty:
            continue
        state, since = None, None  # current label, and the row it started on
        for e in block.itertuples(index=False):
            if state == "D" and e.kind in POSSESSION_EVENTS:  # a set piece, or play resumed
                labels[since : e.start] = "D"
                state, since = e.team, e.start
            elif e.kind in POSSESSION_EVENTS and e.team != state:
                if state is not None:
                    labels[since : e.start] = state
                state, since = e.team, e.start
            dead = _dead_from(e)
            if dead is not None and state != "D":
                if state is not None:
                    labels[since:dead] = state
                state, since = "D", dead
        stop = min(int(block["end"].max()), last) + 1
        if state is not None:
            labels[since:stop] = state
    _require_control(labels, events)
    return pd.Series(labels, index=pd.RangeIndex(meta.n_frames, name="frame"), name="label")


def _require_control(labels: np.ndarray, events: pd.DataFrame) -> None:
    """Give a spell won in play without control back to the team that had the ball, in place."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    passes = events[events["kind"] == "PASS"]
    previous = None
    for start, end in zip(starts, ends, strict=True):
        label = labels[start]
        if label in ("H", "A") and previous in ("H", "A") and previous != label:
            own = passes[(passes["team"] == label) & passes["start"].between(start, end - 1)]
            if end - start < MIN_CONTROL_FRAMES and len(own) < MIN_PASSES:
                labels[start:end] = previous
                label = previous
        previous = label
