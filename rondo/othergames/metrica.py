"""Metrica's public sample games: other matches to validate on, with answer keys from their events.

The official rules forbid hand labels of the provided match for any purpose, validation included,
and allow labelled data from other matches. Metrica publishes three anonymised matches with
tracking and human-recorded event data (``github.com/metrica-sports/sample-data``). Game 3 uses
the same FIFA EPTS tracking format as the provided match; games 1 and 2 use Metrica's CSV format
(home and away tracking in separate files, CSV events). All three load into the same
``MatchData`` and run through the same pipeline.

**The answer key is built by code from the events, following our possession rules:**

- A team has the ball from the start of its first possession event after the other team's
  (``PASS``, ``CARRY``, ``RECOVERY``, ``SET PIECE``, ``SHOT``, ``BALL LOST``, ``BALL OUT``):
  the spell starts where control began. ``CHALLENGE`` events change nothing.
- **Control, as in the official hint** ("keeps the ball for a few seconds" or "more than two
  consecutive passes"): a spell won from the other team in play counts only if it holds
  ``MIN_PASSES`` (3) passes of the new team, or lasts ``MIN_CONTROL_FRAMES`` (2 s) and, in a match
  whose event log records carries, the new team logs another action of its own
  (``FOLLOW_UP_EVENTS``: a pass, carry or shot) after the spell's first event. A log without any
  ``CARRY`` (games 1 and 2) cannot tell a ball dribbled and then lost from a lone touch, so there
  the 2 s time test stands alone. Otherwise the spell stays with the team that had the ball. The annotators log every
  recovery, even a one-touch one that is lost straight back, while the official answers count a
  change only under control. Only events decide this, never tracking, which is the system's own
  sensor. A spell starting from a dead ball (a restart) always counts.
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

GAMES = (1, 2, 3)
METADATA, TRACKING, EVENTS = (
    "Sample_Game_3_metadata.xml",
    "Sample_Game_3_tracking.txt",
    "Sample_Game_3_events.json",
)
CSV_PITCH = (105.0, 68.0)  # the sample README: every game is 105 x 68 m
CSV_FPS = 25
SPEED_HALF_WINDOW = 2  # frames either side of the centred difference for player speed
POSSESSION_EVENTS = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST", "BALL OUT"}
STOPPAGE_SUBTYPES = {"OFFSIDE", "HAND BALL", "REFEREE HIT"}
MIN_CONTROL_FRAMES = 50  # 2 s, as ``RulesParams.min_control_frames``
MIN_PASSES = 3  # the official hint: "more than two consecutive passes"
FOLLOW_UP_EVENTS = {"PASS", "CARRY", "SHOT"}  # an own action after winning the ball shows control


def game_folder(root: Path, game: int = 3) -> Path:
    """The folder of sample game ``game`` inside a checkout of the sample-data repository."""
    return Path(root) / "data" / f"Sample_Game_{game}"


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


def _read_epts_tracking(path: Path, info: dict, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Ball ``(rows, 2)`` and players ``(2, slots, rows)`` positions; NaN where not tracked."""
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
    return ball, np.stack([px, py])


def _read_csv_side(path: Path) -> tuple[pd.DataFrame, int]:
    """One side's CSV tracking: ``(table, players)``. Columns: ``0`` period, ``1`` frame, then
    ``x, y`` per player, then the ball's ``x, y`` (the time column is dropped)."""
    with open(path, encoding="utf-8") as handle:
        names = [next(handle) for _ in range(3)][2].strip().split(",")
    players = sum(1 for name in names[3:] if name.startswith("Player"))
    table = pd.read_csv(path, skiprows=3, header=None, usecols=range(3 + 2 * players + 2))
    return table.drop(columns=2), players


def _players(
    positions: np.ndarray, slot_team: np.ndarray, pitch: tuple[float, float], fps: int
) -> pd.DataFrame:
    """Long table ``frame, slot, team, x, y, speed``; speed from a centred difference in metres."""
    px, py = positions
    width, height = pitch
    k = SPEED_HALF_WINDOW
    dx = np.full_like(px, np.nan)
    dy = np.full_like(py, np.nan)
    dx[:, k:-k] = (px[:, 2 * k :] - px[:, : -2 * k]) * width
    dy[:, k:-k] = (py[:, 2 * k :] - py[:, : -2 * k]) * height
    speed = np.hypot(dx, dy) * fps / (2 * k)
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


def _match(ball: np.ndarray, positions: np.ndarray, slot_team: np.ndarray, meta: Meta) -> MatchData:
    """Assemble and check the C1 tables from ball and player positions per row."""
    n_frames = meta.n_frames
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in meta.half_blocks:
        half[first : last + 1] = number
    frames = pd.DataFrame(
        {
            "frame": np.arange(n_frames, dtype="int64"),
            "half": pd.array(np.where(half > 0, half, pd.NA), dtype="Int8"),
            "graded": half > 0,
            "ball_x": ball[:, 0],
            "ball_y": ball[:, 1],
        }
    )
    players = _players(positions, slot_team, meta.pitch, meta.fps)
    tags = _no_tags()
    validate_meta(meta)
    validate_frames(frames, meta)
    validate_players(players, meta)
    validate_tags(tags, meta)
    return MatchData(frames=frames, players=players, tags=tags, meta=meta)


def _load_epts(folder: Path) -> MatchData:
    info = _meta_xml(folder / METADATA)
    meta = Meta(
        n_frames=info["halves"][-1][2] + 1,
        fps=info["fps"],
        pitch=info["pitch"],
        half_blocks=info["halves"],
        home_name=info["home"],
        away_name=info["away"],
    )
    ball, positions = _read_epts_tracking(folder / TRACKING, info, meta.n_frames)
    slot_team = np.array([info["player_team"][p] for p in info["slots"]])
    return _match(ball, positions, slot_team, meta)


def _load_csv(folder: Path, game: int) -> MatchData:
    prefix = f"Sample_Game_{game}"
    home, n_home = _read_csv_side(folder / f"{prefix}_RawTrackingData_Home_Team.csv")
    away, n_away = _read_csv_side(folder / f"{prefix}_RawTrackingData_Away_Team.csv")
    n_frames = int(max(home[1].max(), away[1].max()))
    rows = home[1].to_numpy(dtype=np.int64) - 1
    periods = home[0].to_numpy()
    halves = [
        (period, int(rows[periods == period].min()), int(rows[periods == period].max()))
        for period in (1, 2)
        if (periods == period).any()
    ]
    meta = Meta(
        n_frames=n_frames,
        fps=CSV_FPS,
        pitch=CSV_PITCH,
        half_blocks=halves,
        home_name="Home",
        away_name="Away",
    )
    px = np.full((n_home + n_away, n_frames), np.nan)
    py = np.full((n_home + n_away, n_frames), np.nan)
    for table, first_slot, count in ((home, 0, n_home), (away, n_home, n_away)):
        at = table[1].to_numpy(dtype=np.int64) - 1
        values = table.to_numpy(dtype="float64")[:, 2:]
        for i in range(count):
            px[first_slot + i, at] = values[:, 2 * i]
            py[first_slot + i, at] = values[:, 2 * i + 1]
    ball = np.full((n_frames, 2), np.nan)
    ball[rows] = home.to_numpy(dtype="float64")[:, -2:]
    slot_team = np.array(["H"] * n_home + ["A"] * n_away)
    return _match(ball, np.stack([px, py]), slot_team, meta)


def load_sample_game(root: Path, game: int = 3) -> MatchData:
    """Sample game ``game`` as a ``MatchData``, the same shape as the provided match, checked
    against C1."""
    folder = game_folder(root, game)
    return _load_epts(folder) if game == 3 else _load_csv(folder, game)


# -- the answer key -------------------------------------------------------------------------------


def home_team_id(root: Path) -> str:
    """Game 3's home team id (the first ``<Team>`` in the metadata, as for the provided match)."""
    return _meta_xml(game_folder(root, 3) / METADATA)["home_id"]


def read_events(path: Path, home_id: str) -> pd.DataFrame:
    """Game 3's JSON events: ``start, end`` (rows), ``kind``, ``subtypes`` (a set), ``team``
    (``H``/``A``), ``half``."""
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


def read_events_csv(path: Path) -> pd.DataFrame:
    """Games 1 and 2's CSV events, as the same table as ``read_events``."""
    raw = pd.read_csv(path, dtype={"Subtype": "str"}, keep_default_na=False)
    start = raw["Start Frame"].astype("int64") - 1
    end = raw["End Frame"].astype("int64") - 1
    table = pd.DataFrame(
        {
            "start": start,
            "end": np.maximum(start, end),  # set pieces carry an end frame of 0
            "kind": raw["Type"].str.strip(),
            "subtypes": [
                frozenset(s.strip() for s in sub.split("-") if s.strip()) for sub in raw["Subtype"]
            ],
            "team": np.where(raw["Team"].str.strip() == "Home", "H", "A"),
            "half": raw["Period"].astype("int64"),
        }
    )
    return table.sort_values(["start", "end"], kind="stable", ignore_index=True)


def load_events(root: Path, game: int) -> pd.DataFrame:
    """Game ``game``'s events, as the common table (``start, end, kind, subtypes, team, half``)."""
    folder = game_folder(root, game)
    if game == 3:
        return read_events(folder / EVENTS, home_team_id(root))
    return read_events_csv(folder / f"Sample_Game_{game}_RawEventsData.csv")


def answer_key(root: Path, game: int, meta: Meta) -> pd.Series:
    """The full-length answer key of sample game ``game``, from its events."""
    return possession_from_events(load_events(root, game), meta)


def nearest_event_kind(events: pd.DataFrame, frame: int, max_gap: int = 25) -> str:
    """The ``kind`` of the event covering ``frame``, or the closest one within ``max_gap`` rows
    (1 s by default); ``"none"`` with nothing that close, or no events at all.

    For characterising an error frame: what was actually happening there (a restart, a challenge,
    the ball out of play, ...), read off the public event log: the sample games have no video.
    """
    if events.empty:
        return "none"
    within = events[(events["start"] <= frame) & (frame <= events["end"])]
    if not within.empty:
        return str(within.iloc[-1]["kind"])
    gap = np.minimum((events["start"] - frame).abs(), (events["end"] - frame).abs())
    closest = gap.idxmin()
    return str(events.loc[closest, "kind"]) if gap.loc[closest] <= max_gap else "none"


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


def possession_from_events(
    events: pd.DataFrame,
    meta: Meta,
    *,
    min_passes: int = MIN_PASSES,
    require_follow_up: bool | None = None,
) -> pd.Series:
    """Full-length ``H/A/D/X`` labels from the events, by the rules in the module docstring.

    ``require_follow_up=None`` applies the follow-up test only when the events record carries.
    ``min_passes=2, require_follow_up=False`` gives the earlier, more generous control test, for
    comparing scores across the change.
    """
    if require_follow_up is None:
        require_follow_up = bool((events["kind"] == "CARRY").any())
    labels = np.full(meta.n_frames, "X", dtype=object)
    for half, _first, last in meta.half_blocks:
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
    _require_control(labels, events, min_passes, require_follow_up)
    return pd.Series(labels, index=pd.RangeIndex(meta.n_frames, name="frame"), name="label")


def _require_control(
    labels: np.ndarray, events: pd.DataFrame, min_passes: int, require_follow_up: bool
) -> None:
    """Give a spell won in play without control back to the team that had the ball, in place."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    possession = events[events["kind"].isin(POSSESSION_EVENTS)]
    previous = None
    for start, end in zip(starts, ends, strict=True):
        label = labels[start]
        if label in ("H", "A") and previous in ("H", "A") and previous != label:
            own = possession[
                (possession["team"] == label) & possession["start"].between(start, end - 1)
            ]
            passes = int((own["kind"] == "PASS").sum())
            follow_up = bool(own["kind"].iloc[1:].isin(FOLLOW_UP_EVENTS).any())
            held = end - start >= MIN_CONTROL_FRAMES and (follow_up or not require_follow_up)
            if passes < min_passes and not held:
                labels[start:end] = previous
                label = previous
        previous = label
