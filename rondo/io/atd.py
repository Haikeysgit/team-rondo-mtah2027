"""Read the FIFA/EPTS tracking files: the metadata XML and the raw TXT."""

from __future__ import annotations

from array import array
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd


def read_atd_meta(meta_path: Path) -> tuple[np.ndarray, str, str]:
    """Return ``(slot_team, home_name, away_name)``.

    ``slot_team[i]`` is ``"H"`` or ``"A"`` for tracking slot ``i`` (the order of the ``<Player>``
    elements). The first ``<Team>`` is the home team.
    """
    root = ElementTree.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"{meta_path}: expected exactly 2 <Team> elements, found {len(teams)}")
    names = [(team.findtext("Name") or "").strip() for team in teams]
    home_id = teams[0].get("id")
    players = list(root.iter("Player"))
    if not players:
        raise ValueError(f"{meta_path}: no <Player> elements found")
    slot_team = np.array(["H" if p.get("teamId") == home_id else "A" for p in players])
    return slot_team, names[0], names[1]


def read_atd_raw(
    raw_path: Path, slot_team: np.ndarray, n_frames: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stream the raw tracking file into ``(frames, players, report)``.

    Line format ``<frame_id>:<x,y,speed>;<x,y,speed>;...:<ball group>`` with ``video_frame =
    frame_id - 1``. A group starting with ``N`` (NaN) is missing. Blank lines are ignored;
    malformed lines and lines whose frame is outside ``0..n_frames-1`` are skipped and counted,
    never shifted into a neighbour's place.

    Report: ``n_records`` (well-formed lines), ``first_id`` / ``last_id`` (frame ids, None when
    there are no records), ``contiguous`` (ids go up by exactly 1 from line to line),
    ``skipped_lines`` (malformed or out of range) and ``ball_fields`` (largest number of comma
    fields seen in a ball group).
    """
    n_slots = len(slot_team)
    ball_x = np.full(n_frames, np.nan)
    ball_y = np.full(n_frames, np.nan)
    p_frame, p_slot = array("q"), array("h")
    p_x, p_y, p_speed = array("d"), array("d"), array("d")

    n_records = skipped = ball_fields = 0
    first_id = last_id = None
    contiguous = True

    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            try:
                if len(parts) != 3:
                    raise ValueError("expected 3 ':'-separated parts")
                frame_id = int(parts[0])
                groups = parts[1].split(";") if parts[1] else []
                if len(groups) > n_slots:
                    raise ValueError("more player groups than slots")
                found = []
                for slot, group in enumerate(groups):
                    if not group or group[0] == "N":
                        continue
                    x, y, speed = group.split(",")[:3]
                    found.append((slot, float(x), float(y), float(speed)))
                ball = parts[2]
                bx = by = np.nan
                fields = 0
                if ball and ball[0] != "N":
                    ball_parts = ball.split(",")
                    fields = len(ball_parts)
                    bx, by = float(ball_parts[0]), float(ball_parts[1])
            except (ValueError, IndexError):
                skipped += 1
                continue

            n_records += 1
            if first_id is None:
                first_id = frame_id
            elif frame_id != last_id + 1:
                contiguous = False
            last_id = frame_id

            frame = frame_id - 1
            if not 0 <= frame < n_frames:
                skipped += 1
                continue
            ball_fields = max(ball_fields, fields)
            ball_x[frame], ball_y[frame] = bx, by
            for slot, x, y, speed in found:
                p_frame.append(frame)
                p_slot.append(slot)
                p_x.append(x)
                p_y.append(y)
                p_speed.append(speed)

    frames = pd.DataFrame(
        {"frame": np.arange(n_frames, dtype="int64"), "ball_x": ball_x, "ball_y": ball_y}
    )
    frame_arr = np.frombuffer(p_frame, dtype="int64")
    slot_arr = np.frombuffer(p_slot, dtype="int16")
    order = np.lexsort((slot_arr, frame_arr))
    players = pd.DataFrame(
        {
            "frame": frame_arr[order],
            "slot": slot_arr[order],
            "team": pd.Series(np.asarray(slot_team)[slot_arr[order]], dtype="str"),
            "x": np.frombuffer(p_x, dtype="float64")[order],
            "y": np.frombuffer(p_y, dtype="float64")[order],
            "speed": np.frombuffer(p_speed, dtype="float64")[order],
        }
    )
    report = {
        "n_records": n_records,
        "first_id": first_id,
        "last_id": last_id,
        "contiguous": contiguous,
        "skipped_lines": skipped,
        "ball_fields": ball_fields,
    }
    return frames, players, report
