"""A tiny game in Metrica's public sample format (EPTS metadata, tracking text, JSON events).

Two halves of 100 frames each (frame ids 1..200). Home team "Team A" (players P1, P2), away team
"Team B" (players P3, P4); at frame 51 player P2 is replaced by P5, so the second tracking
segment lists a different player in position 2.
"""

from __future__ import annotations

import json
from pathlib import Path

PLAYERS = {"P1": "TA", "P2": "TA", "P3": "TB", "P4": "TB", "P5": "TA"}
SEGMENTS = [(1, 50, ["P1", "P2", "P3", "P4"]), (51, 200, ["P1", "P5", "P3", "P4"])]
N_FRAMES = 200


def _metadata() -> str:
    teams = "".join(
        f'<Team id="{tid}"><Name>{name}</Name></Team>'
        for tid, name in (("TA", "Team A"), ("TB", "Team B"))
    )
    players = "".join(
        f'<Player id="{pid}" teamId="{team}"><Name>{pid}</Name></Player>'
        for pid, team in PLAYERS.items()
    )
    channels = "".join(
        f'<PlayerChannel id="{pid.lower()}_{axis}" playerId="{pid}" channelId="{axis}" />'
        for pid in PLAYERS
        for axis in ("x", "y")
    )
    specs = ""
    for start, end, order in SEGMENTS:
        groups = "".join(
            f'<SplitRegister separator=","><PlayerChannelRef playerChannelId="{p.lower()}_x" />'
            f'<PlayerChannelRef playerChannelId="{p.lower()}_y" /></SplitRegister>'
            for p in order
        )
        specs += (
            f'<DataFormatSpecification startFrame="{start}" endFrame="{end}" separator=":">'
            f'<StringRegister name="frameCount" /><SplitRegister separator=";">{groups}'
            "</SplitRegister></DataFormatSpecification>"
        )
    params = "".join(
        f"<ProviderParameter><Name>{name}</Name><Value>{value}</Value></ProviderParameter>"
        for name, value in (
            ("first_half_start", 1),
            ("first_half_end", 100),
            ("second_half_start", 101),
            ("second_half_end", 200),
        )
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?><main><Metadata><GlobalConfig>'
        f"<FrameRate>25</FrameRate><ProviderGlobalParameters>{params}</ProviderGlobalParameters>"
        "</GlobalConfig><Sessions><Session><MatchParameters><FieldSize><Width>105</Width>"
        "<Height>68</Height></FieldSize></MatchParameters></Session></Sessions>"
        f"<Teams>{teams}</Teams><Players>{players}</Players><Devices><Device><Sensors><Sensor>"
        f"<PlayerChannels>{channels}</PlayerChannels></Sensor></Sensors></Device></Devices>"
        f"</Metadata><DataFormatSpecifications>{specs}</DataFormatSpecifications></main>"
    )


def _tracking() -> str:
    """Player in position 1 walks along x at 0.002 of the pitch length per frame (5.265 m/s); the
    others stand still. The ball is missing on frames 10-12."""
    lines = []
    for frame_id in range(1, N_FRAMES + 1):
        groups = [f"{0.1 + 0.002 * frame_id:.5f},0.50000", "0.30000,0.40000"]
        groups += ["0.60000,0.50000", "0.70000,0.60000"]
        ball = "NaN,NaN" if 10 <= frame_id <= 12 else "0.50000,0.50000"
        lines.append(f"{frame_id}:{';'.join(groups)}:{ball}")
    return "\n".join(lines) + "\n"


def event(kind, team, start, end=None, subtypes=(), period=1) -> dict:
    return {
        "team": {"name": f"Team {team}", "id": "TA" if team == "A" else "TB"},
        "type": {"name": kind},
        "subtypes": [{"name": s} for s in subtypes] or None,
        "start": {"frame": start},
        "end": {"frame": start if end is None else end},
        "period": period,
    }


EVENTS = [
    event("SET PIECE", "A", 5, subtypes=["KICK OFF"]),
    event("PASS", "A", 5, 15),
    event("CARRY", "A", 15, 25),
    event("CHALLENGE", "B", 26, subtypes=["GROUND", "LOST"]),
    event("RECOVERY", "B", 30, subtypes=["INTERCEPTION"]),
    event("PASS", "B", 30, 40),
    event("BALL OUT", "B", 40, 45),
    event("SET PIECE", "A", 60, subtypes=["THROW IN"]),
    event("PASS", "A", 60, 70),
    event("FAULT RECEIVED", "A", 80),
    event("SET PIECE", "A", 90, subtypes=["FREE KICK"]),
    event("PASS", "A", 90, 95),
    event("SET PIECE", "B", 110, subtypes=["KICK OFF"], period=2),
    event("SHOT", "B", 120, 125, subtypes=["ON TARGET", "GOAL"], period=2),
    event("SET PIECE", "A", 150, subtypes=["KICK OFF"], period=2),
    event("CARRY", "A", 150, 180, period=2),
]


def write_game(root: Path, events: list[dict] | None = None) -> Path:
    """Write the game under ``root`` with the sample repo's layout; return the game folder."""
    folder = root / "data" / "Sample_Game_3"
    folder.mkdir(parents=True)
    (folder / "Sample_Game_3_metadata.xml").write_text(_metadata(), encoding="utf-8")
    (folder / "Sample_Game_3_tracking.txt").write_text(_tracking(), encoding="utf-8")
    body = {"data": EVENTS if events is None else events, "metadata": {}}
    (folder / "Sample_Game_3_events.json").write_text(json.dumps(body), encoding="utf-8")
    return folder
