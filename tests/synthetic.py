"""Synthetic match generator: the only data the tests use (the real data never leaves the laptop).

``make_match`` plays a script of ``Step`` events on a small pitch with 22 players and returns a
``MatchData`` (contract C1) plus the expected label for every frame, following docs/POSSESSION.md.

Model. Positions are kept in metres (x 0..105.3, y 0..68) and converted to 0..1 on output.
Home (slots 0-10) attacks towards +x, away (slots 11-21) towards -x. Players start on a 4-4-2 grid
(plus seeded noise) and wander around it. A step overrides only the players it needs (the carrier,
passer, receiver ...), which are moved at constant velocity or held still, so every position is
continuous and ``players.speed`` is exactly the speed implied by consecutive positions.
The first frame of the block repeats the second frame's speed.

Events (``Step.team`` is the team in possession unless noted):

- ``dribble``: carrier of ``team`` runs at 5 m/s, ball 0.3 m from him. Truth ``team``.
- ``pass``: the holder keeps the ball, then it flies in a straight line at >= 15 m/s to a teammate
  (both stand still). Truth unchanged.
- ``interception``: ``team`` is the *other* team. The ball flies from the holder to a player of
  ``team`` who stands still (flight takes the first half of the step, at >= 15 m/s when a far
  enough player exists), then sits with him. Truth switches to ``team`` on the first frame the
  ball is within 0.5 m of him and slow (he controls it): frame ``n // 2`` of the step. The flight
  is a pass of up to ~30 m, so keep the step to about 4 s at most.
- ``deflection``: the ball flies from the holder, touches a standing opponent (usually for one
  frame: ball within 0.5 m at >= 15 m/s, never controlled), and goes on to a teammate. Truth
  unchanged.
- ``aerial``: the ball flies to between a player of each team and hovers there, > 0.5 m from both.
  Nobody controls it. Truth unchanged.
- ``out``: the ball flies towards a line (players keep moving, truth stays the current team),
  crosses it on frame ``max(1, n // 4)`` of the step and stays at least 1 m outside. From that
  frame truth is ``D`` and every player stands still, except that when a ``restart`` follows, the
  player of that team who will take it jogs (<= 6 m/s) to the line and the ball ends up just
  outside next to him (the mean speed of all players on D frames stays <= 0.8 m/s).
- ``restart``: the ball is placed with that player (a synthetic throw-in on the line, 0.5 m
  inside it; without a preceding ``out`` the nearest player of ``team``) and he keeps it for the
  step; the other players move again. Truth ``team``, with a player of ``team`` within 0.5 m of
  the ball on every frame.
- ``missing``: ball NaN, players keep moving, possession continues. ``team`` is ignored.

Steps that need the ball (``pass``, ``interception``, ``deflection``, ``aerial``, ``dribble``) hand
it to the nearest player of the possessing team if nobody holds it (for example after ``aerial``);
the ball may jump by a few metres then. After ``out`` only ``restart``, ``dribble``, ``missing``
and ``out`` are allowed; ``pass``, ``deflection``, ``aerial`` and ``interception`` raise
``ValueError``. ``dribble`` and ``restart`` set the team in
possession from ``team``; for the first step all events start with ``team``.
Frames outside the half block have no players, no ball and truth ``X``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from rondo.contracts import MatchData, Meta

Event = Literal[
    "dribble", "pass", "interception", "deflection", "aerial", "out", "restart", "missing"
]
EVENTS = ("dribble", "pass", "interception", "deflection", "aerial", "out", "restart", "missing")

FPS = 25
PITCH_M = (105.3, 68.0)
W, H = PITCH_M
N_PLAYERS = 22
OFFSET = np.array([0.3, 0.0])  # ball position relative to the player holding it
CARRY_SPEED = 5.0  # m/s
PASS_SPEED = 15.0  # minimum ball speed in flight, m/s
OUT_SPEED = 12.0
TAKER_SPEED = 6.0  # m/s, the player who will take a restart jogs to the line
RESTART_LEAD_FRAMES = 88  # 3.5 s
OFFICIAL_FRAMES = 199319

_SPRING = 0.3  # 1/s, pull towards the grid position
_WANDER_DECAY, _WANDER_SIGMA, _MIN_FREE, _MAX_FREE = 0.97, 0.45, 1.6, 6.0
_WANDER_STATIONARY = _WANDER_SIGMA / math.sqrt(1 - _WANDER_DECAY**2)  # per-axis spread, m/s


@dataclass
class Step:
    event: Event
    team: Literal["H", "A"]
    seconds: float


def _slots(team: str) -> range:
    return range(11) if team == "H" else range(11, 22)


def _other(team: str) -> str:
    return "A" if team == "H" else "H"


def _grid(rng: np.random.Generator) -> np.ndarray:
    ys = [10.0, 26.0, 42.0, 58.0]
    home = [(5.0, 34.0)] + [(22.0, y) for y in ys] + [(40.0, y) for y in ys]
    home += [(55.0, 24.0), (55.0, 44.0)]
    home = np.array(home)
    away = home.copy()
    away[:, 0] = W - away[:, 0]
    base = np.vstack([home, away]) + rng.normal(0.0, 1.5, (N_PLAYERS, 2))
    return np.clip(base, [2.0, 2.0], [W - 2.0, H - 2.0])


def _is_outside(point: np.ndarray) -> bool:
    return not (0.0 <= point[0] <= W and 0.0 <= point[1] <= H)


class _World:
    def __init__(self, rng: np.random.Generator, first_team: str, block_start: int):
        self.rng = rng
        self.block_start = block_start
        self.base = _grid(rng)
        self.pos = self.base.copy()
        self.wander = rng.normal(0.0, _WANDER_STATIONARY, (N_PLAYERS, 2))
        self.next_restart: str | None = None  # team of a restart that follows this step
        self.taker: int | None = None
        self.team = first_team  # team in possession (kept while the ball is dead)
        self.label = first_team  # H, A or D
        self.carrier = min(_slots(first_team), key=lambda s: (self._dist(s, (W / 2, H / 2)), s))
        self.ball = self.pos[self.carrier] + OFFSET
        self.pos_rows: list[np.ndarray] = []
        self.ball_rows: list[np.ndarray] = []
        self.labels: list[str] = []
        self.tags: list[dict] = []

    # ---- helpers
    def _dist(self, slot: int, point) -> float:
        return float(np.linalg.norm(self.pos[slot] - np.asarray(point)))

    def _nearest(self, team: str, point) -> int:
        return min(_slots(team), key=lambda s: (self._dist(s, point), s))

    def _closest_to_distance(self, slots, origin, target: float, min_d: float = 4.0) -> int:
        """Slot whose distance from ``origin`` is nearest ``target`` (ignoring ones too close)."""
        slots = list(slots)
        far = [s for s in slots if self._dist(s, origin) >= min_d] or slots
        return min(far, key=lambda s: (abs(self._dist(s, origin) - target), s))

    def _holder(self) -> int:
        """The possessing team's player who has the ball; hands it over if nobody does."""
        c = self.carrier
        if c is not None and c in _slots(self.team) and self._dist(c, self.ball) <= 0.5:
            return c
        c = self._nearest(self.team, self.ball)
        self.ball = self.pos[c] + OFFSET
        return c

    def _move(self, overrides: dict[int, np.ndarray] | None = None, halt_all: bool = False):
        self.wander = _WANDER_DECAY * self.wander + _WANDER_SIGMA * self.rng.normal(
            size=(N_PLAYERS, 2)
        )
        v = _SPRING * (self.base - self.pos) + self.wander
        norm = np.linalg.norm(v, axis=1)
        v *= (np.clip(norm, _MIN_FREE, _MAX_FREE) / np.maximum(norm, 1e-9))[:, None]
        if halt_all:
            v[:] = 0.0
        for slot, vel in (overrides or {}).items():
            v[slot] = vel
        self.pos = np.clip(self.pos + v / FPS, [0.5, 0.5], [W - 0.5, H - 0.5])

    def _record(self, label: str, visible: bool = True):
        self.pos_rows.append(self.pos.copy())
        self.ball_rows.append(self.ball.copy() if visible else np.full(2, np.nan))
        self.labels.append(label)

    def _hold(self, slots, n: int, label: str):
        """``n`` frames with ``slots`` standing still and the ball with the first of them."""
        for _ in range(n):
            self._move({s: np.zeros(2) for s in slots})
            self.ball = self.pos[slots[0]] + OFFSET
            self._record(label)

    def _fly(self, slots, route: list[np.ndarray], frames: list[int], label: str):
        """Ball along a polyline, ``frames[i]`` frames for leg ``i``; ``slots`` stand still."""
        for a, b, f in zip(route[:-1], route[1:], frames):
            for k in range(1, f + 1):
                self._move({s: np.zeros(2) for s in slots})
                self.ball = a + (b - a) * (k / f)
                self._record(label)

    @staticmethod
    def _flight_frames(distance: float, cap: int) -> int:
        return int(np.clip(math.floor(FPS * distance / PASS_SPEED), 1, max(1, cap)))

    def _require_live(self, event: str):
        if self.label == "D":
            raise ValueError(f"{event} after out needs a restart (or dribble) first")

    def _take_possession(self, team: str):
        if team != self.team:
            self.carrier = None
        self.team, self.label = team, team

    # ---- events
    def dribble(self, team: str, n: int):
        self._take_possession(team)
        c = self.carrier
        if c is None or c not in _slots(team) or self._dist(c, self.ball) > 0.5:
            c = self._nearest(team, self.ball)
            self.ball = self.pos[c] + OFFSET
        direction = 1.0 if c < 11 else -1.0
        end_x = self.pos[c][0] + direction * CARRY_SPEED * n / FPS
        if not 2.0 <= end_x <= W - 2.0:
            direction = -direction
        velocity = np.array([direction * CARRY_SPEED, 0.0])
        for _ in range(n):
            self._move({c: velocity})
            self.ball = self.pos[c] + OFFSET
            self._record(team)
        self.carrier = c

    def pass_(self, team: str, n: int):
        self._require_live("pass")
        passer = self._holder()
        receiver = self._closest_to_distance(
            [s for s in _slots(self.team) if s != passer], self.pos[passer], 18.0, min_d=2.0
        )
        dest = self.pos[receiver] + OFFSET
        flight = self._flight_frames(float(np.linalg.norm(dest - self.ball)), n)
        self._hold([passer, receiver], n - flight, self.label)
        self._fly([passer, receiver], [self.ball.copy(), dest], [flight], self.label)
        self.carrier = receiver

    def interception(self, team: str, n: int):
        self._require_live("interception")
        if team == self.team:
            raise ValueError("interception: team must be the team that does not have the ball")
        if n < 2:
            raise ValueError("interception needs at least 2 frames (flight and control)")
        holder = self._holder()
        flight = n // 2  # fixed by the step length, so the truth never depends on the seed
        slowest = PASS_SPEED * flight / FPS  # shortest pass that still flies at PASS_SPEED
        taker = self._closest_to_distance(
            _slots(team), self.pos[holder], slowest * 1.05, min_d=slowest
        )
        dest = self.pos[taker] + OFFSET
        self._fly([holder, taker], [self.ball.copy(), dest], [flight], self.label)
        self._hold([taker, holder], n - flight, team)
        self.team, self.label, self.carrier = team, team, taker

    def deflection(self, team: str, n: int):
        self._require_live("deflection")
        if n < 2:
            raise ValueError("deflection needs at least 2 frames")
        passer = self._holder()
        touch = self._closest_to_distance(_slots(_other(self.team)), self.pos[passer], 12.0)
        receiver = self._closest_to_distance(
            [s for s in _slots(self.team) if s != passer], self.pos[touch], 12.0
        )
        start, via, dest = self.ball.copy(), self.pos[touch].copy(), self.pos[receiver] + OFFSET
        d1, d2 = float(np.linalg.norm(via - start)), float(np.linalg.norm(dest - via))
        f1, f2 = self._flight_frames(d1, n), self._flight_frames(d2, n)
        if f1 + f2 > n:  # squeeze both legs into the step (faster ball)
            f1 = int(np.clip(round(n * d1 / max(d1 + d2, 1e-9)), 1, n - 1))
            f2 = n - f1
        stand = [passer, touch, receiver]
        self._hold([passer, touch, receiver], n - f1 - f2, self.label)
        self._fly(stand, [start, via, dest], [f1, f2], self.label)
        self.carrier = receiver

    def aerial(self, team: str, n: int):
        self._require_live("aerial")
        holder = self._holder()
        pairs = [
            (q, o) for q in _slots(self.team) if q != holder for o in _slots(_other(self.team))
        ]

        def mid(pair):
            return (self.pos[pair[0]] + self.pos[pair[1]]) / 2

        clear = [p for p in pairs if np.linalg.norm(mid(p) - self.pos[holder]) >= 3.0] or pairs
        q, o = min(clear, key=lambda p: (self._dist(p[0], self.pos[p[1]]), p))
        spot = mid((q, o))
        gap = self.pos[q] - self.pos[o]
        unit = gap / max(float(np.linalg.norm(gap)), 1e-9)
        goals = {q: spot + 0.9 * unit, o: spot - 0.9 * unit}
        start = self.ball.copy()
        flight = self._flight_frames(float(np.linalg.norm(spot - start)), n)
        for i in range(n):
            vel = {}
            for s, goal in goals.items():
                v = 2.0 * (goal - self.pos[s])
                vel[s] = v * min(1.0, 8.0 / max(float(np.linalg.norm(v)), 1e-9))
            vel[holder] = np.zeros(2)
            self._move(vel)
            if i < flight:
                self.ball = start + (spot - start) * ((i + 1) / flight)
            else:
                self.ball = spot + np.array([0.0, 0.2 * math.sin(2 * math.pi * (i - flight) / FPS)])
            self._record(self.label)
        self.carrier = q

    def _pick_taker(self, team: str) -> tuple[int, np.ndarray, np.ndarray]:
        """Player of ``team`` closest to a line: (slot, outward unit normal, spot 0.5 m inside)."""
        best = None
        for slot in _slots(team):
            x, y = self.pos[slot]
            for gap, normal, spot in (
                (x, (-1.0, 0.0), (0.5, y)),
                (W - x, (1.0, 0.0), (W - 0.5, y)),
                (y, (0.0, -1.0), (x, 0.5)),
                (H - y, (0.0, 1.0), (x, H - 0.5)),
            ):
                if best is None or gap < best[0]:
                    best = (gap, slot, np.array(normal), np.array(spot))
        return best[1], best[2], best[3]

    def out(self, team: str, n: int):
        b = self.ball.copy()
        taker, spot = None, None
        normal, gap, cross, rest = np.zeros(2), 0.0, 0, b
        if not _is_outside(b):
            cross = min(max(1, n // 4), n - 1)  # first dead frame, fixed by the step length
            if self.next_restart is not None:
                taker, normal, spot = self._pick_taker(self.next_restart)
            else:
                sides = [
                    (b[0], (-1.0, 0.0)),
                    (W - b[0], (1.0, 0.0)),
                    (b[1], (0.0, -1.0)),
                    (H - b[1], (0.0, 1.0)),
                ]
                normal = np.array(min(sides, key=lambda s: s[0])[1])
            axis = 0 if normal[0] else 1
            line = (W, H)[axis] if normal[axis] > 0 else 0.0
            gap = abs(line - b[axis])
            # Rest point: 1.5 m beyond the line, next to the taker when there is one.
            rest = spot + normal * 2.0 if spot is not None else b + normal * (gap + 1.5)
        live = self.label
        for i in range(n):
            dead = i >= cross
            if i < cross:
                u = gap * (i + 1) / (cross + 1) / (gap + 1.5)  # still inside the pitch
            else:
                u = (gap + 1.0 + 0.5 * min(1.0, (i - cross + 1) / 3)) / (gap + 1.5)
            overrides = {}
            if taker is not None:
                to_spot = spot - self.pos[taker]
                dist = float(np.linalg.norm(to_spot))
                if dist > 0:
                    overrides[taker] = to_spot / dist * min(TAKER_SPEED, dist * FPS)
            self._move(overrides, halt_all=dead)
            self.ball = b + (rest - b) * u
            self._record("D" if dead else live)
        self.label, self.carrier, self.taker = "D", None, taker

    def restart(self, team: str, n: int):
        step_start = self.block_start + len(self.labels)
        self._take_possession(team)
        taker = self.taker
        if taker is None or taker not in _slots(team):
            taker = self._nearest(team, self.ball)
        self.taker = None
        for _ in range(n):
            self._move({taker: np.zeros(2)})
            self.ball = self.pos[taker] + OFFSET
            self._record(team)
        self.carrier = taker
        self.tags.append(
            {
                "start_frame": max(self.block_start, step_start - RESTART_LEAD_FRAMES),
                "end_frame": step_start + n,
                "code": "SET PIECES",
                "team": team,
                "type": "Throw-in",
                "half": 1,
            }
        )

    def missing(self, team: str, n: int):
        attach = self.label != "D"
        holder = self._holder() if attach else None
        for _ in range(n):
            self._move()
            if attach:
                self.ball = self.pos[holder] + OFFSET
            self._record(self.label, visible=False)


def make_match(
    steps: list[Step], *, n_frames: int = 3000, start: int = 100, seed: int = 0
) -> tuple[MatchData, pd.Series]:
    """Play ``steps``; return the match (one half block starting at ``start``) and its truth."""
    if not steps:
        raise ValueError("make_match needs at least one step")
    for s in steps:
        if s.event not in EVENTS:
            raise ValueError(f"unknown event {s.event!r}; expected one of {', '.join(EVENTS)}")
        if s.team not in ("H", "A"):
            raise ValueError(f"step team must be 'H' or 'A', got {s.team!r}")
    if steps[0].event == "interception":
        raise ValueError("interception cannot be the first step: nobody has the ball yet")
    lengths = [round(s.seconds * FPS) for s in steps]
    if min(lengths) < 1:
        raise ValueError("every step must last at least one frame")
    total = sum(lengths)
    if start < 0 or start + total > n_frames:
        raise ValueError(f"steps need {total} frames from {start} and do not fit in {n_frames}")

    world = _World(np.random.default_rng(seed), steps[0].team, start)
    for i, (step, n) in enumerate(zip(steps, lengths)):
        following = steps[i + 1] if i + 1 < len(steps) else None
        is_restart = following is not None and following.event == "restart"
        world.next_restart = following.team if is_restart else None
        name = "pass_" if step.event == "pass" else step.event
        getattr(world, name)(step.team, n)

    end = start + total  # exclusive
    pos = np.stack(world.pos_rows)  # (total, 22, 2) metres
    scale = np.array([W, H])
    pos_norm = pos / scale
    metres = pos_norm * scale
    step_len = np.linalg.norm(np.diff(metres, axis=0), axis=2) * FPS
    speed = np.vstack([step_len[:1], step_len]) if total > 1 else np.zeros((1, N_PLAYERS))

    frame_ids = np.arange(n_frames, dtype="int64")
    inside = (frame_ids >= start) & (frame_ids < end)
    half = pd.array([pd.NA] * n_frames, dtype="Int8")
    half[start:end] = 1
    ball = np.full((n_frames, 2), np.nan)
    ball[start:end] = np.stack(world.ball_rows) / scale
    frames = pd.DataFrame(
        {
            "frame": frame_ids,
            "half": half,
            "graded": inside,
            "ball_x": ball[:, 0],
            "ball_y": ball[:, 1],
        }
    )
    players = pd.DataFrame(
        {
            "frame": np.repeat(frame_ids[start:end], N_PLAYERS),
            "slot": np.tile(np.arange(N_PLAYERS, dtype="int16"), total),
            "team": np.tile(np.array(["H"] * 11 + ["A"] * 11), total),
            "x": pos_norm[:, :, 0].ravel(),
            "y": pos_norm[:, :, 1].ravel(),
            "speed": speed.ravel(),
        }
    )
    tags = pd.DataFrame(
        {
            "start_frame": pd.Series([t["start_frame"] for t in world.tags], dtype="int64"),
            "end_frame": pd.Series([t["end_frame"] for t in world.tags], dtype="int64"),
            "code": pd.Series([t["code"] for t in world.tags], dtype="str"),
            "team": pd.Series([t["team"] for t in world.tags], dtype="str"),
            "type": pd.Series([t["type"] for t in world.tags], dtype="str"),
            "half": pd.array([t["half"] for t in world.tags], dtype="Int8"),
        }
    )
    meta = Meta(
        n_frames=n_frames,
        fps=FPS,
        pitch=PITCH_M,
        half_blocks=[(1, start, end - 1)],
        home_name="FC Barcelona",
        away_name="Athletic Club",
    )
    labels = np.full(n_frames, "X", dtype="<U1")
    labels[start:end] = world.labels
    truth = pd.Series(labels, index=pd.RangeIndex(n_frames, name="frame"), name="label")
    return MatchData(frames, players, tags, meta), truth


def full_length(labels: pd.Series, n: int = OFFICIAL_FRAMES) -> pd.Series:
    """Pad a label series (index 0..k-1) with ``X`` to ``n`` frames, as the submission needs."""
    if len(labels) > n:
        raise ValueError(f"labels are longer ({len(labels)}) than the {n} frames of the match")
    if labels.index.tolist() != list(range(len(labels))):
        raise ValueError("labels must be indexed 0..k-1")
    padded = np.full(n, "X", dtype="<U1")
    padded[: len(labels)] = labels.to_numpy()
    return pd.Series(padded, index=pd.RangeIndex(n, name="frame"), name="label")
