"""Viterbi decoder (C4): the most likely ``H/A/D`` sequence for each half, with a cost per change.

Each emission row says how likely each label is on that frame alone. The decoder looks at a whole
half at once and adds a fixed cost every time the label changes, so a brief flicker of evidence
is not worth a switch while sustained evidence is, and the switch lands where that evidence
starts. Frames with no emission row carry no evidence, so the current label simply continues.

Restart anchors: Smart Tagging marks each restart (``SET PIECES``) with a ~5 s window and the
restarting team. Inside such a window, leaving a dead ball towards that team is cheaper by
``restart_bonus`` and towards the other team dearer by the same amount. A restart tag with no
team (throw-ins are tagged ``N/A``) changes nothing.

With ``restart_team_blind`` the tag's team is ignored: inside every restart window, team or not,
leaving a dead ball towards either team is cheaper by ``restart_bonus``. The window then says only
that play resumes here, and the evidence decides who has the ball.

Known limitation: after a stoppage with no team on the restart tag, the team that resumes play is
decided by the evidence alone, and an exact tie goes to ``H`` (the earlier state).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from rondo.contracts import Meta, is_restart_code
from rondo.registry import register_decoder

LABELS = ("H", "A", "D")  # state order
H, A, D = 0, 1, 2
TIE_TOLERANCE = 1e-6  # far below any real evidence, far above rounding in the running scores
MIN_PROB = 1e-9
NO_ANCHOR = -1
ANY_TEAM = 2  # anchor value of a team-blind restart window


@dataclass(frozen=True)
class ViterbiParams:
    """Transition costs, in the same log-probability units as the emissions.

    Checked on construction: every cost is finite and >= 0, ``restart_bonus`` stays below
    ``dead_exit_cost`` (so no exit is free), and ``switch_cost`` stays below
    ``max_switch_cost()``, the price of the cheapest detour through one dead frame.
    """

    switch_cost: float = 8.0  # H <-> A
    dead_entry_cost: float = 25.0  # H or A -> D
    dead_exit_cost: float = 25.0  # D -> H or A
    restart_bonus: float = 6.0  # D -> the tagged team, inside its restart window
    restart_team_blind: bool = False  # the bonus goes to D -> either team, in every window

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.type in (bool, "bool"):
                continue
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"viterbi {field.name} must be finite and >= 0, got {value}")
        if self.restart_bonus >= self.dead_exit_cost:
            raise ValueError(
                f"viterbi restart_bonus ({self.restart_bonus}) must be below dead_exit_cost "
                f"({self.dead_exit_cost}), or leaving a dead ball becomes free"
            )
        if self.switch_cost >= self.max_switch_cost():
            raise ValueError(
                f"viterbi switch_cost ({self.switch_cost}) must be below "
                f"{self.max_switch_cost():.2f} (dead_entry_cost + dead_exit_cost - restart_bonus "
                f"+ log({MIN_PROB:g})), or a team change can route through a one-frame D"
            )

    def max_switch_cost(self) -> float:
        """Above this, H -> D -> A over a single frame of certain ``D`` evidence (whose gain over
        a clipped team probability is ``-log(MIN_PROB)``, about 20.7) beats a direct H -> A inside
        a restart window, leaving a phantom one-frame stoppage at the team change."""
        return self.dead_entry_cost + self.dead_exit_cost - self.restart_bonus + math.log(MIN_PROB)

    @classmethod
    def from_dict(cls, params: dict) -> ViterbiParams:
        """Build from ``decoder_params``; an unknown name is an error, not silently ignored."""
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(params) - known)
        if unknown:
            raise ValueError(
                f"unknown viterbi parameter(s): {', '.join(unknown)}; "
                f"known: {', '.join(sorted(known))}"
            )
        flags = {field.name for field in fields(cls) if field.type in (bool, "bool")}
        for name in flags & set(params):
            if not isinstance(params[name], bool):
                # ValueError like every other bad config value, so callers handle one type
                raise ValueError(f"viterbi {name} must be true or false, got {params[name]!r}")  # noqa: TRY004
        return cls(**{name: bool(v) if name in flags else float(v) for name, v in params.items()})

    def transitions(self) -> np.ndarray:
        """``T[i, j]``: score added when the label goes from state ``i`` to state ``j``."""
        switch, entry, exit_ = self.switch_cost, self.dead_entry_cost, self.dead_exit_cost
        return -np.array(
            [[0.0, switch, entry], [switch, 0.0, entry], [exit_, exit_, 0.0]], dtype="float64"
        )


@register_decoder("viterbi")
def decode(emissions: pd.DataFrame, tags: pd.DataFrame, meta: Meta, params: dict) -> pd.DataFrame:
    """``frame, label`` on every frame of each half block, each half decoded on its own."""
    config = ViterbiParams.from_dict(params)
    base = config.transitions()
    anchored = {team: _anchored(base, team, config.restart_bonus) for team in (H, A)}
    anchored[ANY_TEAM] = _anchored_any(base, config.restart_bonus)
    anchor = _restart_anchor(tags, meta.n_frames, blind=config.restart_team_blind)
    by_frame = emissions.set_index("frame")[["p_H", "p_A", "p_D"]]

    frames_out, labels_out = [], []
    for _, first, last in meta.half_blocks:
        frames = np.arange(first, last + 1, dtype="int64")
        probs = by_frame.reindex(frames).to_numpy(dtype="float64")
        log_e = np.log(np.clip(probs, MIN_PROB, 1.0))
        log_e[np.isnan(probs).any(axis=1)] = 0.0  # no row: no evidence either way
        path = _best_path(log_e, base, anchored, anchor[first : last + 1])
        frames_out.append(frames)
        labels_out.append(np.array(LABELS)[path])

    frames = np.concatenate(frames_out) if frames_out else np.array([], dtype="int64")
    labels = np.concatenate(labels_out) if labels_out else np.array([], dtype=object)
    return pd.DataFrame({"frame": frames, "label": pd.Series(labels, dtype="str")})


def _anchored(base: np.ndarray, team: int, bonus: float) -> np.ndarray:
    matrix = base.copy()
    matrix[D, team] += bonus
    matrix[D, A if team == H else H] -= bonus
    return matrix


def _anchored_any(base: np.ndarray, bonus: float) -> np.ndarray:
    matrix = base.copy()
    matrix[D, H] += bonus
    matrix[D, A] += bonus
    return matrix


def _restart_anchor(tags: pd.DataFrame, n_frames: int, blind: bool = False) -> np.ndarray:
    """Per frame, the team of the restart window covering it (``H``/``A``), else ``NO_ANCHOR``;
    with ``blind``, every restart window (team or not) is ``ANY_TEAM``.

    Windows are painted in start order, so where two overlap the later one wins.
    """
    anchor = np.full(n_frames, NO_ANCHOR, dtype=np.int8)
    if blind:
        restarts = tags[is_restart_code(tags["code"])]
        for start, end in sorted(zip(restarts["start_frame"], restarts["end_frame"], strict=True)):
            anchor[max(int(start), 0) : min(int(end), n_frames)] = ANY_TEAM
        return anchor
    restarts = tags[is_restart_code(tags["code"]) & tags["team"].isin(["H", "A"])]
    for start, end, team in sorted(
        zip(restarts["start_frame"], restarts["end_frame"], restarts["team"], strict=True)
    ):
        anchor[max(int(start), 0) : min(int(end), n_frames)] = H if team == "H" else A
    return anchor


def _best_path(
    log_e: np.ndarray, base: np.ndarray, anchored: dict[int, np.ndarray], anchor: np.ndarray
) -> np.ndarray:
    """State index per frame on the highest-scoring path.

    While the ball travels between teams, H and A are level, so every moment to switch scores the
    same. Such a tie goes to the latest moment, where the new team gains control, whichever team
    that is: walking the path backwards, a switch is preferred to staying when they are within
    ``TIE_TOLERANCE`` (scores that should be equal differ by rounding). A tie over the final label
    goes to the earlier state.
    """
    n = len(log_e)
    path = np.empty(n, dtype=np.int8)
    if n == 0:
        return path
    states = np.arange(len(LABELS))
    prefer_switch = np.eye(len(LABELS)) * TIE_TOLERANCE
    back = np.zeros((n, len(LABELS)), dtype=np.int8)
    delta = log_e[0].copy()
    for t in range(1, n):
        transition = base if anchor[t] == NO_ANCHOR else anchored[int(anchor[t])]
        scores = delta[:, None] + transition
        best = (scores - prefer_switch).argmax(axis=0)
        back[t] = best
        delta = scores[best, states] + log_e[t]
    state = int(delta.argmax())
    path[-1] = state
    for t in range(n - 1, 0, -1):
        state = int(back[t, state])
        path[t - 1] = state
    return path
