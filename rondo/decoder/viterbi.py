"""Viterbi decoder (C4): the most likely ``H/A/D`` sequence for each half, with a cost per change.

Each emission row says how likely each label is on that frame alone. The decoder looks at a whole
half at once and adds a fixed cost every time the label changes, so a brief flicker of evidence
is not worth a switch while sustained evidence is, and the switch lands where that evidence
starts. Frames with no emission row carry no evidence, so the current label simply continues.

Restart anchors: Smart Tagging marks each restart (``SET PIECES``) with a ~5 s window and the
restarting team. Inside such a window, leaving a dead ball towards that team is cheaper by
``restart_bonus`` and towards the other team dearer by the same amount. A restart tag with no
team (throw-ins are tagged ``N/A``) changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from rondo.contracts import Meta, is_restart_code
from rondo.registry import register_decoder

LABELS = ("H", "A", "D")  # state order; ties go to the earlier state
H, A, D = 0, 1, 2
MIN_PROB = 1e-9
NO_ANCHOR = -1


@dataclass(frozen=True)
class ViterbiParams:
    """Transition costs, in the same log-probability units as the emissions."""

    switch_cost: float = 8.0  # H <-> A
    dead_entry_cost: float = 25.0  # H or A -> D
    dead_exit_cost: float = 25.0  # D -> H or A
    restart_bonus: float = 6.0  # D -> the tagged team, inside its restart window

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
        return cls(**{name: float(value) for name, value in params.items()})

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
    anchor = _restart_anchor(tags, meta.n_frames)
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


def _restart_anchor(tags: pd.DataFrame, n_frames: int) -> np.ndarray:
    """Per frame, the team of the restart window covering it (``H``/``A``), else ``NO_ANCHOR``.

    Windows are painted in start order, so where two overlap the later one wins.
    """
    anchor = np.full(n_frames, NO_ANCHOR, dtype=np.int8)
    restarts = tags[is_restart_code(tags["code"]) & tags["team"].isin(["H", "A"])]
    for start, end, team in sorted(
        zip(restarts["start_frame"], restarts["end_frame"], restarts["team"], strict=True)
    ):
        anchor[max(int(start), 0) : min(int(end), n_frames)] = H if team == "H" else A
    return anchor


def _best_path(
    log_e: np.ndarray, base: np.ndarray, anchored: dict[int, np.ndarray], anchor: np.ndarray
) -> np.ndarray:
    """State index per frame on the highest-scoring path; ties go to the earlier state."""
    n = len(log_e)
    path = np.empty(n, dtype=np.int8)
    if n == 0:
        return path
    states = np.arange(len(LABELS))
    back = np.zeros((n, len(LABELS)), dtype=np.int8)
    delta = log_e[0].copy()
    for t in range(1, n):
        transition = base if anchor[t] == NO_ANCHOR else anchored[int(anchor[t])]
        scores = delta[:, None] + transition
        best = scores.argmax(axis=0)
        back[t] = best
        delta = scores[best, states] + log_e[t]
    state = int(delta.argmax())
    path[-1] = state
    for t in range(n - 1, 0, -1):
        state = int(back[t, state])
        path[t - 1] = state
    return path
