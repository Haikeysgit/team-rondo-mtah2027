"""Rules emission v0 (C3): hand-set evidence weights, turned into probabilities by a softmax.

Every frame starts level between ``H`` and ``A``, with ``D`` ``alive_margin`` below them: play is
alive unless something says otherwise. Evidence then adds to one label's score:

- **Control**: the nearest player is within ``control_dist_m`` of a ball moving slower than
  ``control_speed_ms`` (a pass in flight is nobody's control) -> ``control_weight`` to that team,
  but only when the team keeps it: its run of control, passes between teammates included, spans
  at least ``min_control_frames`` (2 s). The organisers' hint and our tactical analyst agree that
  control means a few seconds on the ball; a touch, block or deflection gives no evidence at all,
  so it cannot flip possession. The run is judged on the whole match, so its evidence still starts
  at the frame the team gained the ball, where the decoder places the switch.
- **Ball out of bounds** -> ``out_weight`` to ``D``. Control is not counted while the ball is out.
- **Stoppage**: players are slow on average (``team_speed_mean_m_s < stopped_speed_ms``) *and*
  the ball is not moving like play (missing, slower than ``still_ball_ms``, or inside a Smart
  Tagging restart window) -> ``stopped_weight`` to ``D``. Slow players alone are not enough: a
  slow build-up with the ball being passed is still play, and a false short ``D`` would split the
  possession in spellF1.
- **Smart Tagging prior**: the team in possession implied by the covering phase ->
  ``prior_weight`` to that team, a lean rather than a decision.

With the ball missing and no stoppage, ``H`` and ``A`` stay exactly level, so the decoder carries
the current possession through. Scores are divided by ``temperature`` before the softmax.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from rondo.emissions.base import build_params, require_columns
from rondo.registry import register_emission

H, A, D = 0, 1, 2
COLUMNS = (
    "nearest_team",
    "nearest_dist_m",
    "ball_speed_ms",
    "ball_in_bounds",
    "ball_missing_run",
    "team_speed_mean_m_s",
    "tag_restart_window",
    "tag_team_prior",
)


@dataclass(frozen=True)
class RulesParams:
    """Distances in metres, speeds in metres per second, weights in log-probability units."""

    control_dist_m: float = 1.5
    control_speed_ms: float = 8.0
    control_weight: float = 2.0
    min_control_frames: int = 50
    max_control_gap_frames: int = 75
    out_weight: float = 5.0
    stopped_speed_ms: float = 1.0
    still_ball_ms: float = 0.5
    stopped_weight: float = 4.0
    prior_weight: float = 0.3
    alive_margin: float = 2.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"rules_v0 temperature must be > 0, got {self.temperature}")


def _sustained(control: np.ndarray, team: np.ndarray, min_frames: int, max_gap: int) -> np.ndarray:
    """True on control frames that belong to a team's sustained run.

    A run is one team's control frames with no control by the other team between them and no gap
    longer than ``max_gap`` frames (passes between teammates and short tracking losses do not end
    it). It is sustained when it spans at least ``min_frames``, first to last control frame.
    """
    idx = np.flatnonzero(control)
    keep = np.zeros(len(control), dtype=bool)
    if len(idx) == 0:
        return keep
    teams = team[idx]
    new_run = np.ones(len(idx), dtype=bool)
    new_run[1:] = (teams[1:] != teams[:-1]) | (np.diff(idx) > max_gap + 1)
    run_id = np.cumsum(new_run) - 1
    starts = idx[new_run]
    ends = np.maximum.reduceat(idx, np.flatnonzero(new_run))
    keep[idx] = (ends - starts + 1)[run_id] >= min_frames
    return keep


@register_emission("rules_v0")
class RulesV0:
    """Rule-based emission; the parameters come from ``[emission_params]`` in the config."""

    def __init__(self, **params: float) -> None:
        self.params = build_params(RulesParams, params, "rules_v0")

    def fit(self, signals: pd.DataFrame, labels: pd.Series) -> RulesV0:
        """Nothing to learn: the weights are set by hand and tuned on the validation windows."""
        return self

    def predict(self, signals: pd.DataFrame) -> pd.DataFrame:
        require_columns(signals, COLUMNS, "rules_v0")
        p = self.params
        n = len(signals)

        missing = signals["ball_missing_run"].to_numpy(dtype="float64") > 0
        speed = signals["ball_speed_ms"].to_numpy(dtype="float64")
        dist = signals["nearest_dist_m"].to_numpy(dtype="float64")
        nearest = signals["nearest_team"].to_numpy(dtype=object)
        out = (~signals["ball_in_bounds"].astype("boolean")).fillna(False).to_numpy(dtype=bool)
        players = signals["team_speed_mean_m_s"].to_numpy(dtype="float64")
        restart = signals["tag_restart_window"].astype("boolean").fillna(False).to_numpy(bool)
        prior = signals["tag_team_prior"].to_numpy(dtype=object)

        scores = np.zeros((n, 3), dtype="float64")
        scores[:, D] = -p.alive_margin

        # NaN speed counts as slow here: the ball is seen, only its speed is unknown (track edges).
        control = ~missing & ~out & (dist <= p.control_dist_m) & ~(speed >= p.control_speed_ms)
        control &= _sustained(control, nearest, p.min_control_frames, p.max_control_gap_frames)
        for team, label in ((H, "H"), (A, "A")):
            scores[:, team] += p.control_weight * (control & (nearest == label))
            scores[:, team] += p.prior_weight * (prior == label)

        scores[:, D] += p.out_weight * out
        ball_still = missing | (speed < p.still_ball_ms) | restart
        scores[:, D] += p.stopped_weight * ((players < p.stopped_speed_ms) & ball_still)

        scaled = scores / p.temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        weights = np.exp(scaled)
        probs = weights / weights.sum(axis=1, keepdims=True)
        return pd.DataFrame(
            {
                "frame": signals["frame"].to_numpy(dtype="int64"),
                "p_H": probs[:, H],
                "p_A": probs[:, A],
                "p_D": probs[:, D],
            }
        )

    def __repr__(self) -> str:
        return f"RulesV0({', '.join(f'{k}={v}' for k, v in asdict(self.params).items())})"
