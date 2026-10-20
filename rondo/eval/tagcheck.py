"""How well Smart Tagging lines up with the tracking and with a truth window: aggregates only.

Smart Tagging is automatic, so its restart windows and team hints are evidence, not answers.
This report says how far to trust them on the real match before tuning their weights:

- ``restart_windows``: set-piece tags; ``restart_windows_with_team_pct``: the share naming a team
  (throw-ins are tagged ``N/A``).
- ``restart_windows_ball_out_pct``: the share whose window contains a frame with the ball out of
  bounds in the tracking, i.e. the tag sits on a real stoppage.
- ``restart_team_vs_truth_pct`` (``_n`` restarts checked): how often the restarting team is the
  team in possession in the truth right after the window.
- ``prior_coverage_graded_pct``: graded frames with a team hint (``tag_team_prior``);
  ``prior_vs_truth_pct``: how often the hint names the truth's team, on frames where the truth
  has a team.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rondo import registry
from rondo.contracts import MatchData, is_restart_code
from rondo.pipeline import load_components

AFTER_RESTART_FRAMES = 250  # how far past a restart window to look for the team in possession


def _pct(hits: int, total: int) -> float | None:
    return round(100.0 * hits / total, 1) if total else None


def tag_report(match: MatchData, truth: pd.Series | None) -> dict:
    """Counts and agreement rates for the Smart Tagging of ``match``; ``truth`` (full-length
    ``H/A/D/X`` labels, ``X`` where unknown) is optional."""
    load_components()
    n = match.meta.n_frames
    tags = match.tags
    restarts = tags[is_restart_code(tags["code"])]
    in_bounds = registry.SIGNALS["ball_bounds"](match)["ball_in_bounds"].astype("boolean")
    out = (~in_bounds).fillna(False).to_numpy(dtype=bool)  # missing ball: not seen out

    ball_out = sum(
        bool(out[max(0, s) : min(n, e)].any())
        for s, e in zip(restarts["start_frame"], restarts["end_frame"])
    )
    graded = np.zeros(n, dtype=bool)
    for _, first, last in match.meta.half_blocks:
        graded[first : last + 1] = True
    prior = registry.SIGNALS["tag_team_prior"](match)["tag_team_prior"].to_numpy(dtype=object)
    has_prior = ~pd.isna(prior)

    report = {
        "restart_windows": len(restarts),
        "restart_windows_with_team_pct": _pct(int(restarts["team"].notna().sum()), len(restarts)),
        "restart_windows_ball_out_pct": _pct(ball_out, len(restarts)),
        "restart_team_vs_truth_n": None,
        "restart_team_vs_truth_pct": None,
        "prior_coverage_graded_pct": _pct(int((has_prior & graded).sum()), int(graded.sum())),
        "prior_vs_truth_pct": None,
    }
    if truth is None:
        return report

    labels = truth.to_numpy(dtype=object)
    team_truth = np.isin(labels, ["H", "A"])
    checked = agree = 0
    for end, team in zip(restarts["end_frame"], restarts["team"]):
        if pd.isna(team):
            continue
        after = np.flatnonzero(team_truth[end : min(n, end + AFTER_RESTART_FRAMES)])
        if len(after):
            checked += 1
            agree += labels[end + after[0]] == team
    report["restart_team_vs_truth_n"] = checked
    report["restart_team_vs_truth_pct"] = _pct(agree, checked)

    both = team_truth & has_prior
    report["prior_vs_truth_pct"] = _pct(int((prior[both] == labels[both]).sum()), int(both.sum()))
    return report
