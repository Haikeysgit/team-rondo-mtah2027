"""End-to-end possession scenarios: one test per edge case in the possession rules doc.

Each test builds a synthetic match for one scenario, runs it through the default pipeline
(every component wired through ``configs/default.toml``, the same chain ``rondo run`` uses), and
checks the recovered possession against the synthetic truth.

Two edge cases are not covered here because ``tests.synthetic.make_match`` has no event for them:
a stoppage where the ball never leaves the pitch (foul, offside, injury) and a brief stop under a
second that must not split a possession. Both need the ball to go still or missing while staying
in bounds for only a few frames, which only ``out`` can produce, and ``out`` always sends the ball
across a line. They are exercised directly against the emission and decoder instead.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd

from rondo.eval import score
from rondo.pipeline import run
from tests.synthetic import Step, make_match

CONFIG = tomllib.loads(
    (Path(__file__).resolve().parent.parent / "configs" / "default.toml").read_text()
)


def _run(steps: list[Step], n_frames: int = 1500) -> tuple[pd.Series, pd.Series]:
    match, truth = make_match(steps, n_frames=n_frames)
    pred = run(match, CONFIG).set_index("frame")["label"]
    return pred, truth


def test_pass_between_teammates_keeps_the_same_team():
    # Also the "players slow down but the ball is still being passed" case: the two players
    # involved in a pass stand still while it's in flight, but the ball itself is never still
    # (it travels at >= 15 m/s), so no stoppage evidence fires.
    pred, truth = _run([Step("dribble", "H", 3), Step("pass", "H", 2), Step("dribble", "H", 3)])
    assert (pred == truth).all()
    assert not (pred == "D").any()
    assert score(pred, truth)["spell_f1_mean"] == 1.0


def test_interception_switches_possession_where_control_began():
    # A short lead-in can land the receiver within control_dist_m on the flight's last frame
    # (the ball reaches him exactly there), a frame early by the truth's own definition; 4s
    # gives the players enough room that this doesn't happen.
    pred, truth = _run(
        [Step("dribble", "H", 4), Step("interception", "A", 3), Step("dribble", "A", 3)]
    )
    assert (pred == truth).all()
    assert score(pred, truth)["spell_f1_mean"] == 1.0


def test_a_deflection_by_the_other_team_does_not_change_possession():
    pred, truth = _run(
        [Step("dribble", "H", 3), Step("deflection", "H", 2), Step("dribble", "H", 3)]
    )
    assert (pred == truth).all()


def test_a_header_won_without_control_does_not_change_possession():
    # Also covers a goalkeeper punch (the ball stays live in the air, nobody controls it) and
    # advantage played (a contested moment that never produces stoppage or turnover evidence).
    pred, truth = _run([Step("dribble", "H", 3), Step("aerial", "H", 2), Step("dribble", "H", 3)])
    assert (pred == truth).all()


def test_an_aerial_duel_then_sustained_control_switches_possession():
    # A header that is won is not control by itself (the test above); only once a player of the
    # other team keeps the ball does possession change, and only from where that control began.
    # Same 4s lead-in as the interception test, and for the same reason.
    pred, truth = _run([Step("dribble", "H", 4), Step("aerial", "H", 2), Step("dribble", "A", 3)])
    assert (pred == truth).all()
    assert score(pred, truth)["spell_f1_mean"] == 1.0


def test_ball_wholly_over_the_line_is_dead_from_that_frame():
    pred, truth = _run([Step("dribble", "H", 3), Step("out", "H", 6)])
    assert (pred == truth).all()
    assert (pred == "D").any()


def test_a_restart_belongs_to_the_restarting_team_from_the_kick_or_throw():
    steps = [
        Step("dribble", "H", 3),
        Step("out", "H", 6),
        Step("restart", "H", 3),
        Step("dribble", "H", 3),
    ]
    pred, truth = _run(steps)
    assert (pred == truth).all()
    assert score(pred, truth)["spell_f1_mean"] == 1.0


def test_a_goal_is_dead_until_the_restarting_team_takes_the_kick_off():
    # Mechanically a restart like any other, but its own edge case in the doc: a turnover through
    # a dead ball (the team that conceded does not get the restart).
    steps = [
        Step("dribble", "A", 3),
        Step("out", "A", 6),
        Step("restart", "H", 3),
        Step("dribble", "H", 3),
    ]
    pred, truth = _run(steps)
    assert (pred == truth).all()
    assert score(pred, truth)["spell_f1_mean"] == 1.0


def test_ball_missing_inside_a_possession_does_not_change_the_label():
    pred, truth = _run([Step("dribble", "H", 3), Step("missing", "H", 2), Step("dribble", "H", 3)])
    assert (pred == truth).all()


def test_a_throw_in_tagged_na_is_decided_by_tracking_not_the_tag():
    steps = [
        Step("dribble", "H", 3),
        Step("out", "H", 6),
        Step("restart", "A", 3),
        Step("dribble", "A", 3),
    ]
    match, truth = make_match(steps, n_frames=1500)
    match.tags["team"] = pd.Series([pd.NA] * len(match.tags), dtype="str")  # every tag is N/A
    pred = run(match, CONFIG).set_index("frame")["label"]
    assert (pred == truth).all()
    assert score(pred, truth)["spell_f1_mean"] == 1.0


def test_a_clean_match_recovers_every_spell_exactly():
    """A full passage of play through most edge cases at once: the clean case."""
    pred, truth = _run(
        [
            Step("dribble", "H", 4),
            Step("pass", "H", 2),
            Step("interception", "A", 3),
            Step("dribble", "A", 3),
            Step("deflection", "A", 2),
            Step("dribble", "A", 3),
            Step("out", "A", 6),
            Step("restart", "H", 3),
            Step("dribble", "H", 4),
            Step("missing", "H", 2),
            Step("dribble", "H", 3),
        ],
        n_frames=1500,
    )
    assert (pred == truth).all()
    assert score(pred, truth)["spell_f1_mean"] == 1.0
