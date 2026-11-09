"""The submitted settings, frozen for the code freeze of 2026-11-14.

Every setting the pipeline runs with, after ``configs/default.toml`` is applied over the code's
defaults. A change to either fails here, so nothing moves after the freeze by accident. Each value
was chosen on the three public sample games (``rondo othergame``), never on the provided match.
"""

import tomllib
from dataclasses import asdict
from pathlib import Path

from rondo.decoder.viterbi import ViterbiParams
from rondo.emissions.base import build_params
from rondo.emissions.rules import RulesParams

DEFAULT = Path(__file__).resolve().parent.parent / "configs" / "default.toml"

FROZEN_COMPONENTS = {
    "signals": [
        "ball_nearest",
        "ball_motion",
        "ball_bounds",
        "team_motion",
        "tag_restarts",
        "tag_team_prior",
    ],
    "emission": "rules_v0",
    "decoder": "viterbi",
}

FROZEN_EMISSION = {
    "control_dist_m": 1.5,
    "control_speed_ms": 8.0,
    "control_weight": 2.0,
    "min_control_frames": 25,
    "max_control_gap_frames": 75,
    "contest_margin_m": 0.0,
    "out_weight": 5.0,
    "out_carry_weight": 5.0,
    "stopped_speed_ms": 1.0,
    "still_ball_ms": 0.5,
    "stopped_weight": 4.0,
    "prior_weight": 0.3,
    "alive_margin": 2.0,
    "temperature": 1.0,
}

FROZEN_DECODER = {
    "switch_cost": 8.0,
    "dead_entry_cost": 25.0,
    "dead_exit_cost": 25.0,
    "restart_bonus": 6.0,
    "restart_team_blind": False,
}


def load_default() -> dict:
    return tomllib.loads(DEFAULT.read_text(encoding="utf-8"))


def test_the_default_runs_the_frozen_components():
    config = load_default()
    assert {key: config[key] for key in FROZEN_COMPONENTS} == FROZEN_COMPONENTS


def test_the_emission_runs_with_the_frozen_settings():
    params = build_params(RulesParams, load_default()["emission_params"], "rules_v0")
    assert asdict(params) == FROZEN_EMISSION


def test_the_decoder_runs_with_the_frozen_settings():
    params = ViterbiParams.from_dict(load_default()["decoder_params"])
    assert asdict(params) == FROZEN_DECODER
