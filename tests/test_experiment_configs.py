"""In-play experiment configs: each is the default with exactly one setting changed."""

import tomllib
from pathlib import Path

import pytest

from rondo.decoder.viterbi import ViterbiParams
from rondo.emissions.base import build_params
from rondo.emissions.rules import RulesParams

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

# name -> (section, setting, value)
IN_PLAY = {
    "switch-12": ("decoder_params", "switch_cost", 12.0),
    "switch-16": ("decoder_params", "switch_cost", 16.0),
    "switch-20": ("decoder_params", "switch_cost", 20.0),
    "control-75": ("emission_params", "min_control_frames", 75),
    "control-100": ("emission_params", "min_control_frames", 100),
    "contest-0.5": ("emission_params", "contest_margin_m", 0.5),
    "contest-1.0": ("emission_params", "contest_margin_m", 1.0),
}


def load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(IN_PLAY))
def test_an_in_play_config_is_the_default_with_one_setting_changed(name):
    section, setting, value = IN_PLAY[name]
    default = load(CONFIGS / "default.toml")
    config = load(CONFIGS / "experiments" / f"{name}.toml")

    expected = {key: dict(val) if isinstance(val, dict) else val for key, val in default.items()}
    expected[section][setting] = value
    assert config == expected


@pytest.mark.parametrize("name", sorted(IN_PLAY))
def test_an_in_play_config_is_accepted_by_the_decoder_and_the_emission(name):
    config = load(CONFIGS / "experiments" / f"{name}.toml")
    ViterbiParams.from_dict(config["decoder_params"])
    build_params(RulesParams, config["emission_params"], "rules_v0")
