"""Rules emission v0: per-frame H/A/D probabilities from the ball, player and tag signals."""

import numpy as np
import pandas as pd
import pytest

import rondo.emissions  # noqa: F401  (registers "rules_v0")
from rondo import registry
from rondo.contracts import Meta, validate_emissions
from rondo.emissions.rules import RulesParams, RulesV0

# One frame of open play: ball visible and rolling slowly, nobody close to it, players moving.
OPEN_PLAY = {
    "nearest_team": "H",
    "nearest_dist_m": 6.0,
    "second_team_dist_m": 9.0,
    "ball_speed_ms": 3.0,
    "ball_in_bounds": True,
    "ball_missing_run": 0,
    "team_moving_H": 6,
    "team_moving_A": 5,
    "team_speed_mean_m_s": 2.5,
    "tag_restart_window": False,
    "tag_restart_team": None,
    "tag_team_prior": None,
}
BALL_MISSING = {
    "nearest_team": None,
    "nearest_dist_m": np.nan,
    "second_team_dist_m": np.nan,
    "ball_speed_ms": np.nan,
    "ball_in_bounds": pd.NA,
    "ball_missing_run": 12,
}


def signals(*rows: dict) -> pd.DataFrame:
    """One frame per row, each row ``OPEN_PLAY`` with the given changes, in the pipeline's dtypes."""
    full = [{**OPEN_PLAY, **row} for row in rows]
    table = pd.DataFrame(full)
    for column in ("nearest_team", "tag_restart_team", "tag_team_prior"):
        values = [np.nan if v is None else v for v in table[column]]
        table[column] = pd.Series(values, dtype="str")
    table["ball_in_bounds"] = table["ball_in_bounds"].astype("boolean")
    table["ball_missing_run"] = table["ball_missing_run"].astype("int64")
    table["tag_restart_window"] = table["tag_restart_window"].astype(bool)
    table.insert(0, "frame", np.arange(len(table), dtype="int64"))
    return table


def predict(table: pd.DataFrame, **params) -> pd.DataFrame:
    return RulesV0(**params).predict(table)


def probs(row: dict, **params) -> tuple[float, float, float]:
    out = predict(signals(row), **params).iloc[0]
    return out["p_H"], out["p_A"], out["p_D"]


HELD = 60  # frames: longer than the default min_control_frames (50, 2 s)


def held_probs(row: dict, **params) -> tuple[float, float, float]:
    """Probabilities in the middle of ``HELD`` frames of ``row``: control needs a sustained run."""
    out = predict(signals(*[row] * HELD), **params).iloc[HELD // 2]
    return out["p_H"], out["p_A"], out["p_D"]


def decode(rows: list[dict]) -> list[str]:
    import rondo.decoder  # noqa: F401  (registers "viterbi")

    table = signals(*rows)
    meta = Meta(len(table), 25, (105.3, 68.0), [(1, 0, len(table) - 1)], "FC Barcelona", "Athletic")
    no_tags = pd.DataFrame(
        {
            "start_frame": pd.Series([], dtype="int64"),
            "end_frame": pd.Series([], dtype="int64"),
            "code": pd.Series([], dtype="str"),
            "team": pd.Series([], dtype="str"),
            "type": pd.Series([], dtype="str"),
            "half": pd.array([], dtype="Int8"),
        }
    )
    return registry.DECODERS["viterbi"](predict(table), no_tags, meta, {})["label"].tolist()


H_CONTROL = {"nearest_team": "H", "nearest_dist_m": 0.7, "ball_speed_ms": 2.0}
A_CONTROL = {"nearest_team": "A", "nearest_dist_m": 0.7, "ball_speed_ms": 2.0}
IN_FLIGHT = {"nearest_team": "H", "nearest_dist_m": 6.0, "ball_speed_ms": 14.0}


# -- the plan's cases ----------------------------------------------------------------------------


def test_a_controlled_ball_near_an_h_player_is_h():
    assert held_probs({"nearest_team": "H", "nearest_dist_m": 0.8, "ball_speed_ms": 2.0})[0] > 0.8


def test_a_controlled_ball_near_an_a_player_is_a():
    assert held_probs({"nearest_team": "A", "nearest_dist_m": 0.8, "ball_speed_ms": 2.0})[1] > 0.8


def test_a_ball_out_of_bounds_is_dead():
    row = {"ball_in_bounds": False, "nearest_team": "H", "nearest_dist_m": 0.8}
    assert probs(row)[2] > 0.8


def test_ball_missing_with_players_moving_is_either_team_and_alive():
    p_h, p_a, p_d = probs(BALL_MISSING)
    assert abs(p_h - p_a) < 1e-9 and p_d < 0.2


def test_ball_missing_with_players_standing_is_dead():
    assert probs({**BALL_MISSING, "team_speed_mean_m_s": 0.3})[2] > 0.6


def test_slow_players_passing_the_ball_is_not_a_stoppage():
    row = {"team_speed_mean_m_s": 0.6, "nearest_team": "H", "nearest_dist_m": 0.8}
    assert probs({**row, "ball_speed_ms": 9.0})[2] < 0.3


# -- the stoppage rule in detail ------------------------------------------------------------------


def test_slow_players_and_a_still_ball_is_a_stoppage():
    row = {"team_speed_mean_m_s": 0.6, "ball_speed_ms": 0.2, "nearest_dist_m": 4.0}
    assert probs(row)[2] > 0.6


def test_slow_players_inside_a_restart_window_is_a_stoppage():
    row = {"team_speed_mean_m_s": 0.6, "ball_speed_ms": 3.0, "tag_restart_window": True}
    assert probs(row)[2] > 0.6


def test_a_still_ball_with_players_moving_is_not_a_stoppage():
    row = {"team_speed_mean_m_s": 2.5, "ball_speed_ms": 0.2, "nearest_dist_m": 4.0}
    assert probs(row)[2] < 0.2


def test_no_tracked_players_is_not_a_stoppage():
    assert probs({**BALL_MISSING, "team_speed_mean_m_s": np.nan})[2] < 0.2


# -- control and priors --------------------------------------------------------------------------


def test_a_fast_ball_near_a_player_is_not_control():
    p_h, p_a, _ = probs({"nearest_team": "H", "nearest_dist_m": 0.8, "ball_speed_ms": 12.0})
    assert abs(p_h - p_a) < 1e-9


def test_a_player_too_far_from_the_ball_is_not_control():
    p_h, p_a, _ = probs({"nearest_team": "H", "nearest_dist_m": 2.5, "ball_speed_ms": 2.0})
    assert abs(p_h - p_a) < 1e-9


def test_the_tag_prior_leans_towards_its_team_without_deciding():
    p_h, p_a, _ = probs({**BALL_MISSING, "tag_team_prior": "A"})
    assert p_a > p_h and p_a < 0.8


def test_control_beats_an_opposing_tag_prior():
    row = {"nearest_team": "H", "nearest_dist_m": 0.8, "ball_speed_ms": 2.0, "tag_team_prior": "A"}
    p_h, p_a, _ = held_probs(row)
    assert p_h > p_a


def test_params_change_the_rules():
    row = {"nearest_team": "H", "nearest_dist_m": 2.5, "ball_speed_ms": 2.0}
    assert held_probs(row)[0] < 0.6
    assert held_probs(row, control_dist_m=3.0)[0] > 0.8


def test_a_higher_temperature_flattens_the_probabilities():
    row = {"nearest_team": "H", "nearest_dist_m": 0.8, "ball_speed_ms": 2.0}
    assert held_probs(row, temperature=4.0)[0] < held_probs(row)[0]


# -- sustained control (official proxy and our analyst: a few seconds on the ball) ---------------


def test_a_brief_touch_gives_no_lean():
    out = predict(signals(*[OPEN_PLAY] * 20, *[A_CONTROL] * 10, *[OPEN_PLAY] * 20))
    touch = out.iloc[25]
    assert abs(touch["p_H"] - touch["p_A"]) < 1e-9


def test_a_brief_opponent_touch_does_not_flip_possession():
    assert decode([H_CONTROL] * 150 + [A_CONTROL] * 12 + [H_CONTROL] * 150) == ["H"] * 312


def test_a_held_ball_switches_possession_where_control_began():
    assert decode([H_CONTROL] * 150 + [A_CONTROL] * 75) == ["H"] * 150 + ["A"] * 75


def test_passes_between_teammates_keep_one_run():
    rows = [H_CONTROL] * 20 + [IN_FLIGHT] * 15 + [H_CONTROL] * 20
    out = predict(signals(*rows))
    assert (out["p_H"].iloc[[0, 19, 35, 54]] > 0.8).all()


def test_a_long_gap_splits_the_run():
    rows = [H_CONTROL] * 30 + [BALL_MISSING] * 100 + [H_CONTROL] * 30
    out = predict(signals(*rows))
    assert abs(out["p_H"].iloc[10] - out["p_A"].iloc[10]) < 1e-9


def test_min_control_frames_sets_what_counts_as_sustained():
    out = predict(
        signals(*[OPEN_PLAY] * 20, *[A_CONTROL] * 10, *[OPEN_PLAY] * 20), min_control_frames=5
    )
    assert out["p_A"].iloc[25] > 0.8


# -- contract ------------------------------------------------------------------------------------


def test_output_passes_the_emission_contract_and_keeps_every_frame():
    table = signals(
        OPEN_PLAY,
        BALL_MISSING,
        {"ball_in_bounds": False},
        {**BALL_MISSING, "team_speed_mean_m_s": 0.3},
        {"nearest_team": "A", "nearest_dist_m": 0.5, "ball_speed_ms": 1.0, "tag_team_prior": "H"},
    )
    out = predict(table)
    meta = Meta(len(table), 25, (105.3, 68.0), [(1, 0, len(table) - 1)], "FC Barcelona", "Athletic")
    validate_emissions(out, meta)
    assert out["frame"].tolist() == table["frame"].tolist()


def test_same_input_gives_the_same_output():
    table = signals(OPEN_PLAY, BALL_MISSING, {"ball_in_bounds": False})
    pd.testing.assert_frame_equal(predict(table), predict(table))


def test_fit_learns_nothing_and_returns_the_model():
    model = RulesV0()
    table = signals(OPEN_PLAY)
    before = model.predict(table)
    assert model.fit(table, pd.Series(["H"])) is model
    pd.testing.assert_frame_equal(model.predict(table), before)


def test_missing_signal_columns_are_named():
    table = signals(OPEN_PLAY).drop(columns=["team_speed_mean_m_s", "ball_in_bounds"])
    with pytest.raises(KeyError, match="ball_in_bounds.*team_speed_mean_m_s"):
        predict(table)


# -- params and registration ---------------------------------------------------------------------


def test_registered_and_built_from_empty_config_params():
    model = registry.EMISSIONS["rules_v0"]()
    assert model.params == RulesParams()


def test_a_misspelt_param_is_refused():
    with pytest.raises(ValueError, match="control_dist"):
        RulesV0(control_dist=1.5)


@pytest.mark.parametrize(
    "params",
    [
        {"control_dist_m": -1.0},
        {"stopped_weight": float("nan")},
        {"temperature": 0.0},
        {"prior_weight": float("inf")},
        {"min_control_frames": -1},
    ],
)
def test_invalid_params_are_refused(params):
    with pytest.raises(ValueError):
        RulesV0(**params)


# -- with the decoder ----------------------------------------------------------------------------


def test_with_viterbi_a_possession_a_stoppage_and_a_restart_become_three_spells():
    import rondo.decoder  # noqa: F401  (registers "viterbi")

    h_control = {"nearest_team": "H", "nearest_dist_m": 0.7, "ball_speed_ms": 2.0}
    a_control = {"nearest_team": "A", "nearest_dist_m": 0.7, "ball_speed_ms": 2.0}
    rows = (
        [h_control] * 150  # H plays it around
        + [BALL_MISSING] * 50  # ball lost by the tracking, players still running
        + [{**BALL_MISSING, "team_speed_mean_m_s": 0.3}] * 100  # play stopped
        + [a_control] * 150  # A restarts and keeps it
    )
    table = signals(*rows)
    emissions = predict(table)
    meta = Meta(len(table), 25, (105.3, 68.0), [(1, 0, len(table) - 1)], "FC Barcelona", "Athletic")
    no_tags = pd.DataFrame(
        {
            "start_frame": pd.Series([], dtype="int64"),
            "end_frame": pd.Series([], dtype="int64"),
            "code": pd.Series([], dtype="str"),
            "team": pd.Series([], dtype="str"),
            "type": pd.Series([], dtype="str"),
            "half": pd.array([], dtype="Int8"),
        }
    )
    labels = registry.DECODERS["viterbi"](emissions, no_tags, meta, {})["label"].tolist()
    assert labels == ["H"] * 200 + ["D"] * 100 + ["A"] * 150


def test_the_default_pipeline_on_task_11_signals_recovers_a_turnover_and_a_restart():
    # Guards the link between rules_v0 and the real ball/player signals (column names, dtypes).
    # The full scenario suite is Task 13's.
    import tomllib
    from pathlib import Path

    from rondo.eval import score
    from rondo.pipeline import run
    from tests.synthetic import Step, make_match

    config_path = Path(__file__).resolve().parents[2] / "configs" / "default.toml"
    config = tomllib.loads(config_path.read_text())
    steps = [
        Step("dribble", "H", 4),
        Step("interception", "A", 3),
        Step("dribble", "A", 3),
        Step("out", "A", 6),
        Step("restart", "H", 3),
        Step("dribble", "H", 4),
    ]
    match, truth = make_match(steps, n_frames=1000)
    pred = run(match, config).set_index("frame")["label"]
    assert (pred == truth).all()
    assert score(pred, truth)["spell_f1_mean"] == 1.0
