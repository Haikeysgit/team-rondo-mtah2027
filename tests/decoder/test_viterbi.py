"""Viterbi decoder: one steady label per frame, switching only on sustained evidence."""

import numpy as np
import pandas as pd
import pytest

import rondo.decoder  # noqa: F401  (registers "viterbi")
from rondo import registry
from rondo.contracts import Meta, validate_labels
from rondo.decoder.viterbi import ViterbiParams

H_EVIDENCE = (0.8, 0.1, 0.1)
A_EVIDENCE = (0.1, 0.8, 0.1)
DEAD_EVIDENCE = (0.05, 0.05, 0.9)
NO_BALL = (0.45, 0.45, 0.1)  # ball missing, players moving: either team, play alive


def decode(emissions, tags=None, meta=None, params=None):
    meta = meta or one_half(len(emissions))
    tags = no_tags() if tags is None else tags
    return registry.DECODERS["viterbi"](emissions, tags, meta, params or {})


def one_half(n: int) -> Meta:
    return Meta(
        n_frames=n,
        fps=25,
        pitch=(105.3, 68.0),
        half_blocks=[(1, 0, n - 1)],
        home_name="FC Barcelona",
        away_name="Athletic Club",
    )


def emissions(*segments) -> pd.DataFrame:
    """``(frames, (p_H, p_A, p_D))`` segments laid end to end from frame 0."""
    rows = [probs for count, probs in segments for _ in range(count)]
    p = np.array(rows, dtype="float64")
    return pd.DataFrame(
        {"frame": np.arange(len(p), dtype="int64"), "p_H": p[:, 0], "p_A": p[:, 1], "p_D": p[:, 2]}
    )


def tags(*rows) -> pd.DataFrame:
    """``(start_frame, end_frame, code, team)`` rows; ``team=None`` is a tag marked N/A."""
    return pd.DataFrame(
        {
            "start_frame": np.array([r[0] for r in rows], dtype="int64"),
            "end_frame": np.array([r[1] for r in rows], dtype="int64"),
            "code": pd.Series([r[2] for r in rows], dtype="str"),
            "team": pd.Series([np.nan if r[3] is None else r[3] for r in rows], dtype="str"),
            "type": pd.Series(["Throw-in"] * len(rows), dtype="str"),
            "half": pd.array([1] * len(rows), dtype="Int8"),
        }
    )


def no_tags() -> pd.DataFrame:
    return tags()


def runs(labels: pd.DataFrame) -> list[tuple[str, int]]:
    """``(label, length)`` for each run of equal labels, in frame order."""
    values = labels["label"].tolist()
    out: list[tuple[str, int]] = []
    for value in values:
        if out and out[-1][0] == value:
            out[-1] = (value, out[-1][1] + 1)
        else:
            out.append((value, 1))
    return out


# -- holding and switching -----------------------------------------------------------------------


def test_a_three_frame_touch_by_the_other_team_does_not_flip_possession():
    out = decode(emissions((100, H_EVIDENCE), (3, A_EVIDENCE), (97, H_EVIDENCE)))
    assert runs(out) == [("H", 200)]


def test_sustained_control_by_the_other_team_switches_once_where_it_starts():
    out = decode(emissions((200, H_EVIDENCE), (75, A_EVIDENCE)))
    assert runs(out) == [("H", 200), ("A", 75)]


def test_ten_seconds_without_the_ball_keeps_the_possession():
    out = decode(emissions((100, H_EVIDENCE), (250, NO_BALL), (100, H_EVIDENCE)))
    assert runs(out) == [("H", 450)]


def test_a_short_false_stop_does_not_split_the_possession():
    out = decode(emissions((200, H_EVIDENCE), (15, DEAD_EVIDENCE), (185, H_EVIDENCE)))
    assert runs(out) == [("H", 400)]


def test_a_real_stoppage_goes_dead_once_and_comes_back_once():
    out = decode(emissions((150, H_EVIDENCE), (100, DEAD_EVIDENCE), (150, H_EVIDENCE)))
    assert runs(out) == [("H", 150), ("D", 100), ("H", 150)]


# -- restart anchors -----------------------------------------------------------------------------

RESTART = emissions((900, H_EVIDENCE), (170, DEAD_EVIDENCE), (230, NO_BALL))  # play resumes at 1070


def first_alive_after_stoppage(out: pd.DataFrame) -> tuple[int, str]:
    after = out[out["frame"] >= 900]
    alive = after[after["label"] != "D"].iloc[0]
    return int(alive["frame"]), alive["label"]


def test_without_a_tag_the_team_after_a_stoppage_is_a_coin_toss_settled_by_label_order():
    frame, label = first_alive_after_stoppage(decode(RESTART))
    assert (frame, label) == (1070, "H")


def test_a_restart_tag_gives_the_restart_to_its_team_where_play_resumes():
    out = decode(RESTART, tags((1000, 1125, "SET PIECES", "A")))
    frame, label = first_alive_after_stoppage(out)
    assert abs(frame - 1070) <= 3
    assert label == "A"
    assert set(out.loc[out["frame"] >= frame, "label"]) == {"A"}


def test_a_restart_tag_without_a_team_changes_nothing():
    untagged = decode(RESTART)
    tagged = decode(RESTART, tags((1000, 1125, "SET PIECES", None)))
    pd.testing.assert_frame_equal(tagged, untagged)


def test_tags_that_are_not_restarts_do_not_anchor():
    untagged = decode(RESTART)
    tagged = decode(RESTART, tags((1000, 1125, "BUILD UP", "A")))
    pd.testing.assert_frame_equal(tagged, untagged)


def test_a_restart_tag_outside_the_stoppage_does_not_force_a_switch():
    out = decode(emissions((400, H_EVIDENCE)), tags((100, 225, "SET PIECES", "A")))
    assert runs(out) == [("H", 400)]


# -- halves, gaps and output ---------------------------------------------------------------------


def test_each_half_is_labelled_on_its_own_frames_only():
    meta = Meta(
        n_frames=300,
        fps=25,
        pitch=(105.3, 68.0),
        half_blocks=[(1, 0, 99), (2, 150, 249)],
        home_name="FC Barcelona",
        away_name="Athletic Club",
    )
    em = emissions((150, H_EVIDENCE), (150, A_EVIDENCE))
    out = decode(em, meta=meta)
    assert out["frame"].tolist() == list(range(100)) + list(range(150, 250))
    assert runs(out) == [("H", 100), ("A", 100)]


def test_frames_without_an_emission_row_keep_the_current_label():
    em = emissions((300, H_EVIDENCE))
    em = em[(em["frame"] < 100) | (em["frame"] >= 200)].reset_index(drop=True)
    out = decode(em, meta=one_half(300))
    assert out["frame"].tolist() == list(range(300))
    assert runs(out) == [("H", 300)]


def test_output_is_a_valid_decoder_label_table():
    out = decode(emissions((150, H_EVIDENCE), (100, DEAD_EVIDENCE), (150, A_EVIDENCE)))
    validate_labels(out, one_half(400), allow_x=False)


def test_same_input_gives_the_same_output():
    em = emissions((120, H_EVIDENCE), (40, NO_BALL), (90, A_EVIDENCE), (60, DEAD_EVIDENCE))
    pd.testing.assert_frame_equal(decode(em), decode(em))


# -- parameters ----------------------------------------------------------------------------------


def test_params_from_the_config_change_the_decoding():
    em = emissions((200, H_EVIDENCE), (75, A_EVIDENCE))
    # Raise the dead-ball costs too: otherwise H -> D -> A (one dead frame) is the cheaper route.
    costly = {"switch_cost": 1000.0, "dead_entry_cost": 1000.0, "dead_exit_cost": 1000.0}
    assert runs(decode(em, params=costly)) == [("H", 275)]


def test_the_largest_allowed_switch_cost_never_routes_a_team_change_through_one_dead_frame():
    # Worst case: one frame of certain D evidence exactly at the change, inside a restart window.
    limit = ViterbiParams().max_switch_cost()
    assert limit == pytest.approx(25.0 + 25.0 - 6.0 + np.log(1e-9))
    em = emissions((100, (1.0, 0.0, 0.0)), (1, (0.0, 0.0, 1.0)), (100, (0.0, 1.0, 0.0)))
    restart = tags((50, 150, "SET PIECES", "A"))
    out = decode(em, restart, params={"switch_cost": limit - 1e-6})
    assert [label for label, _ in runs(out)] == ["H", "A"]


def test_a_switch_cost_above_the_limit_is_refused():
    # Above the limit the same input gives H, D x1, A: a phantom stoppage at the team change.
    with pytest.raises(ValueError, match="switch_cost"):
        decode(emissions((10, H_EVIDENCE)), params={"switch_cost": 24.0})
    with pytest.raises(ValueError, match="switch_cost"):
        decode(emissions((10, H_EVIDENCE)), params={"switch_cost": 1000.0, "dead_entry_cost": 4.0})


@pytest.mark.parametrize(
    "params",
    [
        {"switch_cost": -1.0},
        {"dead_entry_cost": -0.5},
        {"dead_exit_cost": -2.0, "dead_entry_cost": 60.0},
        {"restart_bonus": -1.0},
        {"restart_bonus": 25.0},
        {"restart_bonus": 30.0, "dead_exit_cost": 30.0, "dead_entry_cost": 60.0},
        {"switch_cost": float("nan")},
        {"dead_entry_cost": float("inf")},
    ],
)
def test_invalid_costs_are_refused(params):
    with pytest.raises(ValueError):
        ViterbiParams.from_dict(params)


def test_the_default_and_config_params_are_valid():
    import tomllib
    from pathlib import Path

    config = Path(__file__).resolve().parents[2] / "configs" / "default.toml"
    with open(config, "rb") as handle:
        ViterbiParams.from_dict(tomllib.load(handle)["decoder_params"])
    ViterbiParams()


# -- edge cases ----------------------------------------------------------------------------------


def test_exact_zero_and_one_probabilities_decode_without_nan():
    em = emissions((100, (1.0, 0.0, 0.0)), (100, (0.0, 0.0, 1.0)), (100, (0.0, 1.0, 0.0)))
    out = decode(em)
    validate_labels(out, one_half(300), allow_x=False)
    assert runs(out) == [("H", 100), ("D", 100), ("A", 100)]


def test_a_one_frame_half_gets_its_most_likely_label():
    meta = Meta(10, 25, (105.3, 68.0), [(1, 0, 4), (2, 7, 7)], "FC Barcelona", "Athletic Club")
    em = emissions((7, H_EVIDENCE), (3, A_EVIDENCE))
    out = decode(em, meta=meta)
    assert out["frame"].tolist() == [0, 1, 2, 3, 4, 7]
    assert out["label"].tolist() == ["H"] * 5 + ["A"]


def test_an_h_restart_tag_beats_mild_a_evidence_after_a_stoppage():
    mild_a = (0.4, 0.5, 0.1)  # A ahead, but by less than the anchor is worth
    em = emissions((900, H_EVIDENCE), (170, DEAD_EVIDENCE), (230, mild_a))
    assert first_alive_after_stoppage(decode(em))[1] == "A"
    out = decode(em, tags((1000, 1125, "SET PIECES", "H")))
    frame, label = first_alive_after_stoppage(out)
    assert abs(frame - 1070) <= 3 and label == "H"


def test_ambiguous_evidence_at_the_start_of_half_two_does_not_inherit_half_one():
    meta = Meta(
        400, 25, (105.3, 68.0), [(1, 0, 199), (2, 200, 399)], "FC Barcelona", "Athletic Club"
    )
    # Half 1 ends in a long A possession; half 2 opens with 60 ambiguous frames, then H.
    em = emissions((200, A_EVIDENCE), (60, NO_BALL), (140, H_EVIDENCE))
    out = decode(em, meta=meta)
    second = out[out["frame"] >= 200]
    assert set(second["label"]) == {"H"}


def test_a_misspelt_param_is_refused_instead_of_ignored():
    with pytest.raises(ValueError, match="switch_cots"):
        decode(emissions((10, H_EVIDENCE)), params={"switch_cots": 8.0})
