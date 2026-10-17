"""Viterbi decoder: one steady label per frame, switching only on sustained evidence."""

import numpy as np
import pandas as pd
import pytest

import rondo.decoder  # noqa: F401  (registers "viterbi")
from rondo import registry
from rondo.contracts import Meta, validate_labels

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


def test_a_team_change_never_costs_more_than_a_one_frame_stop():
    em = emissions((200, H_EVIDENCE), (75, A_EVIDENCE))
    out = decode(em, params={"switch_cost": 1000.0, "dead_entry_cost": 4.0, "dead_exit_cost": 4.0})
    # Known limit for tuning: keep switch_cost below dead_entry_cost + dead_exit_cost.
    assert [(label, length) for label, length in runs(out) if label == "D"] == [("D", 1)]
    assert [label for label, _ in runs(out)] == ["H", "D", "A"]


def test_a_misspelt_param_is_refused_instead_of_ignored():
    with pytest.raises(ValueError, match="switch_cots"):
        decode(emissions((10, H_EVIDENCE)), params={"switch_cots": 8.0})
