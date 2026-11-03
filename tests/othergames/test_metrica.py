"""Another match for validation: Metrica's public sample game 3, and an answer key from its events."""

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rondo.contracts import validate_frames, validate_players, validate_tags
from rondo.othergames.metrica import (
    load_sample_game,
    nearest_event_kind,
    possession_from_events,
    read_events,
)
from rondo.pipeline import run
from tests.othergames.fixtures import event, write_game

DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "default.toml"


@pytest.fixture
def game(tmp_path):
    write_game(tmp_path)
    return load_sample_game(tmp_path)


def runs(labels: pd.Series) -> list[tuple[str, int, int]]:
    """``(label, first row, last row)`` for each run of equal labels."""
    out = []
    for row, value in enumerate(labels.tolist()):
        if out and out[-1][0] == value:
            out[-1] = (value, out[-1][1], row)
        else:
            out.append((value, row, row))
    return out


# -- the match ------------------------------------------------------------------------------------


def test_the_match_follows_the_contracts(game):
    meta = game.meta
    assert meta.n_frames == 200 and meta.fps == 25 and meta.pitch == (105.0, 68.0)
    assert meta.half_blocks == [(1, 0, 99), (2, 100, 199)]
    assert (meta.home_name, meta.away_name) == ("Team A", "Team B")
    validate_frames(game.frames, meta)
    validate_players(game.players, meta)
    validate_tags(game.tags, meta)
    assert game.tags.empty  # the sample games have no Smart Tagging


def test_a_substitute_takes_his_own_slot_and_team(game):
    players = game.players
    before = players[players["frame"] == 49]
    after = players[players["frame"] == 60]
    assert len(before) == len(after) == 4
    assert set(before["slot"]) != set(after["slot"])  # P2 out, P5 in: a new person, a new slot
    assert after.groupby("team").size().to_dict() == {"A": 2, "H": 2}


def test_speeds_are_computed_from_positions_in_metres_per_second(game):
    walker = game.players[(game.players["frame"] == 100) & (game.players["x"] > 0.25)]
    walker = walker[walker["team"] == "H"]
    assert walker["speed"].iloc[0] == pytest.approx(0.002 * 105 * 25, rel=1e-3)
    still = game.players[(game.players["frame"] == 100) & (game.players["team"] == "A")]
    assert np.allclose(still["speed"], 0.0)


def test_the_ball_is_missing_where_the_tracking_says_nan(game):
    ball = game.frames.set_index("frame")["ball_x"]
    assert ball.loc[9:11].isna().all() and ball.loc[[8, 12]].notna().all()


# -- the answer key -------------------------------------------------------------------------------


def test_the_answer_key_follows_our_possession_rules(tmp_path, game):
    folder = write_game(tmp_path / "again")
    truth = possession_from_events(
        read_events(folder / "Sample_Game_3_events.json", "TA"), game.meta
    )
    assert runs(truth) == [
        ("X", 0, 3),  # before the kick-off
        ("H", 4, 43),  # a challenge changes nothing, and B's 15 frames with one pass are no control
        ("D", 44, 58),  # from where the ball left the pitch, until the throw-in
        ("H", 59, 78),
        ("D", 79, 88),  # from the foul until the free kick
        ("H", 89, 94),
        ("X", 95, 108),  # after the last event of the half, before the second kick-off
        ("A", 109, 123),
        ("D", 124, 148),  # from the goal until the kick-off
        ("H", 149, 179),
        ("X", 180, 199),
    ]


def test_the_recovery_spell_starts_where_control_began(tmp_path, game):
    events = [
        event("SET PIECE", "A", 5, subtypes=["KICK OFF"]),
        event("PASS", "A", 5, 25),
        event("RECOVERY", "B", 30, subtypes=["INTERCEPTION"]),
        event("CARRY", "B", 30, 95),
    ]
    folder = write_game(tmp_path / "again", events)
    truth = possession_from_events(
        read_events(folder / "Sample_Game_3_events.json", "TA"), game.meta
    )
    assert runs(truth)[:3] == [("X", 0, 3), ("H", 4, 28), ("A", 29, 94)]


def test_dead_without_a_set_piece_ends_at_the_next_team_event(tmp_path, game):
    events = [
        event("SET PIECE", "A", 5, subtypes=["KICK OFF"]),
        event("BALL OUT", "A", 10, 20),
        event("PASS", "B", 40, 50),  # play resumed with no set piece logged
    ]
    folder = write_game(tmp_path / "again", events)
    truth = possession_from_events(
        read_events(folder / "Sample_Game_3_events.json", "TA"), game.meta
    )
    assert runs(truth)[:4] == [("X", 0, 3), ("H", 4, 18), ("D", 19, 38), ("A", 39, 49)]


# -- end to end -----------------------------------------------------------------------------------


def test_the_default_pipeline_runs_on_the_sample_game(game):
    config = tomllib.loads(DEFAULT.read_text(encoding="utf-8"))
    labels = run(game, config)
    assert len(labels) == 200 and set(labels["label"]) <= {"H", "A", "D"}


def test_cli_scores_a_config_on_the_sample_game(capsys, tmp_path):
    from rondo.cli import main

    write_game(tmp_path)
    assert main(["othergame", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "sample3" in out and "spellF1" in out
    assert len([line for line in out.splitlines() if line.startswith("sample3 ")]) == 1


def test_cli_names_a_missing_sample_checkout(capsys, tmp_path):
    from rondo.cli import main

    assert main(["othergame", "--root", str(tmp_path / "nowhere")]) == 1
    assert "sample-data" in capsys.readouterr().out


def test_a_brief_spell_without_control_stays_with_the_team_that_had_the_ball(tmp_path, game):
    events = [
        event("SET PIECE", "A", 5, subtypes=["KICK OFF"]),
        event("PASS", "A", 5, 40),
        event("RECOVERY", "B", 41),  # B wins it for under 2 s, no passes...
        event("BALL LOST", "B", 55),
        event("RECOVERY", "A", 60),  # ...and A has it back
        event("CARRY", "A", 60, 99),
    ]
    folder = write_game(tmp_path / "again", events)
    truth = possession_from_events(
        read_events(folder / "Sample_Game_3_events.json", "TA"), game.meta
    )
    assert runs(truth) == [("X", 0, 3), ("H", 4, 98), ("X", 99, 199)]


def test_a_brief_spell_with_two_passes_is_control(tmp_path, game):
    events = [
        event("SET PIECE", "A", 5, subtypes=["KICK OFF"]),
        event("PASS", "A", 5, 40),
        event("RECOVERY", "B", 41),
        event("PASS", "B", 42, 46),
        event("PASS", "B", 47, 52),  # two quick passes: control, even under 2 s
        event("RECOVERY", "A", 60),
        event("CARRY", "A", 60, 99),
    ]
    folder = write_game(tmp_path / "again", events)
    truth = possession_from_events(
        read_events(folder / "Sample_Game_3_events.json", "TA"), game.meta
    )
    assert [label for label, _, _ in runs(truth)][:3] == ["X", "H", "A"]


# -- games 1 and 2 (CSV) --------------------------------------------------------------------------


def test_a_csv_game_follows_the_contracts(tmp_path):
    from tests.othergames.fixtures import write_csv_game

    write_csv_game(tmp_path, game=1)
    match = load_sample_game(tmp_path, game=1)
    meta = match.meta
    assert meta.n_frames == 200 and meta.half_blocks == [(1, 0, 99), (2, 100, 199)]
    assert meta.pitch == (105.0, 68.0)
    validate_frames(match.frames, meta)
    validate_players(match.players, meta)
    assert match.players.groupby("team")["slot"].nunique().to_dict() == {"A": 2, "H": 2}
    assert match.frames.set_index("frame")["ball_x"].loc[9:11].isna().all()
    walker = match.players[(match.players["frame"] == 100) & (match.players["x"] > 0.25)]
    assert walker["speed"].iloc[0] == pytest.approx(0.002 * 105 * 25, rel=1e-3)


def test_a_csv_game_gets_the_same_answer_key_as_the_json_one(tmp_path, game):
    from rondo.othergames.metrica import answer_key
    from tests.othergames.fixtures import write_csv_game

    write_csv_game(tmp_path / "csv", game=2)
    write_game(tmp_path / "json")
    assert answer_key(tmp_path / "csv", 2, game.meta).equals(
        answer_key(tmp_path / "json", 3, game.meta)
    )


def test_cli_scores_every_game_present_and_all_together(capsys, tmp_path):
    from rondo.cli import main
    from tests.othergames.fixtures import write_csv_game

    write_csv_game(tmp_path, game=1)
    write_game(tmp_path)
    assert main(["othergame", "--root", str(tmp_path)]) == 0
    rows = [line.split()[0] for line in capsys.readouterr().out.splitlines()[1:]]
    assert rows == ["sample1", "sample3", "all"]


# -- error breakdown by event kind -----------------------------------------------------------------


def test_nearest_event_kind_reads_off_the_event_covering_the_frame(tmp_path):
    from rondo.othergames.metrica import load_events

    write_game(tmp_path)
    events = load_events(tmp_path, 3)
    assert nearest_event_kind(events, 20) == "CARRY"  # inside the CARRY event's own span


def test_nearest_event_kind_falls_back_to_the_closest_one_nearby(tmp_path):
    from rondo.othergames.metrica import load_events

    write_game(tmp_path)
    events = load_events(tmp_path, 3)
    assert nearest_event_kind(events, 46) == "BALL OUT"  # a few rows after it ends, nothing between


def test_nearest_event_kind_is_none_far_from_every_event(tmp_path):
    from rondo.othergames.metrica import load_events

    write_game(tmp_path)
    events = load_events(tmp_path, 3)
    assert nearest_event_kind(events, 199, max_gap=5) == "none"  # 19 rows past the last event


def test_nearest_event_kind_of_no_events_at_all():
    assert nearest_event_kind(pd.DataFrame(columns=["start", "end", "kind"]), 10) == "none"


def test_load_events_agrees_with_answer_key_on_which_events_it_used(tmp_path, game):
    from rondo.othergames.metrica import answer_key, load_events

    write_game(tmp_path / "again")
    events = load_events(tmp_path / "again", 3)
    truth = possession_from_events(events, game.meta)
    assert truth.equals(answer_key(tmp_path / "again", 3, game.meta))


def test_cli_errors_breaks_missed_and_spurious_down_by_event_kind(capsys, tmp_path):
    from rondo.cli import main

    write_game(tmp_path)
    assert main(["othergame", "--root", str(tmp_path), "--errors"]) == 0
    out = capsys.readouterr().out
    assert "sample3 errors (" in out
    # one indented "status  event kind  count" line per (status, event kind) pair found; event
    # kind can itself contain a space ("BALL OUT", "FAULT RECEIVED"), so just check the ends
    lines = [line for line in out.splitlines() if line.startswith("  ")]
    assert lines and all(line.split()[0] in ("missed", "spurious", "late") for line in lines)
    assert lines and all(line.split()[-1].isdigit() for line in lines)


def test_cli_errors_combines_every_game_once_more_than_one_is_scored(capsys, tmp_path):
    from rondo.cli import main
    from tests.othergames.fixtures import write_csv_game

    write_csv_game(tmp_path, game=1)
    write_game(tmp_path)
    assert main(["othergame", "--root", str(tmp_path), "--errors"]) == 0
    out = capsys.readouterr().out
    assert "sample1 errors (" in out and "sample3 errors (" in out and "all errors (" in out


def test_cli_without_errors_prints_no_breakdown(capsys, tmp_path):
    from rondo.cli import main

    write_game(tmp_path)
    assert main(["othergame", "--root", str(tmp_path)]) == 0
    assert "errors (" not in capsys.readouterr().out


# -- quiet-ball check ----------------------------------------------------------------------------


def test_quiet_ball_check_splits_quiet_and_active_spurious_switches(tmp_path):
    import rondo.signals.ball  # noqa: F401 - registers the signals _quiet_ball_check needs
    from rondo.cli import _quiet_ball_check
    from tests.synthetic import Step, make_match

    write_game(tmp_path)  # its events are all within frames 1-200, nothing near 1000+
    # n_frames == start + the steps' own length: no padding after the match, so the baseline
    # below is exact and not a mix of real coverage and surrounding "X" padding.
    match, truth = make_match(
        [Step("restart", "H", 6), Step("dribble", "H", 6)], n_frames=1300, start=1000
    )
    pred = truth.copy()
    pred.loc[1060:1089] = "A"  # a false switch while the ball sits held at the restart
    pred.loc[1200:1229] = "A"  # a false switch while the ball is genuinely moving

    stats = _quiet_ball_check(tmp_path, 3, match, pred, truth)
    assert stats["n"] == 4
    assert stats["quiet"] == 2
    assert stats["active"] == 2
    assert stats["untracked"] == 0
    # The 1000 frames of "X" padding before the match starts, plus the signal's own warm-up lag
    # once real ball tracking begins: a fixed, reproducible count, not a guess.
    assert stats["baseline_frames"] == 1300
    assert stats["baseline_untracked_frames"] == 1003


def test_quiet_ball_check_is_a_noop_with_no_spurious_switches(tmp_path):
    """The empty-DataFrame edge case: spurious[[]] used to drop every column, not every row."""
    import rondo.signals.ball  # noqa: F401
    from rondo.cli import _quiet_ball_check
    from tests.synthetic import Step, make_match

    write_game(tmp_path)
    match, truth = make_match([Step("dribble", "H", 4)], n_frames=1500, start=1000)
    stats = _quiet_ball_check(tmp_path, 3, match, truth, truth)  # pred == truth: nothing spurious
    assert stats["n"] == 0
    assert stats["quiet"] == 0
    assert stats["active"] == 0
    assert stats["untracked"] == 0
    # Computed even with no spurious switches, so a game with none still feeds the combined total.
    assert stats["baseline_frames"] == 1500
    assert stats["baseline_untracked_frames"] > 0


def test_cli_quiet_ball_prints_a_report(capsys, tmp_path):
    from rondo.cli import main

    write_game(tmp_path)
    assert main(["othergame", "--root", str(tmp_path), "--quiet-ball"]) == 0
    assert "sample3 quiet-ball check (" in capsys.readouterr().out


def test_cli_without_quiet_ball_prints_no_report(capsys, tmp_path):
    from rondo.cli import main

    write_game(tmp_path)
    assert main(["othergame", "--root", str(tmp_path)]) == 0
    assert "quiet-ball check" not in capsys.readouterr().out
