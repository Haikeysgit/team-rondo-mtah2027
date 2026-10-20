"""Smart Tagging check on a match: counts and agreement rates only, never rows."""

from dataclasses import replace

import pandas as pd
import pytest

from rondo.eval.tagcheck import tag_report
from tests.synthetic import Step, make_match

STEPS = [
    Step("dribble", "H", 4),
    Step("out", "H", 6),
    Step("restart", "A", 3),
    Step("dribble", "A", 4),
    Step("out", "A", 6),
    Step("restart", "H", 3),
    Step("dribble", "H", 4),
]


@pytest.fixture(scope="module")
def match_and_truth():
    return make_match(STEPS, n_frames=1500)


def test_counts_restart_windows_and_their_teams(match_and_truth):
    match, truth = match_and_truth
    report = tag_report(match, truth)
    assert report["restart_windows"] == 2
    assert report["restart_windows_with_team_pct"] == 100.0


def test_restart_windows_see_the_ball_out(match_and_truth):
    match, truth = match_and_truth
    assert tag_report(match, truth)["restart_windows_ball_out_pct"] == 100.0


def test_restart_team_matches_who_has_the_ball_after_the_window(match_and_truth):
    match, truth = match_and_truth
    report = tag_report(match, truth)
    assert report["restart_team_vs_truth_n"] == 2
    assert report["restart_team_vs_truth_pct"] == 100.0


def test_a_wrong_restart_team_lowers_the_agreement(match_and_truth):
    match, truth = match_and_truth
    flipped = match.tags.copy()
    flipped.loc[flipped.index[0], "team"] = "H"  # the first restart is A's
    report = tag_report(replace(match, tags=flipped), truth)
    assert report["restart_team_vs_truth_pct"] == 50.0


def test_team_prior_coverage_and_agreement(match_and_truth):
    match, truth = match_and_truth
    graded = truth[truth != "X"]
    first = int(graded.index[0])
    h_spell = graded[graded == "H"].index
    build_up = pd.DataFrame(
        {
            "start_frame": pd.Series([first], dtype="int64"),
            "end_frame": pd.Series([int(h_spell[50]) + 1], dtype="int64"),
            "code": pd.Series(["BUILD UP"], dtype="str"),
            "team": pd.Series(["H"], dtype="str"),
            "type": pd.Series([pd.NA], dtype="str"),
            "half": pd.array([1], dtype="Int8"),
        }
    )
    tags = pd.concat([match.tags, build_up], ignore_index=True)
    report = tag_report(replace(match, tags=tags), truth)
    assert 0 < report["prior_coverage_graded_pct"] < 100
    assert report["prior_vs_truth_pct"] == 100.0


def test_without_truth_only_the_tag_counts_are_reported(match_and_truth):
    match, _ = match_and_truth
    report = tag_report(match, None)
    assert report["restart_windows"] == 2
    assert report["restart_team_vs_truth_pct"] is None
    assert report["prior_vs_truth_pct"] is None
