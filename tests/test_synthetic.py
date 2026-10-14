import numpy as np
import pandas as pd
import pytest

from rondo.contracts import (
    validate_frames,
    validate_labels,
    validate_players,
    validate_tags,
)
from tests.synthetic import Step, full_length, make_match

W, H = 105.3, 68.0
EVENTS = ["dribble", "pass", "interception", "deflection", "aerial", "out", "restart", "missing"]


def runs(labels):
    seq = labels[labels != "X"].tolist()
    return [k for i, k in enumerate(seq) if i == 0 or k != seq[i - 1]]


def validate_all(m, truth):
    validate_frames(m.frames, m.meta)
    validate_players(m.players, m.meta)
    validate_tags(m.tags, m.meta)
    validate_labels(truth.rename("label").reset_index(), m.meta, allow_x=True)


def player_arrays(m):
    """(frame -> (22, 2) metres, (22,) speed) for the half block."""
    p = m.players.sort_values(["frame", "slot"])
    frames = p["frame"].to_numpy().reshape(-1, 22)[:, 0]
    xy = np.stack([p["x"].to_numpy() * W, p["y"].to_numpy() * H], axis=1).reshape(-1, 22, 2)
    return frames, xy, p["speed"].to_numpy().reshape(-1, 22)


def ball_m(m):
    return np.stack([m.frames["ball_x"].to_numpy() * W, m.frames["ball_y"].to_numpy() * H], axis=1)


def test_synthetic_match_is_valid_and_labels_follow_rules():
    m, truth = make_match(
        [
            Step("dribble", "H", 4),
            Step("pass", "H", 2),
            Step("interception", "A", 1),
            Step("dribble", "A", 3),
            Step("out", "A", 2),
            Step("restart", "H", 2),
        ]
    )
    validate_frames(m.frames, m.meta)
    validate_players(m.players, m.meta)
    validate_tags(m.tags, m.meta)
    validate_labels(truth.rename("label").reset_index(), m.meta, allow_x=True)
    seq = truth[truth != "X"]
    runs = [k for i, k in enumerate(seq) if i == 0 or k != seq.iloc[i - 1]]
    assert runs == ["H", "A", "D", "H"]


def test_synthetic_is_deterministic():
    a = make_match([Step("dribble", "H", 2)], seed=3)[0].players
    b = make_match([Step("dribble", "H", 2)], seed=3)[0].players
    assert a.equals(b)


def test_all_tables_equal_for_same_seed_and_truth_independent_of_seed():
    steps = [Step("dribble", "H", 2), Step("pass", "H", 2), Step("interception", "A", 2)]
    m1, t1 = make_match(steps, seed=1)
    m2, t2 = make_match(steps, seed=1)
    m3, t3 = make_match(steps, seed=2)
    assert m1.frames.equals(m2.frames) and m1.tags.equals(m2.tags) and t1.equals(t2)
    assert t1.equals(t3)
    assert not m1.players.equals(m3.players)


def test_block_layout_and_meta():
    m, truth = make_match([Step("dribble", "H", 2), Step("pass", "H", 1)], n_frames=500, start=40)
    assert m.meta.n_frames == 500 and m.meta.fps == 25 and m.meta.pitch == (105.3, 68.0)
    assert m.meta.half_blocks == [(1, 40, 40 + 75 - 1)]
    assert (m.meta.home_name, m.meta.away_name) == ("FC Barcelona", "Athletic Club")
    assert len(truth) == 500 and truth.name == "label"
    assert (truth.iloc[:40] == "X").all() and (truth.iloc[115:] == "X").all()
    assert (truth.iloc[40:115] != "X").all()
    f = m.frames
    inside = f["frame"].between(40, 114)
    assert (f["graded"] == inside).all()
    assert (f.loc[inside, "half"] == 1).all() and f.loc[~inside, "half"].isna().all()
    assert set(m.players["frame"]) == set(range(40, 115))
    assert len(m.players) == 75 * 22
    assert (m.players.groupby("frame")["team"].apply(lambda s: (s == "H").sum()) == 11).all()
    slot_team = m.players.drop_duplicates("slot").set_index("slot")["team"]
    assert (slot_team[slot_team.index < 11] == "H").all()
    assert (slot_team[slot_team.index >= 11] == "A").all()


def test_steps_round_seconds_to_frames():
    m, _ = make_match([Step("dribble", "H", 0.5), Step("dribble", "H", 0.3)], start=0)
    assert m.meta.half_blocks == [(1, 0, 12 + 8 - 1)]  # round(12.5) = 12, round(7.5) = 8


def test_steps_that_do_not_fit_raise():
    with pytest.raises(ValueError, match="fit"):
        make_match([Step("dribble", "H", 200)], n_frames=3000)
    with pytest.raises(ValueError, match="at least one"):
        make_match([])


def test_interception_cannot_be_first_or_same_team():
    with pytest.raises(ValueError, match="interception"):
        make_match([Step("interception", "A", 2)])
    with pytest.raises(ValueError, match="interception"):
        make_match([Step("dribble", "H", 1), Step("interception", "H", 2)])


@pytest.mark.parametrize("event", EVENTS)
def test_each_event_truth_sequence(event):
    team = "A" if event == "interception" else "H"
    m, truth = make_match([Step("dribble", "H", 2), Step(event, team, 2)], start=100)
    validate_all(m, truth)
    block = truth.iloc[100:200]
    assert (block.iloc[:50] == "H").all()
    second = block.iloc[50:]
    if event == "out":
        assert (second.iloc[:12] == "H").all() and (second.iloc[12:] == "D").all()
    elif event == "interception":
        assert second.iloc[0] == "H" and second.iloc[-1] == "A"
        switch = (second == "A").idxmax()
        assert (second.loc[switch:] == "A").all() and (second.loc[: switch - 1] == "H").all()
    else:
        assert (second == "H").all()


@pytest.mark.parametrize("event", ["dribble", "restart"])
def test_first_step_team_sets_initial_possession(event):
    _, truth = make_match([Step(event, "A", 1)])
    assert runs(truth) == ["A"]


@pytest.mark.parametrize("event", ["pass", "deflection", "aerial", "missing", "out"])
def test_first_step_other_events_start_with_team(event):
    _, truth = make_match([Step(event, "A", 2)])
    assert runs(truth) == (["A", "D"] if event == "out" else ["A"])


def test_possession_after_restart_and_loose_state():
    _, truth = make_match(
        [
            Step("dribble", "A", 1),
            Step("out", "A", 1),
            Step("restart", "H", 1),
            Step("pass", "H", 1),
        ]
    )
    assert runs(truth) == ["A", "D", "H"]


def test_missing_gives_nan_ball_and_keeps_label():
    m, truth = make_match(
        [Step("dribble", "A", 1), Step("missing", "H", 2), Step("dribble", "A", 1)], start=50
    )
    validate_all(m, truth)
    ball = m.frames[["ball_x", "ball_y"]]
    assert ball.iloc[75:125].isna().all().all()
    assert ball.iloc[50:75].notna().all().all() and ball.iloc[125:150].notna().all().all()
    assert ball.iloc[:50].isna().all().all()  # nothing is tracked outside the block
    assert (truth.iloc[50:150] == "A").all()
    _, _, speed = player_arrays(m)
    assert speed[25:75].mean() > 0.3  # players keep moving while the ball is missing


def test_missing_after_out_keeps_dead_label():
    _, truth = make_match([Step("dribble", "H", 1), Step("out", "H", 1), Step("missing", "H", 1)])
    assert runs(truth) == ["H", "D"]


def test_out_ball_leaves_pitch_and_players_slow():
    m, truth = make_match([Step("dribble", "H", 2), Step("out", "H", 3)], start=100)
    ball = ball_m(m)[150:225]
    labels = truth.iloc[150:225].to_numpy()
    outside = (ball[:, 0] < 0) | (ball[:, 0] > W) | (ball[:, 1] < 0) | (ball[:, 1] > H)
    dead = labels == "D"
    assert dead.any() and (outside == dead).all()  # D exactly while the ball is outside
    first = dead.argmax()
    assert dead[first:].all() and (labels[:first] == "H").all() and first == 75 // 4
    _, _, speed = player_arrays(m)
    out_speed = speed[50:]
    assert out_speed[dead].max() <= 0.8
    assert out_speed[:first].mean() > 0.3  # players keep moving while the ball is still live
    assert speed[:50].mean() > 0.3


def test_restart_tag_window():
    m, _ = make_match(
        [Step("dribble", "H", 4), Step("out", "H", 1), Step("restart", "A", 2)], start=100
    )
    step_start, step_end = 100 + 125, 100 + 125 + 50
    assert len(m.tags) == 1
    t = m.tags.iloc[0]
    assert t["code"] == "SET PIECES" and t["team"] == "A" and t["type"] == "Throw-in"
    assert t["half"] == 1
    assert t["start_frame"] == step_start - 88 and t["end_frame"] == step_end


def test_restart_tag_clipped_at_block_start():
    m, _ = make_match([Step("restart", "H", 2), Step("dribble", "H", 1)], start=100)
    t = m.tags.iloc[0]
    assert t["start_frame"] == 100 and t["end_frame"] == 150


def test_no_restart_means_empty_valid_tags():
    m, _ = make_match([Step("dribble", "H", 1)])
    assert len(m.tags) == 0
    validate_tags(m.tags, m.meta)


def test_restart_ball_is_placed_inside_pitch_near_player_of_team():
    m, _ = make_match(
        [Step("dribble", "H", 2), Step("out", "H", 2), Step("restart", "A", 3)], start=100
    )
    ball = ball_m(m)[200:275]  # restart step
    assert ((ball[:, 0] > 0) & (ball[:, 0] < W) & (ball[:, 1] > 0) & (ball[:, 1] < H)).all()
    _, xy, speed = player_arrays(m)
    last = xy[-1]
    d = np.linalg.norm(last - ball[-1], axis=1)
    assert d[11:].min() < 0.5  # an A player holds the ball by the end
    all_d = np.linalg.norm(xy[-75:] - ball[:, None, :], axis=2)
    assert (all_d[:, 11:].min(axis=1) <= 0.5).all()  # a player of the team is on the ball always
    assert speed[100:].max() > 1.0  # players move again after the out step


def test_dribble_carrier_runs_at_5ms_with_ball_close():
    m, _ = make_match([Step("dribble", "H", 4)], start=0)
    _, xy, speed = player_arrays(m)
    ball = ball_m(m)[:100]
    d = np.linalg.norm(xy - ball[:, None, :], axis=2)
    assert (d.min(axis=1) <= 0.5).all()
    carrier = d[0].argmin()
    assert (d[:, carrier] <= 0.5).all()
    assert abs(speed[1:, carrier].mean() - 5.0) < 0.2
    assert speed[1:, carrier].min() > 4.8 and speed[1:, carrier].max() < 5.2


def test_pass_ball_travels_straight_and_fast_between_teammates():
    m, truth = make_match([Step("dribble", "H", 2), Step("pass", "H", 2)], start=0)
    ball = ball_m(m)
    sp = np.linalg.norm(np.diff(ball, axis=0), axis=1) * 25
    flight = np.where(sp[49:] >= 12)[0] + 49  # diffs inside the pass step
    assert len(flight) >= 5
    a, b = ball[flight[0]], ball[flight[-1] + 1]
    direction = (b - a) / np.linalg.norm(b - a)
    rel = ball[flight[0] : flight[-1] + 2] - a
    cross = rel[:, 0] * direction[1] - rel[:, 1] * direction[0]
    assert np.abs(cross).max() < 1e-6
    _, xy, _ = player_arrays(m)
    start_d = np.linalg.norm(xy[flight[0]] - a, axis=1)
    end_d = np.linalg.norm(xy[flight[-1] + 1] - b, axis=1)
    assert (
        start_d[:11].min() < 0.5 and end_d[:11].min() < 0.5 and start_d.argmin() != end_d.argmin()
    )
    assert (truth.iloc[:100] == "H").all()


def test_interception_ball_goes_from_current_team_to_interceptor():
    m, truth = make_match([Step("dribble", "H", 2), Step("interception", "A", 2)], start=0)
    ball = ball_m(m)[:100]
    _, xy, _ = player_arrays(m)
    sp = np.r_[0, np.linalg.norm(np.diff(ball, axis=0), axis=1) * 25]
    d = np.linalg.norm(xy - ball[:, None, :], axis=2)
    controlled = np.where((d[:, 11:].min(axis=1) <= 0.5) & (sp < 1.0))[0]
    controlled = controlled[controlled >= 50]
    assert len(controlled) > 0
    assert truth.iloc[controlled[0]] == "A"
    assert truth.iloc[controlled[0] - 1] == "H"
    flight_speed = sp[50 : controlled[0]]
    assert flight_speed.max() >= 12


def test_deflection_touches_opponent_briefly_without_control():
    m, truth = make_match([Step("dribble", "H", 2), Step("deflection", "H", 3)], start=0)
    ball = ball_m(m)[50:125]
    _, xy, _ = player_arrays(m)
    sp = np.r_[0, np.linalg.norm(np.diff(ball, axis=0), axis=1) * 25]
    d = np.linalg.norm(xy[50:125] - ball[:, None, :], axis=2)
    touch = np.where(d[:, 11:].min(axis=1) <= 0.5)[0]
    assert 1 <= len(touch) <= 2
    assert sp[touch].min() >= 12  # moving fast at the touch, never controlled
    assert (truth.iloc[50:125] == "H").all()
    assert d[-1, :11].min() <= 0.5  # ends with a current-team player


def test_aerial_is_never_controlled():
    for seed in range(4):
        m, truth = make_match([Step("dribble", "H", 2), Step("aerial", "H", 3)], start=0, seed=seed)
        ball = ball_m(m)[50:125]
        _, xy, _ = player_arrays(m)
        sp = np.r_[0, np.linalg.norm(np.diff(ball, axis=0), axis=1) * 25][1:]
        d = np.linalg.norm(xy[50:125] - ball[:, None, :], axis=2)[1:]
        assert not ((d.min(axis=1) <= 0.5) & (sp < 2.0)).any()
        assert (truth.iloc[50:125] == "H").all()


def test_speed_matches_displacement():
    m, _ = make_match(
        [
            Step("dribble", "H", 2),
            Step("pass", "H", 2),
            Step("interception", "A", 2),
            Step("out", "A", 1),
            Step("restart", "H", 2),
        ],
        start=30,
    )
    _, xy, speed = player_arrays(m)
    implied = np.linalg.norm(np.diff(xy, axis=0), axis=2) * 25
    assert np.allclose(speed[1:], implied, atol=1e-6)
    assert np.allclose(speed[0], speed[1])
    assert np.isfinite(speed).all() and (speed >= 0).all()


def test_players_stay_on_pitch():
    m, _ = make_match(
        [Step("dribble", "A", 5), Step("pass", "A", 3), Step("dribble", "A", 6)], seed=5
    )
    assert m.players[["x", "y"]].min().min() >= 0 and m.players[["x", "y"]].max().max() <= 1


@pytest.mark.parametrize("seed", range(5))
def test_long_mixed_match_is_valid_for_many_seeds(seed):
    steps = []
    team, other = "H", "A"
    for ev in ["dribble", "pass", "deflection", "aerial", "interception", "dribble", "pass"]:
        if ev == "interception":
            team, other = other, team
        steps.append(Step(ev, team, 1.5))
    steps += [Step("missing", "H", 1), Step("out", team, 1), Step("restart", other, 1.5)]
    m, truth = make_match(steps, seed=seed)
    validate_all(m, truth)
    assert runs(truth) == ["H", "A", "D", "H"]


def test_full_length_pads_with_x():
    s = pd.Series(["H", "A", "D"], name="label")
    out = full_length(s, n=6)
    assert out.tolist() == ["H", "A", "D", "X", "X", "X"]
    assert out.index.tolist() == list(range(6))
    assert full_length(s).shape == (199319,)
    assert full_length(s).iloc[3:].eq("X").all()
    with pytest.raises(ValueError, match="longer"):
        full_length(s, n=2)


@pytest.mark.parametrize("seconds", [0.5, 1, 2, 3])
def test_interception_switches_at_half_of_the_step(seconds):
    n = round(seconds * 25)
    _, truth = make_match([Step("dribble", "H", 1), Step("interception", "A", seconds)], start=0)
    assert (truth.iloc[25 : 25 + n // 2] == "H").all()
    assert (truth.iloc[25 + n // 2 : 25 + n] == "A").all()
