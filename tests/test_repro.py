"""Reproducibility: the same input gives a byte-identical submission, on every platform."""

import subprocess

import numpy as np
import pandas as pd

from rondo import cli, registry
from rondo.contracts import MatchData
from rondo.pipeline import run
from rondo.post import write_submission
from tests.synthetic import OFFICIAL_FRAMES, Step, make_match

# SHA-256 of the golden submission. Computed once on Windows; CI checks it on Linux and Windows.
# It changes only if the pipeline, the Viterbi decoder, finalize/write_submission or the
# synthetic generator change behaviour: then check why, and update it in the same PR.
GOLDEN_SHA256 = "9e64e9c9f46533d08667978e86c11cb2ddbc9910fa939bd5fab9edce8d4d2516"

GOLDEN_STEPS = [
    Step("dribble", "H", 3),
    Step("pass", "H", 2),
    Step("deflection", "H", 1.5),
    Step("dribble", "H", 2),
    Step("interception", "A", 2),
    Step("dribble", "A", 3),
    Step("aerial", "A", 2),
    Step("pass", "A", 2),
    Step("out", "A", 3),
    Step("restart", "H", 3),
    Step("dribble", "H", 2),
    Step("missing", "H", 1.5),
    Step("interception", "A", 2),
    Step("dribble", "A", 2),
    Step("interception", "H", 2),
    Step("dribble", "H", 3),
]
GOLDEN_START = 11185  # the official first-half kick-off frame
GOLDEN_CONFIG = {
    "signals": ["stub_golden_distance"],
    "emission": "stub_golden_emission",
    "decoder": "viterbi",
    "emission_params": {},
    "decoder_params": {
        "switch_cost": 8.0,
        "dead_entry_cost": 25.0,
        "dead_exit_cost": 25.0,
        "restart_bonus": 6.0,
    },
}


def stub_golden_distance(match: MatchData) -> pd.DataFrame:
    """Per frame: each team's nearest player to the ball (m), and whether the ball is out."""
    frames = match.frames[["frame", "ball_x", "ball_y"]]
    joined = match.players.merge(frames, on="frame")
    width, height = match.meta.pitch
    dx = (joined["x"] - joined["ball_x"]) * width
    dy = (joined["y"] - joined["ball_y"]) * height
    joined["d"] = np.hypot(dx, dy)
    best = joined.groupby(["frame", "team"])["d"].min().unstack("team")
    best = best.reindex(index=frames["frame"], columns=["H", "A"])
    ball = frames[["ball_x", "ball_y"]]
    return pd.DataFrame(
        {
            "frame": frames["frame"].to_numpy(),
            "stub_golden_home_m": best["H"].to_numpy(),
            "stub_golden_away_m": best["A"].to_numpy(),
            "stub_golden_out": (ball.lt(0) | ball.gt(1)).any(axis=1).to_numpy(),
        }
    )


class StubGoldenEmission:
    """Soft nearest-team evidence: close calls are near 50/50, a missing ball is an exact tie."""

    def fit(self, signals, labels):
        return self

    def predict(self, signals: pd.DataFrame) -> pd.DataFrame:
        home = signals["stub_golden_home_m"].to_numpy(dtype="float64")
        away = signals["stub_golden_away_m"].to_numpy(dtype="float64")
        out = signals["stub_golden_out"].to_numpy(dtype=bool)
        with np.errstate(invalid="ignore"):
            share = 1.0 / (1.0 + np.exp((home - away) / 3.0))  # P(home | not dead)
        p_h, p_a, p_d = 0.9 * share, 0.9 * (1.0 - share), np.full(len(home), 0.1)
        unknown = np.isnan(share)
        p_h[unknown] = p_a[unknown] = p_d[unknown] = 1.0 / 3.0
        p_h[out], p_a[out], p_d[out] = 0.1, 0.1, 0.8
        return pd.DataFrame(
            {"frame": signals["frame"].to_numpy(), "p_H": p_h, "p_A": p_a, "p_D": p_d}
        )


def _register_once(register, name, obj) -> None:
    try:
        register(name)(obj)
    except ValueError:  # already registered by an earlier collection
        pass


_register_once(registry.register_signal, "stub_golden_distance", stub_golden_distance)
_register_once(registry.register_emission, "stub_golden_emission", StubGoldenEmission)


def _golden_labels() -> pd.DataFrame:
    match, _ = make_match(GOLDEN_STEPS, n_frames=OFFICIAL_FRAMES, start=GOLDEN_START, seed=7)
    return run(match, GOLDEN_CONFIG)


def test_golden_case_exercises_the_decoder():
    labels = _golden_labels()
    first_half = labels["label"].to_numpy()[:110808]
    graded = first_half[first_half != "X"]
    changes = int((graded[1:] != graded[:-1]).sum())
    assert {"H", "A", "D"} <= set(graded) and changes >= 4
    # finalize forces the dead ball before the second-half kick-off on a full-length match.
    assert set(labels["label"].to_numpy()[110808:113742]) == {"D"}


def test_golden_submission_sha256(tmp_path):
    digest = write_submission(_golden_labels(), tmp_path / "team-rondo.csv")
    assert digest == GOLDEN_SHA256


STUB_CONFIG = 'signals = ["stub_nearest"]\nemission = "stub_emission"\ndecoder = "stub_argmax"\n'


def test_verify_repro_runs_twice_and_reports_identical(capsys, monkeypatch, tmp_bundle, tmp_path):
    config = tmp_path / "stub.toml"
    config.write_text(STUB_CONFIG + "[emission_params]\n[decoder_params]\n")
    monkeypatch.setattr(cli, "RUN_MODULE", "tests.repro_entry")
    cache_file = tmp_bundle / "cache" / "match.pkl"
    # A stale cache from earlier: run 1 must ignore it (cold) and rewrite it.
    cache_file.parent.mkdir()
    cache_file.write_bytes(b"stale")
    calls = []
    real_run = subprocess.run

    def spy(argv, **kwargs):
        done = real_run(argv, **kwargs)
        calls.append((argv, cache_file.stat().st_mtime_ns, cache_file.read_bytes()))
        return done

    monkeypatch.setattr(cli.subprocess, "run", spy)
    argv = ["verify-repro", "--data-dir", str(tmp_bundle), "--config", str(config)]
    assert cli.main(argv + ["--expected-frames", "50", "--validate-rows", "50"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3 and lines[-1] == "IDENTICAL"
    first, second = (line.split()[-1] for line in lines[:2])
    assert first == second and len(first) == 64

    (cold, after_cold, cold_bytes), (warm, after_warm, warm_bytes) = calls
    assert "--refresh-cache" in cold and "--no-cache" not in cold  # built from the source files
    assert "--refresh-cache" not in warm and "--no-cache" not in warm
    assert cold_bytes != b"stale"  # run 1 rebuilt the cache ...
    assert (after_warm, warm_bytes) == (after_cold, cold_bytes)  # ... and run 2 only read it


def test_verify_repro_reports_a_failed_run(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "RUN_MODULE", "tests.repro_entry")
    assert cli.main(["verify-repro", "--data-dir", str(tmp_path / "nothing")]) == 1
    assert "failed" in capsys.readouterr().out
