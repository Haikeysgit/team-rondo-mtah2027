import tomllib
from pathlib import Path

import pandas as pd
import pytest

from rondo import cli, kit
from rondo.cli import main
from rondo.post import validate_official

BASELINE = kit.KIT_DIR / "evaluation" / "results" / "baseline_smooth50.csv"
STUB_CONFIG = 'signals = ["stub_nearest"]\nemission = "stub_emission"\ndecoder = "stub_argmax"\n'


def test_cli_check_prints_no_rows(capsys, tmp_bundle):
    assert main(["check", "--data-dir", str(tmp_bundle), "--expected-records", "50"]) == 0
    out = capsys.readouterr().out
    assert "n_records: 50" in out and "records" in out and "\n0," not in out


def test_cli_tags_prints_rates_and_no_rows(capsys, tmp_bundle):
    assert main(["tags", "--data-dir", str(tmp_bundle), "--expected-frames", "50"]) == 0
    out = capsys.readouterr().out
    assert "restart_windows:" in out and "prior_coverage_graded_pct:" in out
    assert "public: skipped" in out  # the public window is full-length, the bundle is not
    assert "\n0," not in out


def test_cli_tags_against_a_labels_file(capsys, tmp_bundle, tmp_path):
    labels = tmp_path / "pred.csv"
    labels.write_text("frame,label\n" + "".join(f"{f},H\n" for f in range(50)))
    argv = ["tags", "--data-dir", str(tmp_bundle), "--expected-frames", "50"]
    assert main([*argv, "--against", str(labels)]) == 0
    out = capsys.readouterr().out
    assert "pred.csv: restart team right" in out and "public" not in out


def test_cli_run_end_to_end(capsys, tmp_bundle, tmp_path):
    config = tmp_path / "stub.toml"
    config.write_text(STUB_CONFIG + "[emission_params]\n[decoder_params]\n")
    out_path = tmp_path / "out" / "team-rondo.csv"
    argv = ["run", "--data-dir", str(tmp_bundle), "--config", str(config)]
    argv += ["--out", str(out_path), "--expected-frames", "50"]
    assert main(argv) == 0
    text = capsys.readouterr().out
    for key in ("rows: 50", "sha256:", "validator: ok", "possessions:"):
        assert key in text
    assert out_path.read_bytes().startswith(b"frame,label\n0,X\n")
    assert validate_official(out_path, rows=50) is None
    assert "0,X" not in text


def test_cli_run_fails_when_validator_fails(tmp_bundle, tmp_path):
    # A 50-row file judged against the official 199,319 rows: the validator must object.
    config = tmp_path / "stub.toml"
    config.write_text(STUB_CONFIG)
    argv = ["run", "--data-dir", str(tmp_bundle), "--config", str(config), "--out"]
    argv += [str(tmp_path / "o.csv"), "--expected-frames", "50", "--validate-rows", "199319"]
    assert main(argv) == 1


def test_cli_rejects_unknown_arguments():
    with pytest.raises(SystemExit):
        main(["check", "--no-such-flag"])


def test_cli_label_requires_window_start_and_end():
    with pytest.raises(SystemExit):
        main(["label", "--window", "w1", "--start", "10"])  # no --end


def test_cli_label_calls_the_labeltool(monkeypatch, tmp_path):
    seen = {}

    def fake_run(data_dir, window, start, end):
        seen.update(data_dir=data_dir, window=window, start=start, end=end)

    monkeypatch.setattr("rondo.labeltool.app.run", fake_run)
    argv = ["label", "--data-dir", str(tmp_path), "--window", "w1", "--start", "10", "--end", "20"]
    assert main(argv) == 0
    assert seen == {"data_dir": tmp_path, "window": "w1", "start": 10, "end": 20}


def test_default_config_is_valid_toml():
    path = Path(__file__).resolve().parent.parent / "configs" / "default.toml"
    with open(path, "rb") as handle:
        cfg = tomllib.load(handle)
    assert cfg["emission"] == "rules_v0" and cfg["decoder"] == "viterbi"
    assert cfg["decoder_params"]["switch_cost"] == 8.0
    assert len(cfg["signals"]) == 6 and cfg["emission_params"] == {}


def test_cli_score_prints_the_official_numbers_per_window(capsys, tmp_path):
    assert main(["score", str(BASELINE), "--data-dir", str(tmp_path), "--bootstrap", "1000"]) == 0
    out = capsys.readouterr().out
    assert "public" in out and "combined" in out
    assert "0.3256" in out and "0.2140" in out and "0.4465" in out and "26 / 17" in out
    assert "\n0," not in out and "11185," not in out


def test_cli_score_writes_a_results_file(capsys, tmp_path):
    result = tmp_path / "results" / "2026-10-17-kiki-smooth50.md"
    argv = ["score", str(BASELINE), "--data-dir", str(tmp_path), "--result", str(result)]
    assert main(argv + ["--owner", "Kiki", "--notes", "Pipeline check."]) == 0
    text = result.read_text(encoding="utf-8")
    assert "# smooth50" in text and "- Owner: Kiki" in text and "| public | 0.3256 |" in text
    assert "rondo score" in text and "Pipeline check." in text


def test_cli_score_needs_an_owner_for_a_results_file(tmp_path):
    with pytest.raises(SystemExit):
        main(["score", str(BASELINE), "--result", str(tmp_path / "r.md")])


def test_cli_score_rejects_an_unknown_window(capsys, tmp_path):
    assert main(["score", str(BASELINE), "--data-dir", str(tmp_path), "--windows", "w9"]) == 2
    assert "w9" in capsys.readouterr().out


def test_cli_score_reports_an_invalid_submission(capsys, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("frame,label\n0,H\n")
    assert main(["score", str(bad), "--data-dir", str(tmp_path)]) == 1
    assert "bad.csv" in capsys.readouterr().out


def test_cli_score_scores_a_subset_without_the_public_window(capsys, tmp_path):
    labels = pd.Series("X", index=pd.RangeIndex(199319, name="frame"), name="label")
    labels.iloc[30000:30500] = "H"
    (tmp_path / "labels").mkdir()
    labels.to_frame().reset_index().to_csv(tmp_path / "labels" / "w1.csv", index=False)
    argv = ["score", str(BASELINE), "--data-dir", str(tmp_path), "--windows", "w1"]
    assert main(argv + ["--bootstrap", "10"]) == 0
    out = capsys.readouterr().out
    assert "w1" in out and "combined" in out and "public" not in out


def test_cli_score_shows_a_dash_for_a_window_without_possessions(capsys, tmp_path):
    labels = pd.Series("X", index=pd.RangeIndex(199319, name="frame"), name="label")
    labels.iloc[30000:30500] = "D"
    (tmp_path / "labels").mkdir()
    labels.to_frame().reset_index().to_csv(tmp_path / "labels" / "w1.csv", index=False)
    assert main(["score", str(BASELINE), "--data-dir", str(tmp_path), "--bootstrap", "10"]) == 0
    out = capsys.readouterr().out
    assert "nan" not in out and [ln for ln in out.splitlines() if ln.startswith("w1 ")]


def _run_argv(tmp_bundle, tmp_path, *extra):
    config = tmp_path / "stub.toml"
    config.write_text(STUB_CONFIG + "[emission_params]\n[decoder_params]\n")
    argv = ["run", "--data-dir", str(tmp_bundle), "--config", str(config), "--out"]
    return argv + [str(tmp_path / "o.csv"), "--expected-frames", "50", *extra]


def test_cli_run_no_cache_leaves_no_cache(tmp_bundle, tmp_path):
    assert main(_run_argv(tmp_bundle, tmp_path, "--no-cache")) == 0
    assert not (tmp_bundle / "cache").exists()
    assert main(_run_argv(tmp_bundle, tmp_path)) == 0
    assert (tmp_bundle / "cache" / "match.pkl").is_file()


def test_cli_run_refresh_cache_rebuilds_it(tmp_bundle, tmp_path, monkeypatch):
    seen = []
    real = cli.load_match
    monkeypatch.setattr(cli, "load_match", lambda *a, **k: seen.append(k) or real(*a, **k))
    assert main(_run_argv(tmp_bundle, tmp_path, "--refresh-cache")) == 0
    assert main(_run_argv(tmp_bundle, tmp_path, "--no-cache")) == 0
    assert main(_run_argv(tmp_bundle, tmp_path)) == 0
    assert [(k["cache"], k["refresh"]) for k in seen] == [
        (True, True),
        (False, False),
        (True, False),
    ]


def test_cli_run_cache_flags_are_exclusive(tmp_bundle, tmp_path):
    with pytest.raises(SystemExit):
        main(_run_argv(tmp_bundle, tmp_path, "--no-cache", "--refresh-cache"))


def test_cli_score_reports_a_bad_window_file_without_a_traceback(capsys, tmp_path):
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "w1.csv").write_text("frame,label\n0,H\n")
    assert main(["score", str(BASELINE), "--data-dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "w1.csv" in out and "Traceback" not in out


def test_cli_score_reports_clashing_windows_without_a_traceback(capsys, tmp_path):
    (tmp_path / "labels").mkdir()
    for name, label in (("w1", "H"), ("w2", "A")):
        labels = pd.Series("X", index=pd.RangeIndex(199319, name="frame"), name="label")
        labels.iloc[30000:30500] = label
        labels.to_frame().reset_index().to_csv(tmp_path / "labels" / f"{name}.csv", index=False)
    assert main(["score", str(BASELINE), "--data-dir", str(tmp_path)]) == 1
    assert "disagree" in capsys.readouterr().out


def _write_window(tmp_path: Path, name: str, labels: pd.Series) -> None:
    (tmp_path / "labels").mkdir(exist_ok=True)
    labels.to_frame().reset_index().to_csv(tmp_path / "labels" / f"{name}.csv", index=False)


def test_cli_errors_prints_counts_and_timestamps(capsys, tmp_path):
    truth = pd.Series("X", index=pd.RangeIndex(199319, name="frame"), name="label")
    truth.iloc[30000:30500] = "H"
    truth.iloc[30500:31000] = "A"
    _write_window(tmp_path, "w1", truth)

    pred = truth.copy()
    pred.iloc[30500:30525] = "H"  # the H -> A transition comes 25 frames late in the prediction
    pred_path = tmp_path / "pred.csv"
    pred.to_frame().reset_index().to_csv(pred_path, index=False)

    assert main(["errors", str(pred_path), "--data-dir", str(tmp_path), "--window", "w1"]) == 0
    out = capsys.readouterr().out
    assert "missed: 0" in out and "spurious: 0" in out and "late: 1" in out
    assert "20:20" in out  # the ground-truth transition's frame, 30500, at 25fps


def test_cli_errors_rejects_an_unknown_window(capsys, tmp_path):
    (tmp_path / "pred.csv").write_text("frame,label\n")  # never read: the window check is first
    argv = ["errors", str(tmp_path / "pred.csv"), "--data-dir", str(tmp_path), "--window", "w9"]
    assert main(argv) == 2
    assert "w9" in capsys.readouterr().out


def test_cli_errors_reports_an_invalid_submission(capsys, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("frame,label\n0,H\n")
    assert main(["errors", str(bad), "--data-dir", str(tmp_path), "--window", "public"]) == 1
    assert "bad.csv" in capsys.readouterr().out


def test_cli_errors_play_opens_each_moment_in_the_labeltool(monkeypatch, tmp_path):
    truth = pd.Series("X", index=pd.RangeIndex(199319, name="frame"), name="label")
    truth.iloc[30000:30500] = "H"
    truth.iloc[30500:31000] = "A"
    _write_window(tmp_path, "w1", truth)

    pred = pd.Series("X", index=pd.RangeIndex(199319, name="frame"), name="label")
    pred.iloc[30000:30500] = "H"  # the H -> A transition at 30500 is missed entirely
    pred.iloc[30200:30260] = "A"  # and a spurious A spell: two more moments
    pred_path = tmp_path / "pred.csv"
    pred.to_frame().reset_index().to_csv(pred_path, index=False)

    opened, seen = [], []

    class FakeSession:
        def close(self):
            opened.append("closed")

    monkeypatch.setattr(
        "rondo.labeltool.app.open_session",
        lambda data_dir: opened.append(data_dir) or FakeSession(),
    )
    monkeypatch.setattr(
        "rondo.labeltool.app.run",
        lambda data_dir, window, start, end, *, session, read_only: seen.append(
            (data_dir, window, start, end, session, read_only)
        ),
    )
    argv = ["errors", str(pred_path), "--data-dir", str(tmp_path), "--window", "w1", "--play"]
    assert main(argv) == 0
    assert opened == [tmp_path, "closed"]  # the video is opened (and counted) once for all
    assert len(seen) == 3
    assert all(read_only for *_, read_only in seen)  # review never writes into labels/
    assert len({id(session) for *_, session, _ in seen}) == 1
    missed = next(s for s in seen if s[1] == "err_missed_30500")
    assert missed[0] == tmp_path
    assert missed[2] == 30500 - cli.PLAY_MARGIN_FRAMES
    assert missed[3] == 30500 + cli.PLAY_MARGIN_FRAMES


def test_cli_errors_warns_its_output_is_frame_level(capsys, tmp_path):
    truth = pd.Series("X", index=pd.RangeIndex(199319, name="frame"), name="label")
    truth.iloc[30000:30500] = "H"
    _write_window(tmp_path, "w1", truth)
    pred_path = tmp_path / "pred.csv"
    truth.to_frame().reset_index().to_csv(pred_path, index=False)
    assert main(["errors", str(pred_path), "--data-dir", str(tmp_path), "--window", "w1"]) == 0
    assert "do not paste" in capsys.readouterr().out


def test_cli_errors_names_the_teams_of_a_spurious_change(capsys, tmp_path):
    truth = pd.Series("X", index=pd.RangeIndex(199319, name="frame"), name="label")
    truth.iloc[30000:30600] = "H"
    truth.iloc[30600:31000] = "A"
    _write_window(tmp_path, "w1", truth)
    pred = truth.copy()
    pred.iloc[30600:31000] = "H"  # the H -> A change at 30600 is missed
    pred.iloc[30200:30300] = "A"  # and a false A spell inside H's possession
    pred_path = tmp_path / "pred.csv"
    pred.to_frame().reset_index().to_csv(pred_path, index=False)
    assert main(["errors", str(pred_path), "--data-dir", str(tmp_path), "--window", "w1"]) == 0
    out = capsys.readouterr().out
    assert "spurious H -> A" in out and "nan" not in out


def _marks(tmp_path, text):
    marks = tmp_path / "labels" / "marks"
    marks.mkdir(parents=True, exist_ok=True)
    (marks / "w1_marks.csv").write_text(text)


def test_cli_rebuild_writes_the_window_from_edited_marks(capsys, tmp_path):
    _marks(tmp_path, "frame,label\n40000,H\n40100,A\n40200,D\n")
    argv = ["rebuild", "--window", "w1", "--start", "40000", "--end", "40299"]
    assert main([*argv, "--data-dir", str(tmp_path)]) == 0
    labels = pd.read_csv(tmp_path / "labels" / "w1.csv")["label"]
    assert len(labels) == 199319
    assert labels.iloc[40000:40100].eq("H").all() and labels.iloc[40100:40200].eq("A").all()
    assert labels.iloc[40200:40300].eq("D").all() and labels.iloc[40300] == "X"
    assert "rebuilt w1.csv from 3 mark(s)" in capsys.readouterr().out


def test_cli_rebuild_reports_bad_marks_without_a_traceback(capsys, tmp_path):
    _marks(tmp_path, "frame,label\n40000,H\n40100,B\n")
    argv = ["rebuild", "--window", "w1", "--start", "40000", "--end", "40299"]
    assert main([*argv, "--data-dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "w1_marks.csv" in out and "Traceback" not in out
    assert not (tmp_path / "labels" / "w1.csv").exists()
