import tomllib
from pathlib import Path

import pandas as pd
import pytest

from rondo import kit
from rondo.cli import main
from rondo.post import validate_official

BASELINE = kit.KIT_DIR / "evaluation" / "results" / "baseline_smooth50.csv"
STUB_CONFIG = 'signals = ["stub_nearest"]\nemission = "stub_emission"\ndecoder = "stub_argmax"\n'


def test_cli_check_prints_no_rows(capsys, tmp_bundle):
    assert main(["check", "--data-dir", str(tmp_bundle), "--expected-records", "50"]) == 0
    out = capsys.readouterr().out
    assert "n_records: 50" in out and "records" in out and "\n0," not in out


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


@pytest.mark.parametrize("command", ["label"])
def test_cli_later_commands_return_2(command, capsys):
    assert main([command]) == 2
    assert "available once Task" in capsys.readouterr().out


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
