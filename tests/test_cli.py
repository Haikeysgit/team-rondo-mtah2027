import tomllib
from pathlib import Path

import pytest

from rondo.cli import main
from rondo.post import validate_official

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


@pytest.mark.parametrize("command", ["score", "label", "verify-repro"])
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
