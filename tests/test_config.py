from pathlib import Path

from rondo import config


def test_data_dir_uses_mtah_data_dir_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("MTAH_DATA_DIR", str(tmp_path / "bundle"))
    monkeypatch.setattr(config, "LAPTOP_DATA_DIR", tmp_path)  # exists, but the variable wins
    assert config.data_dir() == tmp_path / "bundle"


def test_data_dir_falls_back_to_the_laptop_folder_when_it_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    monkeypatch.setattr(config, "LAPTOP_DATA_DIR", tmp_path)
    assert config.data_dir() == tmp_path


def test_data_dir_falls_back_to_the_repo_data_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    monkeypatch.setattr(config, "LAPTOP_DATA_DIR", tmp_path / "absent")
    repo_root = Path(config.__file__).resolve().parent.parent
    assert config.data_dir() == repo_root / "data"


def test_laptop_folder_is_the_team_default():
    assert config.LAPTOP_DATA_DIR == Path(r"C:\rondo-data")
