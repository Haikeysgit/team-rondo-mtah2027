from rondo import kit
from rondo.config import data_dir


def test_kit_constants():
    assert kit.games.GAMES["BAR-ATH"]["n_video_frames"] == 199319
    assert kit.games.half_video_blocks("BAR-ATH") == [(1, 11185, 90651), (2, 110808, 190197)]


def test_kit_scorer_constant_guess_is_zero():
    import numpy as np

    gt = np.full(199319, kit.pm.LABELS.index("X"), dtype=np.int8)
    gt[11185:18685] = kit.pm.LABELS.index("H")
    pred = np.full(199319, kit.pm.LABELS.index("A"), dtype=np.int8)
    assert kit.pm.evaluate(gt, pred)["spell_f1_mean"] == 0.0


def test_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MTAH_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path
