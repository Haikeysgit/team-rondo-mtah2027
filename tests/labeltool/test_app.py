import pandas as pd

from rondo.eval import windows
from rondo.labeltool.app import _save


def test_save_keeps_transition_marks_out_of_window_discovery(tmp_path, monkeypatch):
    n_frames = 300
    public_path = tmp_path / "public.csv"
    pd.DataFrame(
        {"frame": range(n_frames), "label": ["X"] * n_frames}
    ).to_csv(public_path, index=False)
    monkeypatch.setattr(windows, "public_gt_path", lambda: public_path)

    _save(tmp_path, "w1", {10: "H", 20: "A"}, 10, 30, n_frames)

    assert (tmp_path / "labels" / "w1.csv").is_file()
    assert (tmp_path / "labels" / "marks" / "w1_marks.csv").is_file()
    assert not (tmp_path / "labels" / "w1_marks.csv").exists()
    assert list(windows.load_windows(tmp_path)) == ["public", "w1"]
