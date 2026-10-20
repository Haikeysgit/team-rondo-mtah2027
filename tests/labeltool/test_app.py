import numpy as np
import pandas as pd
import pytest

from rondo.eval import windows
from rondo.labeltool import app
from rondo.labeltool.app import Session, _save
from tests.synthetic import Step, make_match


def test_save_keeps_transition_marks_out_of_window_discovery(tmp_path, monkeypatch):
    n_frames = 300
    public_path = tmp_path / "public.csv"
    pd.DataFrame({"frame": range(n_frames), "label": ["X"] * n_frames}).to_csv(
        public_path, index=False
    )
    monkeypatch.setattr(windows, "public_gt_path", lambda: public_path)

    _save(tmp_path, "w1", {10: "H", 20: "A"}, 10, 30, n_frames)

    assert (tmp_path / "labels" / "w1.csv").is_file()
    assert (tmp_path / "labels" / "marks" / "w1_marks.csv").is_file()
    assert not (tmp_path / "labels" / "w1_marks.csv").exists()
    assert list(windows.load_windows(tmp_path)) == ["public", "w1"]


class _BlankReader:
    def __init__(self):
        self.closed = False

    def read(self, frame: int) -> np.ndarray:
        return np.zeros((40, 60, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def session():
    match, _ = make_match([Step("dribble", "H", 3)], n_frames=300)
    n = match.meta.n_frames
    matrices = np.full((n, 3, 3), np.nan)
    return Session(match, _BlankReader(), matrices, np.zeros(n, dtype=bool))


@pytest.fixture
def keys(monkeypatch):
    """Feed ``app.run`` a fixed key sequence instead of a live OpenCV window."""
    pressed: list[int] = []
    monkeypatch.setattr(app.cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(app.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(app.cv2, "destroyWindow", lambda *a: None)
    monkeypatch.setattr(app.cv2, "waitKeyEx", lambda *a: pressed.pop(0))
    return pressed


SAVE_A_MARK = [ord("h"), ord("s"), app.KEY_QUIT]


def test_labelling_saves_marks(tmp_path, session, keys):
    keys += SAVE_A_MARK
    app.run(tmp_path, "w1", 10, 50, session=session)
    assert (tmp_path / "labels" / "w1.csv").is_file()


def test_review_mode_never_writes_a_window(tmp_path, session, keys, capsys):
    keys += SAVE_A_MARK
    app.run(tmp_path, "err_missed_30", 10, 50, session=session, read_only=True)
    assert not (tmp_path / "labels").exists()
    assert "review" in capsys.readouterr().out


def test_a_shared_session_stays_open_for_the_next_review(tmp_path, session, keys):
    keys += [app.KEY_QUIT, app.KEY_QUIT]
    app.run(tmp_path, "a", 10, 50, session=session, read_only=True)
    app.run(tmp_path, "b", 60, 90, session=session, read_only=True)
    assert not session.reader.closed
