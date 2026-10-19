"""Interactive viewer for marking possession transitions on the tactical video.

Not unit tested: it opens a live OpenCV window and reads the keyboard, which only makes sense
against the real video, so it is checked by hand on the real video.
`expand.py`, `overlay.py` and `reader.py` hold everything here that *is* testable; new testable
logic belongs there, not in this loop.

Keys: ``,``/``.`` step -1/+1 frame, ``<``/``>`` step -25/+25 frame (left/right arrow and
shift+arrow are also wired, best-effort -- OpenCV's arrow-key codes are not portable across
platforms and builds, so confirm them against the real setup on Oct 20 and fix ``KEY_*`` below if
they don't match; the comma/period keys always work). ``h``/``a``/``d`` mark a transition to that
label at the current frame. ``u`` undoes the last mark. ``s`` saves. ``Esc`` quits.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from rondo import config, kit
from rondo.io.load import load_match
from rondo.labeltool.expand import expand_transitions
from rondo.labeltool.overlay import draw_overlay, load_homography
from rondo.labeltool.reader import FrameReader

# Best-effort; see the module docstring. These are the common Windows (highgui) waitKeyEx() codes.
KEY_LEFT, KEY_RIGHT = 2424832, 2555904
KEY_SHIFT_LEFT, KEY_SHIFT_RIGHT = 2424832, 2555904  # Windows does not reliably distinguish Shift
KEY_STEP_BACK, KEY_STEP_FWD = ord(","), ord(".")  # always reliable: use these if arrows misfire
KEY_JUMP_BACK, KEY_JUMP_FWD = ord("<"), ord(">")
KEY_UNDO, KEY_SAVE, KEY_QUIT = ord("u"), ord("s"), 27  # Esc
STEP_KEYS = {ord("h"): "H", ord("a"): "A", ord("d"): "D"}
JUMP_FRAMES = 25


def _sorted_marks(marks: dict[int, str]) -> pd.DataFrame:
    frames = sorted(marks)
    return pd.DataFrame({"frame": frames, "label": [marks[f] for f in frames]})


def _save(data_dir: Path, window: str, marks: dict[int, str], start: int, end: int, n: int) -> None:
    if not marks or min(marks) != start:
        print(f"not saved: need a mark at the window start ({start}) before the first save")
        return
    out_dir = Path(data_dir) / "labels"
    marks_dir = out_dir / "marks"
    out_dir.mkdir(parents=True, exist_ok=True)
    marks_dir.mkdir(parents=True, exist_ok=True)
    marks_df = _sorted_marks(marks)
    marks_path = marks_dir / f"{window}_marks.csv"
    marks_df.to_csv(marks_path, index=False)
    expand_transitions(marks_df, start, end, n).to_csv(out_dir / f"{window}.csv")
    print(f"saved {len(marks_df)} mark(s) to {marks_path} and {window}.csv")


def run(data_dir: Path | None, window: str, start: int, end: int) -> None:
    """Open the labelling window on ``[start, end]`` of the tactical video; blocks until closed."""
    if start > end:
        raise ValueError(f"start ({start}) must not be after end ({end})")
    data_dir = Path(data_dir) if data_dir is not None else config.data_dir()
    match = load_match(data_dir)
    video_path = Path(kit.games.video_path(data_dir=str(data_dir)))
    reader = FrameReader(video_path, expected_frames=match.meta.n_frames)
    matrices, _, valid = load_homography(data_dir=str(data_dir))
    players_by_frame = {f: rows for f, rows in match.players.groupby("frame")}
    ball = match.frames.set_index("frame")[["ball_x", "ball_y"]]

    marks: dict[int, str] = {}
    history: list[int] = []  # frames marked, in order, for undo
    frame = start
    win_name = f"rondo label - {window}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    print(__doc__.split("Keys:", 1)[1].strip())
    try:
        while True:
            image = reader.read(frame)
            players = players_by_frame.get(frame, match.players.iloc[0:0])
            ball_xy = tuple(ball.loc[frame]) if frame in ball.index else None
            homography = matrices[frame] if frame < len(valid) and valid[frame] else None
            label = marks.get(frame, "?")
            hud = f"{window}  frame {frame}  [{start}-{end}]  mark={label}  marks={len(marks)}"
            cv2.imshow(win_name, draw_overlay(image, players, ball_xy, homography, hud))

            key = cv2.waitKeyEx(0)
            if key in (KEY_LEFT, KEY_STEP_BACK):
                frame = max(start, frame - 1)
            elif key in (KEY_RIGHT, KEY_STEP_FWD):
                frame = min(end, frame + 1)
            elif key in (KEY_SHIFT_LEFT, KEY_JUMP_BACK):
                frame = max(start, frame - JUMP_FRAMES)
            elif key in (KEY_SHIFT_RIGHT, KEY_JUMP_FWD):
                frame = min(end, frame + JUMP_FRAMES)
            elif key in STEP_KEYS:
                if frame not in marks:
                    history.append(frame)
                marks[frame] = STEP_KEYS[key]
            elif key == KEY_UNDO and history:
                del marks[history.pop()]
            elif key == KEY_SAVE:
                _save(data_dir, window, marks, start, end, match.meta.n_frames)
            elif key == KEY_QUIT:
                break
    finally:
        cv2.destroyWindow(win_name)
        reader.close()
