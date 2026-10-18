"""The labelling tool (Task 14): mark possession transitions on the tactical video by hand.

``expand_transitions`` (`expand.py`), ``project_points``/``draw_overlay`` (`overlay.py`) and
``FrameReader`` (`reader.py`) are the tested building blocks; `app.py` wires them into the
interactive ``rondo label`` command and is verified by hand, not by a test, since it needs a real
video window and keyboard.
"""

from rondo.labeltool.expand import expand_transitions
from rondo.labeltool.overlay import draw_overlay, load_homography, project_points
from rondo.labeltool.reader import FrameReader

__all__ = [
    "FrameReader",
    "draw_overlay",
    "expand_transitions",
    "load_homography",
    "project_points",
]
