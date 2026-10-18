"""Project pitch positions onto the video, and draw the labelling tool's overlay.

The projection is the official kit's own homography math (``vendor/mtah-kit/starter``), used
unchanged, like every part of the official kit: a different formula here would make the overlay lie about where the
official matrix actually places things.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pandas as pd

from rondo.kit import KIT_DIR

_STARTER_DIR = KIT_DIR / "starter"
if str(_STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(_STARTER_DIR))

import load_homography as _mtah_homography

load_homography = _mtah_homography.load  # (matrices, reprojection_error, valid), see its docstring

TEAM_COLOR = {"H": (255, 120, 0), "A": (0, 80, 255)}  # BGR: a blue-ish and a red-ish, not the kits
BALL_COLOR = (0, 230, 255)
HUD_COLOR = (255, 255, 255)
HUD_HEIGHT = 28


def project_points(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Normalised pitch coordinates ``(n, 2)``, 0..1, to image pixels ``(n, 2)``.

    Thin wrapper on the kit's own ``project`` (scalar/array x, y in, x, y out) so every caller in
    this package takes and returns one ``(n, 2)`` array.
    """
    xy = np.asarray(xy, dtype=np.float64)
    x_px, y_px = _mtah_homography.project(matrix, xy[:, 0], xy[:, 1])
    return np.stack([x_px, y_px], axis=1)


def draw_overlay(
    image: np.ndarray,
    players: pd.DataFrame,
    ball_xy: tuple[float, float] | None,
    homography: np.ndarray | None,
    hud: str,
) -> np.ndarray:
    """``image`` with players, the ball (if tracked) and a HUD line drawn on top, as a new array.

    ``players``: this frame's rows only (``team``, ``x``, ``y``). ``homography``: this frame's
    matrix, or ``None`` when the frame has no valid one (``valid`` False) -- nothing is projected
    then, since there is nowhere trustworthy to put it.
    """
    canvas = image.copy()
    if homography is not None:
        if len(players):
            pixels = project_points(homography, players[["x", "y"]].to_numpy())
            for (x, y), team in zip(pixels, players["team"]):
                colour = TEAM_COLOR.get(team, (200, 200, 200))
                cv2.circle(canvas, (round(x), round(y)), 6, colour, -1)
        if ball_xy is not None and not any(np.isnan(ball_xy)):
            bx, by = project_points(homography, np.array([ball_xy]))[0]
            cv2.circle(canvas, (round(bx), round(by)), 4, BALL_COLOR, -1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], HUD_HEIGHT), (0, 0, 0), -1)
    cv2.putText(canvas, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COLOR, 1, cv2.LINE_AA)
    return canvas
