import numpy as np
import pandas as pd

from rondo.labeltool.overlay import draw_overlay, project_points


def test_project_points_identity_returns_inputs():
    xy = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.3]])
    out = project_points(np.eye(3), xy)
    assert out.shape == xy.shape
    np.testing.assert_allclose(out, xy)


def test_project_points_scale_and_translate():
    # pixel = 100*x + 10, 100*y + 20
    matrix = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]])
    out = project_points(matrix, np.array([[0.0, 0.0], [1.0, 1.0]]))
    np.testing.assert_allclose(out, [[10.0, 20.0], [110.0, 120.0]])


def test_project_points_single_point():
    out = project_points(np.eye(3), np.array([[0.2, 0.8]]))
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out[0], [0.2, 0.8])


def test_draw_overlay_returns_same_shape_and_does_not_mutate_input():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    original = image.copy()
    players = pd.DataFrame({"team": ["H", "A"], "x": [0.3, 0.7], "y": [0.5, 0.5]})
    out = draw_overlay(image, players, (0.5, 0.5), np.eye(3), "frame 0")
    assert out.shape == image.shape
    np.testing.assert_array_equal(image, original)  # the input is never drawn on in place
    assert not np.array_equal(out, original)  # something was actually drawn


def test_draw_overlay_with_no_homography_still_draws_the_hud():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = draw_overlay(image, pd.DataFrame({"team": [], "x": [], "y": []}), None, None, "no H")
    assert out.shape == image.shape
    assert not np.array_equal(out, image)  # the HUD bar was drawn even with no homography


def test_draw_overlay_with_no_ball_does_not_crash():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    players = pd.DataFrame({"team": ["H"], "x": [0.5], "y": [0.5]})
    out = draw_overlay(image, players, None, np.eye(3), "no ball")
    assert out.shape == image.shape
