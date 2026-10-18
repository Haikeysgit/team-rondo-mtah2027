import cv2
import numpy as np
import pytest

from rondo.labeltool.reader import FrameReader

N_FRAMES = 300
SIZE = (64, 64)  # width, height


def _make_video(path, n_frames=N_FRAMES, size=SIZE, fps=25):
    """A video where frame i is a solid grey level i % 256.

    MJPG (each frame compressed on its own, no inter-frame prediction) keeps solid colours close
    to exact; mp4v's inter-frame coding drifts a few levels on a sequence like this.
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError("could not open a VideoWriter for the test fixture")
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("labeltool_video") / "test.avi"
    _make_video(path)
    return path


def test_frame_reader_reads_exact_frames_in_any_order(video_path):
    reader = FrameReader(video_path, expected_frames=N_FRAMES)
    for i in (299, 0, 150, 1, 149):
        image = reader.read(i)
        assert abs(round(image.mean()) - (i % 256)) <= 4


def test_frame_reader_rereading_the_same_frame_is_stable(video_path):
    reader = FrameReader(video_path, expected_frames=N_FRAMES)
    first = reader.read(150)
    second = reader.read(150)
    np.testing.assert_array_equal(first, second)


def test_frame_reader_wrong_expected_count_raises(video_path):
    with pytest.raises(ValueError):
        FrameReader(video_path, expected_frames=N_FRAMES + 1)


def test_frame_reader_rejects_out_of_range_frames(video_path):
    reader = FrameReader(video_path, expected_frames=N_FRAMES)
    with pytest.raises(ValueError):
        reader.read(N_FRAMES)
    with pytest.raises(ValueError):
        reader.read(-1)


def test_frame_reader_rejects_a_missing_file(tmp_path):
    with pytest.raises(ValueError):
        FrameReader(tmp_path / "nope.mp4", expected_frames=10)
