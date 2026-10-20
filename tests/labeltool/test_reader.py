import tracemalloc

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


def test_frame_reader_keeps_no_decoded_images_per_checkpoint(video_path, monkeypatch):
    # The real video is 1920x1080 with ~800 checkpoints: keeping each image would cost ~5 GB.
    # Here 30 checkpoints of a 64x64 frame would hold ~370 KB; allow less than one frame.
    monkeypatch.setattr("rondo.labeltool.reader.CHECKPOINT_INTERVAL", 10)
    one_frame = SIZE[0] * SIZE[1] * 3
    tracemalloc.start()
    try:
        reader = FrameReader(video_path, expected_frames=N_FRAMES)
        held, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert held < one_frame
    for i in (0, 10, 155, 3, 299):
        assert abs(round(reader.read(i).mean()) - (i % 256)) <= 4


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


# -- the frame index: count the video once, then reopen in seconds ------------------------------


def test_reopening_with_an_index_skips_the_count(video_path, tmp_path, monkeypatch):
    index = tmp_path / "video_index.json"
    FrameReader(video_path, expected_frames=N_FRAMES, index_path=index)
    assert index.is_file()

    def no_scan(*args, **kwargs):
        raise AssertionError("the video was decoded again")

    monkeypatch.setattr("rondo.labeltool.reader._scan", no_scan)
    reader = FrameReader(video_path, expected_frames=N_FRAMES, index_path=index)
    for i in (299, 0, 150, 1, 249, 250):
        assert abs(round(reader.read(i).mean()) - (i % 256)) <= 4


def test_an_index_for_another_file_is_not_trusted(tmp_path):
    index = tmp_path / "video_index.json"
    path = tmp_path / "v.avi"
    _make_video(path, n_frames=60)
    FrameReader(path, expected_frames=60, index_path=index)
    _make_video(path, n_frames=80)  # same path, a different file
    with pytest.raises(ValueError):
        FrameReader(path, expected_frames=60, index_path=index)


def test_a_wrong_count_is_never_written_to_the_index(video_path, tmp_path):
    index = tmp_path / "video_index.json"
    with pytest.raises(ValueError):
        FrameReader(video_path, expected_frames=N_FRAMES + 1, index_path=index)
    assert not index.exists()


def test_the_count_reports_progress(video_path):
    seen = []
    FrameReader(video_path, expected_frames=N_FRAMES, progress=seen.append)
    assert seen and seen[-1] == N_FRAMES and seen == sorted(seen)
