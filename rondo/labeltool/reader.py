"""Exact-frame random access into a video, via periodic cached checkpoints.

OpenCV's ``CAP_PROP_POS_FRAMES`` seek is not reliable to a specific frame on most codecs: the
decoder can only start cleanly at a keyframe, so seeking straight to an arbitrary frame and
reading once can silently hand back the wrong image. ``FrameReader`` never does that: it only
seeks to a position it has itself decoded and cached, verifies the seek landed there by comparing
pixels, and then decodes forward frame by frame -- counting as it goes -- to the one actually
asked for. That makes the result exact by construction, not by trusting the codec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import cv2
import numpy as np

OFFICIAL_FRAMES = 199319
CHECKPOINT_INTERVAL = 250  # frames (~10s @ 25fps): bounds the worst-case reseek/redecode cost


class FrameReader:
    """Random access by frame index into ``video_path``, verified exact on open.

    Raises ``ValueError`` immediately if the file does not decode to ``expected_frames`` frames
    (the kit's FAQ: a different count means a different decode of the same file).
    """

    def __init__(self, video_path: Path, expected_frames: int = OFFICIAL_FRAMES):
        self.path = Path(video_path)
        self._checkpoints: dict[int, np.ndarray] = {}
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"could not open video: {self.path}")
        count = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            if count % CHECKPOINT_INTERVAL == 0:
                self._checkpoints[count] = image.copy()
            count += 1
        if count != expected_frames:
            raise ValueError(
                f"{self.path} decodes to {count} frames, expected {expected_frames} "
                "(a different count means a different decode: re-export or re-check the file)"
            )
        self.n_frames = count
        self._pos = count  # the capture is exhausted after the counting pass above

    def read(self, frame: int) -> np.ndarray:
        """The exact image for ``frame`` (BGR, as OpenCV decodes it)."""
        if not 0 <= frame < self.n_frames:
            raise ValueError(f"frame {frame} out of range 0..{self.n_frames - 1}")
        if frame in self._checkpoints:
            return self._checkpoints[frame].copy()

        checkpoint = (frame // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
        needs_seek = frame < self._pos or checkpoint > self._pos
        if needs_seek and not self._verified_seek(checkpoint):
            self._restart()  # this decode cannot be trusted to seek: read from the start

        image = None
        while self._pos <= frame:
            ok, image = self._cap.read()
            if not ok:
                raise RuntimeError(f"{self.path} ended before frame {frame} on a re-read")
            self._pos += 1
        return image

    def _verified_seek(self, checkpoint: int) -> bool:
        """Try to land exactly on ``checkpoint``; True only once that is confirmed by pixels."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, checkpoint)
        ok, image = self._cap.read()
        if ok and np.array_equal(image, self._checkpoints[checkpoint]):
            self._pos = checkpoint + 1
            return True
        return False

    def _restart(self) -> None:
        """Reopen the video at frame 0, for when a seek cannot be trusted."""
        self._cap.release()
        self._cap = cv2.VideoCapture(str(self.path))
        self._pos = 0

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
