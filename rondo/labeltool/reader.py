"""Exact-frame random access into a video, via periodic cached checkpoints.

OpenCV's ``CAP_PROP_POS_FRAMES`` seek is not reliable to a specific frame on most codecs: the
decoder can only start cleanly at a keyframe, so seeking straight to an arbitrary frame and
reading once can silently hand back the wrong image. ``FrameReader`` never does that: it only
seeks to a position it has itself decoded and fingerprinted, verifies the seek landed there by
comparing pixel digests, and then decodes forward frame by frame -- counting as it goes -- to the one actually
asked for. That makes the result exact by construction, not by trusting the codec.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Self

import cv2
import numpy as np

OFFICIAL_FRAMES = 199319
CHECKPOINT_INTERVAL = 250  # frames (~10s @ 25fps): bounds the worst-case reseek/redecode cost


PROGRESS_EVERY = 10_000  # frames between progress reports during the full decode


def _digest(image: np.ndarray) -> bytes:
    """A 16-byte fingerprint of a decoded image: equal pixels, equal digest."""
    return hashlib.blake2b(image.tobytes(), digest_size=16).digest()


def _scan(cap, progress: Callable[[int], None] | None) -> tuple[int, dict[int, bytes]]:
    """Decode every frame once: the count, and a digest of every checkpoint frame.

    Digests, not images: ~800 full-HD checkpoints would hold ~5 GB of pixels.
    """
    checkpoints: dict[int, bytes] = {}
    count = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if count % CHECKPOINT_INTERVAL == 0:
            checkpoints[count] = _digest(image)
        count += 1
        if progress is not None and count % PROGRESS_EVERY == 0:
            progress(count)
    if progress is not None and count % PROGRESS_EVERY:
        progress(count)
    return count, checkpoints


def _index_key(path: Path) -> dict:
    """What must match for a saved index to describe this file as this decoder reads it."""
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "opencv": cv2.__version__,
    }


def _read_index(path: Path | None, key: dict, expected: int):
    if path is None or not Path(path).is_file():
        return None
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        if saved["key"] != key or saved["count"] != expected:
            return None
        checkpoints = {int(f): bytes.fromhex(d) for f, d in saved["checkpoints"].items()}
    except (ValueError, KeyError, TypeError):
        return None  # unreadable or from an older format: decode again
    return saved["count"], checkpoints


def _write_index(path: Path, key: dict, count: int, checkpoints: dict[int, bytes]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "key": key,
        "count": count,
        "checkpoints": {str(f): d.hex() for f, d in sorted(checkpoints.items())},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body), encoding="utf-8")
    tmp.replace(path)  # atomic: a half-written index is never read


class FrameReader:
    """Random access by frame index into ``video_path``, verified exact on open.

    Raises ``ValueError`` immediately if the file does not decode to ``expected_frames`` frames
    (the kit's FAQ: a different count means a different decode of the same file).
    """

    def __init__(
        self,
        video_path: Path,
        expected_frames: int = OFFICIAL_FRAMES,
        index_path: Path | None = None,
        progress: Callable[[int], None] | None = None,
    ):
        """``index_path``: where to keep the count and checkpoint digests, so the full decode
        (tens of minutes on a full-HD match) happens once per file and decoder, not per open.
        ``progress``: called with the running frame count during that decode."""
        self.path = Path(video_path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"could not open video: {self.path}")
        key = _index_key(self.path)
        index = _read_index(index_path, key, expected_frames)
        if index is not None:
            count, self._checkpoints = index
            self._pos = 0
        else:
            count, self._checkpoints = _scan(self._cap, progress)
            self._pos = count  # the capture is exhausted after the counting pass
        if count != expected_frames:
            raise ValueError(
                f"{self.path} decodes to {count} frames, expected {expected_frames} "
                "(a different count means a different decode: re-export or re-check the file)"
            )
        if index is None and index_path is not None:
            _write_index(index_path, key, count, self._checkpoints)
        self.n_frames = count

    def read(self, frame: int) -> np.ndarray:
        """The exact image for ``frame`` (BGR, as OpenCV decodes it)."""
        if not 0 <= frame < self.n_frames:
            raise ValueError(f"frame {frame} out of range 0..{self.n_frames - 1}")
        checkpoint = (frame // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
        image = None
        if frame < self._pos or checkpoint > self._pos:
            image = self._verified_seek(checkpoint)
            if image is None:
                self._restart()  # this decode cannot be trusted to seek: read from the start

        while self._pos <= frame:
            ok, image = self._cap.read()
            if not ok:
                raise RuntimeError(f"{self.path} ended before frame {frame} on a re-read")
            self._pos += 1
        return image

    def _verified_seek(self, checkpoint: int) -> np.ndarray | None:
        """Land exactly on ``checkpoint`` and return its image, or ``None`` if the pixels show the
        seek landed elsewhere."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, checkpoint)
        ok, image = self._cap.read()
        if ok and _digest(image) == self._checkpoints[checkpoint]:
            self._pos = checkpoint + 1
            return image
        return None

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
