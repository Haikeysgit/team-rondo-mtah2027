"""Turn sparse hand-marked transitions into a full-length label series."""

from __future__ import annotations

import numpy as np
import pandas as pd

OFFICIAL_FRAMES = 199319
VALID_LABELS = ("H", "A", "D")


def expand_transitions(
    marks: pd.DataFrame, start: int, end: int, n_frames: int = OFFICIAL_FRAMES
) -> pd.Series:
    """Full-length labels (index 0..``n_frames``-1) from ``marks`` (``frame, label``).

    Each mark means "from this frame on, the label is ...", up to the next mark or to ``end``
    (inclusive). ``X`` outside ``[start, end]``. The first mark must be at ``start``, marks must be
    strictly increasing and inside the window, and every label must be ``H``, ``A`` or ``D``.
    """
    if start > end:
        raise ValueError(f"start ({start}) must not be after end ({end})")
    if end >= n_frames:
        raise ValueError(f"end ({end}) must be before n_frames ({n_frames})")
    if marks.empty:
        raise ValueError("marks must have at least one row, the first at the window start")

    frames = marks["frame"].to_numpy()
    labels = marks["label"].to_numpy()
    if int(frames[0]) != start:
        raise ValueError(f"first mark must be at the window start ({start}), got {frames[0]}")
    if (np.diff(frames) <= 0).any():
        raise ValueError("marks frame must be strictly increasing")
    if frames[-1] > end:
        raise ValueError(f"marks must not go past the window end ({end})")
    bad = set(labels) - set(VALID_LABELS)
    if bad:
        raise ValueError(f"marks label must be one of {VALID_LABELS}, got {sorted(bad)}")

    out = np.full(n_frames, "X", dtype=object)
    bounds = [int(f) for f in frames] + [end + 1]
    for i, label in enumerate(labels):
        out[bounds[i] : bounds[i + 1]] = label
    return pd.Series(out, index=pd.RangeIndex(n_frames, name="frame"), name="label", dtype="str")
