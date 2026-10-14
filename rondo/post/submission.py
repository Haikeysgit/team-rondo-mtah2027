"""Turn decoder output into the submission table, write it, and check it with the official kit."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from rondo import kit
from rondo.contracts import Meta, validate_labels

OFFICIAL_FRAMES = 199319
# The kit's FAQ.md: the second-half graded block opens at frame 110808 but play only restarts at
# 113742; the frames in between are a dead ball and are graded D. Not derived from games.py
# (its kickoff_2nd anchor computes 113741, one frame off).
SECOND_HALF_GRADED_START = 110808
SECOND_HALF_KICKOFF = 113742


def finalize(labels: pd.DataFrame, meta: Meta) -> pd.DataFrame:
    """Full-length submission table: ``X`` outside the half blocks, forced ``D`` after halftime.

    ``labels`` (``frame, label``) may cover any subset of frames, but every graded frame must be
    present; labels given for frames outside the half blocks are ignored.
    """
    frames = labels["frame"].to_numpy()
    if frames.size and (frames.min() < 0 or frames.max() >= meta.n_frames):
        raise ValueError(f"labels have frames outside 0..{meta.n_frames - 1}")
    if len(np.unique(frames)) != len(frames):
        raise ValueError("labels have duplicate frames")

    graded = np.zeros(meta.n_frames, dtype=bool)
    for _, first, last in meta.half_blocks:
        graded[first : last + 1] = True
    covered = np.zeros(meta.n_frames, dtype=bool)
    covered[frames] = True
    missing = int((graded & ~covered).sum())
    if missing:
        raise ValueError(f"{missing} graded frame(s) have no label")

    out = np.full(meta.n_frames, "X", dtype="<U1")
    out[frames] = labels["label"].to_numpy()
    out[~graded] = "X"
    if meta.n_frames == OFFICIAL_FRAMES:
        out[SECOND_HALF_GRADED_START:SECOND_HALF_KICKOFF] = "D"
    result = pd.DataFrame({"frame": np.arange(meta.n_frames, dtype="int64"), "label": out})
    validate_labels(result, meta, allow_x=True)
    return result


def write_submission(labels: pd.DataFrame, path: Path) -> str:
    """Write ``frame,label`` with LF endings and return the SHA-256 (hex) of the bytes written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "frame,label\n" + "".join(
        f"{f},{label}\n" for f, label in zip(labels["frame"].tolist(), labels["label"].tolist())
    )
    data = text.encode("utf-8")
    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write(text)
    return hashlib.sha256(data).hexdigest()


def validate_official(path: Path, rows: int = OFFICIAL_FRAMES) -> str | None:
    """The official validator's reason the file is unusable, or ``None``. Never raises."""
    try:
        frame = kit.pm.read_label_csv(path)
        return kit.pm.validate_submission(frame, rows)
    except (OSError, ValueError) as error:  # a broken file is a reason, not a crash
        return f"could not read submission: {error}"


def summarize(labels: pd.DataFrame, meta: Meta) -> dict:
    """Aggregates for the CLI: label counts and a possession count (maximal H/A runs per half)."""
    codes = labels["label"].to_numpy()
    counts = {label: int((codes == label).sum()) for label in kit.pm.LABELS}
    lengths: list[int] = []
    for _, first, last in meta.half_blocks:
        block = codes[first : last + 1]
        start = 0
        for i in range(1, len(block) + 1):
            if i == len(block) or block[i] != block[start]:
                if block[start] in ("H", "A"):
                    lengths.append(i - start)
                start = i
    mean_seconds = float(np.mean(lengths)) / meta.fps if lengths else 0.0
    return {"counts": counts, "possessions": len(lengths), "mean_possession_s": mean_seconds}
