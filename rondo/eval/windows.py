"""Validation windows: the official public ground truth plus our hand-labelled files.

Every window is a full-length ``frame,label`` series (one row per video frame) that is ``X``
outside the frames it grades, so any window can be scored against a whole submission.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rondo import kit

PUBLIC = "public"
PUBLIC_GT_PATTERN = "*_possession_gt_public.csv"


def read_labels(path: Path, n_frames: int | None = None) -> pd.Series:
    """Read a ``frame,label`` CSV with the official reader and checks; labels upper-cased.

    ``n_frames=None`` takes the row count from the file itself. Raises ``ValueError`` naming the
    file when it is not a valid label table with frames ``0..n_frames-1``.
    """
    path = Path(path)
    try:
        table = kit.pm.read_label_csv(path)
    except Exception as error:  # unreadable file, missing columns
        raise ValueError(f"{path.name}: could not read ({error})") from error
    if n_frames is None:
        n_frames = len(table)
    reason = kit.pm.validate_submission(table, n_frames)
    if reason:
        raise ValueError(f"{path.name}: {reason}")
    labels = table["label"].astype(str).str.strip().str.upper().to_numpy(dtype=object)
    return pd.Series(labels, index=pd.RangeIndex(n_frames, name="frame"), name="label")


def public_gt_path() -> Path:
    """The public ground truth shipped with the kit (5 minutes of the first half)."""
    found = sorted((kit.KIT_DIR / "evaluation" / "gt").glob(PUBLIC_GT_PATTERN))
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected one {PUBLIC_GT_PATTERN} in the kit, found {len(found)}; "
            "run `git submodule update --init`"
        )
    return found[0]


def load_windows(data_dir: Path | None) -> dict[str, pd.Series]:
    """``public`` first, then every ``<data_dir>/labels/*.csv`` by file name (stem = window name).

    Hand-labelled files are full-length ``frame,label`` tables, ``X`` outside the window. With
    ``data_dir=None``, or no ``labels`` folder yet, only the public window is returned.
    """
    public = read_labels(public_gt_path())
    windows = {PUBLIC: public}
    labels_dir = Path(data_dir) / "labels" if data_dir is not None else None
    if labels_dir is None or not labels_dir.is_dir():
        return windows
    for path in sorted(labels_dir.glob("*.csv")):
        if path.stem == PUBLIC:
            raise ValueError(
                f"{path.name}: the name {PUBLIC!r} is reserved for the official window"
            )
        labels = read_labels(path, n_frames=len(public))
        if not (labels != "X").any():
            raise ValueError(f"{path.name}: no graded frames (every label is X)")
        windows[path.stem] = labels
    return windows
