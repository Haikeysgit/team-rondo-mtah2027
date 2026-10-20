"""Error moments: predicted transitions worth a human looking at the video.

Built entirely from the official transition matching -- ``kit.pm.evaluate(...)["transitions"][tol]``
and its ``pairs``/``missed``/``spurious`` lists, the same lists ``run_evaluation.py`` writes to
``reports/<name>_transitions.csv``. We never re-implement that matching here.
"""

from __future__ import annotations

import pandas as pd

from rondo import kit
from rondo.contracts import labels_to_codes

#: a matched pair (within ``tol``) off by more than this many frames is reported as "late"
#: rather than counted as a clean match; this is the scorer's own tight tolerance (`TAUS[0]`)
LATE_TOLERANCE = kit.pm.TAUS[0]

#: the tactical video's frame rate, used only to turn a frame into an ``mm:ss`` timestamp
FPS = kit.games.GAMES[kit.games.GAME]["fps"]

COLUMNS = ["frame", "status", "gt_from", "gt_to", "pred_from", "pred_to", "latency"]


def error_moments(pred: pd.Series, truth: pd.Series, tol: int = 25) -> pd.DataFrame:
    """Transitions worth reviewing: ``missed``, ``spurious``, or ``late``, sorted by frame.

    ``pred`` and ``truth`` are full-length ``H``/``A``/``D``/``X`` label series indexed by frame
    (the shape ``rondo.eval.harness.score`` takes). A transition matched within ``tol`` frames but
    off by more than ``LATE_TOLERANCE`` is ``late``; closer matches are not errors and are dropped.

    ``frame`` is the ground-truth transition's frame for ``missed`` and ``late`` rows, or the
    prediction's frame for ``spurious`` rows (there is no ground-truth transition to anchor to).
    ``latency`` is ``pred_frame - gt_frame`` (positive: the prediction came late); ``None`` for
    ``missed`` and ``spurious`` rows, which have no pair.
    """
    result = kit.pm.evaluate(labels_to_codes(truth), labels_to_codes(pred), taus=(tol,))
    transitions = result["transitions"][tol]

    rows = []
    for gt, prediction in transitions["pairs"]:
        latency = prediction[0] - gt[0]
        if abs(latency) > LATE_TOLERANCE:
            rows.append(_row(gt[0], "late", gt, prediction, latency))
    for gt in transitions["missed"]:
        rows.append(_row(gt[0], "missed", gt, None, None))
    for prediction in transitions["spurious"]:
        rows.append(_row(prediction[0], "spurious", None, prediction, None))

    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values("frame", kind="stable").reset_index(drop=True)


def format_timestamp(frame: int, fps: float = FPS) -> str:
    """``frame`` as ``mm:ss`` of video time, for printing to a human rather than a seek target."""
    total_seconds = int(frame / fps)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _row(
    frame: int, status: str, gt: tuple | None, pred: tuple | None, latency: int | None
) -> dict:
    labels = kit.pm.LABELS
    return {
        "frame": int(frame),
        "status": status,
        "gt_from": labels[gt[1]] if gt is not None else None,
        "gt_to": labels[gt[2]] if gt is not None else None,
        "pred_from": labels[pred[1]] if pred is not None else None,
        "pred_to": labels[pred[2]] if pred is not None else None,
        "latency": latency,
    }
