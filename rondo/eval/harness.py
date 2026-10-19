"""Score predictions on every validation window with the official scorer, and log the result.

All scoring goes through ``possession_metrics.evaluate`` from the pinned kit; this module only
picks the figures we compare on and combines windows. Bootstrap intervals are the kit's 5-95
percentiles (a 90% interval), resampling whole possessions.
"""

from __future__ import annotations

import re
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

from rondo import kit
from rondo.contracts import labels_to_codes

COMBINED = "combined"
KEYS = (
    "spell_f1_mean",
    "spell_bootstrap_low",
    "spell_bootstrap_high",
    "macro_f1",
    "n_pred_spells",
    "n_gt_spells",
    "mean_spell_duration_s",
    "gt_mean_spell_duration_s",
    "n_graded",
)
RESULT_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})-[^-]+-(.+)$")


def score(pred: pd.Series, truth: pd.Series, bootstrap: int = 0) -> dict:
    """Official metrics for one prediction against one ground truth, both full-length labels.

    Returns the ``KEYS`` figures; the bootstrap bounds are ``None`` when ``bootstrap`` is 0.
    """
    if len(pred) != len(truth) or not pred.index.equals(truth.index):
        raise ValueError(
            f"prediction and truth must cover the same frames ({len(pred)} vs {len(truth)} rows)"
        )
    result = kit.pm.evaluate(labels_to_codes(truth), labels_to_codes(pred), bootstrap_n=bootstrap)
    return {key: result[key] for key in KEYS}


def combine(windows: dict[str, pd.Series]) -> pd.Series:
    """One truth series graded on the union of the windows' graded frames, ``X`` elsewhere.

    Windows may overlap only where they agree; a frame two windows label differently raises.
    """
    names = list(windows)
    combined = windows[names[0]].copy()
    for name in names[1:]:
        labels = windows[name]
        clash = (combined != "X") & (labels != "X") & (combined != labels)
        if clash.any():
            frame = int(clash.idxmax())
            owner = next(n for n in names if windows[n].loc[frame] == combined.loc[frame])
            raise ValueError(f"windows {owner} and {name} disagree at frame {frame}")
        combined = combined.where(combined != "X", labels)
    return combined


def score_all(pred: pd.Series, windows: dict[str, pd.Series], bootstrap: int = 0) -> pd.DataFrame:
    """One row per window, in the given order, then ``combined`` (all windows scored together)."""
    rows = [{"window": name, **score(pred, truth, bootstrap)} for name, truth in windows.items()]
    rows.append({"window": COMBINED, **score(pred, combine(windows), bootstrap)})
    return pd.DataFrame(rows, columns=["window", *KEYS])


def write_result(path: Path, owner: str, command: str, table: pd.DataFrame, notes: str) -> None:
    """Write an experiment results file (Markdown, the team's template). Aggregates only.

    The date and topic come from the file name (``YYYY-MM-DD-name-topic.md``), so the same run
    writes the same file; the commit is the current ``HEAD``.
    """
    path = Path(path)
    match = RESULT_NAME.match(path.stem)
    day, topic = (match[1], match[2].replace("-", " ")) if match else (str(date.today()), path.stem)
    lines = [
        f"# {topic}",
        "",
        f"- Date: {day}",
        f"- Owner: {owner}",
        f"- Commit: `{_commit()}`",
        f"- Command: `{command}`",
        "",
        "| window | spellF1 | 90% interval | macroF1 | possessions (ours / true) |",
        "|---|---|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.window} | {row.spell_f1_mean:.4f} | {_interval(row)} | {row.macro_f1:.4f} "
            f"| {row.n_pred_spells} / {row.n_gt_spells} |"
        )
    lines += ["", "## Notes", "", notes.strip(), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _interval(row) -> str:
    low, high = row.spell_bootstrap_low, row.spell_bootstrap_high
    if pd.isna(low) or pd.isna(high):
        return "—"
    return f"{low:.4f}–{high:.4f}"


def _commit() -> str:
    """Short ``HEAD`` hash, flagged when the working tree has uncommitted changes."""
    root = Path(__file__).resolve().parent.parent.parent
    try:
        head = _git(root, "rev-parse", "--short", "HEAD")
        dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head} + uncommitted changes" if dirty else head


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return out.stdout.strip()
