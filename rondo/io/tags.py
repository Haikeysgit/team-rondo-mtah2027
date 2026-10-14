"""Read the Smart Tagging (SportsCode) XML into the C1 ``tags`` table."""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

FPS = 25

# Words that many clubs share or that carry no identity ("FC", "Athletic", ...).
_GENERIC = {
    "fc", "cf", "club", "sd", "cd", "ud", "rcd", "real", "athletic", "atletic",
    "atlético", "deportivo", "de", "the",
}  # fmt: skip
_HALVES = {"1st half": 1, "2nd half": 2}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[^\W\d_]+", text.casefold())
    return {w for w in words if w not in _GENERIC and len(w) >= 4}


def map_tag_team(raw: str | None, home_name: str) -> str | None:
    """``"H"`` if the tag's team shares a distinctive word with the home club, ``"A"`` for any
    other value, ``None`` for missing / empty / ``N/A``."""
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text or text.casefold() == "n/a":
        return None
    return "H" if _tokens(text) & _tokens(home_name) else "A"


def read_tags(xml_path: Path, home_name: str) -> pd.DataFrame:
    """Tags per C1: frames are ``round(seconds * 25)``, ``end_frame`` exclusive, code upper case,
    ``half`` from the ``Half`` label (NA when absent), team null for ``N/A``."""
    root = ElementTree.parse(xml_path).getroot()
    rows = []
    for inst in root.iter("instance"):
        labels: dict[str, str] = {}
        for label in inst.findall("label"):
            group = (label.findtext("group") or "").strip()
            if group and group not in labels:
                labels[group] = (label.findtext("text") or "").strip()
        rows.append(
            (
                round(float(inst.findtext("start")) * FPS),
                round(float(inst.findtext("end")) * FPS),
                (inst.findtext("code") or "").strip().upper(),
                map_tag_team(labels.get("Team"), home_name),
                labels.get("Type") or None,
                _HALVES.get(labels.get("Half", "").casefold()),
            )
        )
    df = pd.DataFrame(rows, columns=["start_frame", "end_frame", "code", "team", "type", "half"])
    df = df.astype(
        {
            "start_frame": "int64",
            "end_frame": "int64",
            "code": "str",
            "team": "str",
            "type": "str",
            "half": "Int8",
        }
    )
    return df.sort_values(["start_frame", "end_frame", "code"], kind="stable", ignore_index=True)
