"""Tiny ATD / Smart Tagging / homography files written from code (never real data)."""

from pathlib import Path

META_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Metadata>
  <Teams>
    <Team id="T1"><Name>FC Barcelona</Name></Team>
    <Team id="T2"><Name>Athletic Club</Name></Team>
  </Teams>
  <Players>
    <Player teamId="T1"/>
    <Player teamId="T2"/>
  </Players>
</Metadata>
"""


def instance(start: float, end: float, code: str, labels: list[tuple[str, str]]) -> str:
    inner = "".join(f"<label><group>{g}</group><text>{t}</text></label>" for g, t in labels)
    return f"<instance><start>{start}</start><end>{end}</end><code>{code}</code>{inner}</instance>"


TAGS_XML = (
    "<file><ALL_INSTANCES>"
    + instance(
        0.2, 0.8, "Set Pieces", [("Team", "Barcelona"), ("Half", "1st Half"), ("Type", "Throw-in")]
    )
    + instance(
        1.0, 1.4, "Set Pieces", [("Team", "Atletic Club"), ("Half", "1st Half"), ("Type", "Corner")]
    )
    + instance(1.4, 1.8, "Goal", [("Team", "N/A"), ("Half", "2nd Half")])
    + "</ALL_INSTANCES></file>"
)


def write_raw_text(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "raw.txt"
    path.write_text(text, encoding="utf-8")
    return path


def write_raw(tmp_path: Path, rows: list[tuple[int, str, str]]) -> Path:
    """rows: (frame_id, player groups joined by ';', ball group)."""
    return write_raw_text(tmp_path, "".join(f"{i}:{p}:{b}\n" for i, p, b in rows))


def write_bundle(tmp_path: Path, first_id: int = 1, n: int = 50, ball: str = "0.5,0.5,0") -> Path:
    """A tiny bundle in the kit's folder layout; returns the data dir.

    Frame ``k`` (id ``first_id + k``) has two players, the second one missing every 5th frame.
    """
    folder = tmp_path / "bundle" / "tactical"
    folder.mkdir(parents=True)
    (folder / "BAR-ATH_tactical_atd.xml").write_text(META_XML, encoding="utf-8")
    lines = []
    for k in range(n):
        second = "NaN,NaN,NaN" if k % 5 == 4 else f"0.{k % 10},0.7,2.0"
        lines.append(f"{first_id + k}:0.4,0.5,1.5;{second}:{ball}")
    (folder / "BAR-ATH_tactical_atd.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "BAR-ATH_tactical_smart_tagging.xml").write_text(TAGS_XML, encoding="utf-8")
    rows = ["frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid"]
    rows += [f"{k},1,0,0,0,1,0,0,0,1,0.1,{0 if k % 4 == 3 else 1}" for k in range(n)]
    (folder / "BAR-ATH_tactical_homography.csv").write_text("\n".join(rows) + "\n")
    return tmp_path / "bundle"
