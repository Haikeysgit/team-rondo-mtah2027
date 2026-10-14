import pandas as pd
import pytest

from rondo.contracts import Meta, validate_tags
from rondo.io import map_tag_team, read_tags
from tests.io.fixtures import TAGS_XML


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Barcelona", "H"),
        ("Atletic Club", "A"),
        ("Athletic Club", "A"),
        ("N/A", None),
        ("", None),
        (None, None),
        ("FC Barcelona", "H"),
        ("barcelona", "H"),
        ("  n/a ", None),
        (pd.NA, None),
        (float("nan"), None),
    ],
)
def test_map_tag_team(raw, expected):
    assert map_tag_team(raw, "FC Barcelona") == expected


def test_read_tags(tmp_path):
    path = tmp_path / "t.xml"
    path.write_text(TAGS_XML, encoding="utf-8")
    tags = read_tags(path, "FC Barcelona")
    assert tags["start_frame"].tolist() == [5, 25, 35]
    assert tags["end_frame"].tolist() == [20, 35, 45]
    assert tags["code"].tolist() == ["SET PIECES", "SET PIECES", "GOAL"]
    assert tags["team"].iloc[:2].tolist() == ["H", "A"] and pd.isna(tags["team"].iloc[2])
    assert tags["type"].iloc[0] == "Throw-in" and pd.isna(tags["type"].iloc[2])
    assert tags["half"].tolist() == [1, 1, 2] and str(tags["half"].dtype) == "Int8"
    meta = Meta(50, 25, (105.3, 68.0), [], "FC Barcelona", "Athletic Club")
    validate_tags(tags, meta)
