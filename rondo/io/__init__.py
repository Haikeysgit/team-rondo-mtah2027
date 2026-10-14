"""Load the organiser's bundle into the C1 tables (docs/CONTRACTS.md)."""

from rondo.io.atd import read_atd_meta, read_atd_raw
from rondo.io.load import check_bundle, load_match
from rondo.io.tags import map_tag_team, read_tags

__all__ = [
    "check_bundle",
    "load_match",
    "map_tag_team",
    "read_atd_meta",
    "read_atd_raw",
    "read_tags",
]
