import numpy as np
import pytest

from rondo.io import check_bundle, read_atd_meta, read_atd_raw
from tests.io.fixtures import META_XML, write_bundle, write_raw, write_raw_text


def test_raw_alignment_and_missing(tmp_path):
    raw = write_raw(
        tmp_path,
        [
            (1, "0.5,0.5,1.0", "0.5,0.5,0"),
            (2, "NaN,NaN,NaN", "NaN,NaN,NaN"),
            (3, "0.6,0.5,1.0", "0.6,0.5,0"),
        ],
    )
    frames, players, rep = read_atd_raw(raw, np.array(["H"]), n_frames=3)
    assert frames["frame"].tolist() == [0, 1, 2]
    assert np.isnan(frames.loc[1, "ball_x"])
    assert players["frame"].tolist() == [0, 2]
    assert rep["contiguous"] and rep["first_id"] == 1


def test_malformed_lines_are_skipped_not_shifted(tmp_path):
    raw = write_raw_text(tmp_path, "1:0.5,0.5,1:0.5,0.5,0\ngarbage\n3:0.5,0.5,1:0.5,0.5,0\n\n")
    frames, _, rep = read_atd_raw(raw, np.array(["H"]), n_frames=3)
    assert rep["skipped_lines"] == 1 and not rep["contiguous"]
    assert np.isnan(frames.loc[1, "ball_x"]) and frames.loc[2, "ball_x"] == 0.5


def test_player_columns_and_order(tmp_path):
    raw = write_raw(tmp_path, [(1, "0.1,0.2,3.0;NaN,NaN,NaN;0.3,0.4,5.0", "0.5,0.5,0")])
    _, players, _ = read_atd_raw(raw, np.array(["H", "A", "A"]), n_frames=1)
    assert players["slot"].tolist() == [0, 2]
    assert players["team"].tolist() == ["H", "A"]
    assert str(players["slot"].dtype) == "int16" and str(players["frame"].dtype) == "int64"
    assert players["speed"].tolist() == [3.0, 5.0]


def test_out_of_range_and_empty_player_section(tmp_path):
    raw = write_raw_text(tmp_path, "1::0.5,0.5,0\n2:0.1,0.1,1:0.5,0.5,0\n9:0.1,0.1,1:0.5,0.5,0\n")
    frames, players, rep = read_atd_raw(raw, np.array(["H"]), n_frames=3)
    assert rep["skipped_lines"] == 1 and rep["n_records"] == 3
    assert players["frame"].tolist() == [1]
    assert frames["ball_x"].notna().tolist() == [True, True, False]


def test_ball_fields_counts_commas(tmp_path):
    raw = write_raw(tmp_path, [(1, "0.5,0.5,1.0", "0.5,0.5,0,Alive,H")])
    _, _, rep = read_atd_raw(raw, np.array(["H"]), n_frames=1)
    assert rep["ball_fields"] == 5


def test_more_groups_than_slots_is_skipped(tmp_path):
    raw = write_raw(tmp_path, [(1, "0.5,0.5,1.0;0.5,0.5,1.0", "0.5,0.5,0")])
    _, players, rep = read_atd_raw(raw, np.array(["H"]), n_frames=1)
    assert rep["skipped_lines"] == 1 and len(players) == 0


def test_meta_reads_slots_and_names(tmp_path):
    path = tmp_path / "m.xml"
    path.write_text(META_XML, encoding="utf-8")
    slot_team, home, away = read_atd_meta(path)
    assert slot_team.tolist() == ["H", "A"]
    assert (home, away) == ("FC Barcelona", "Athletic Club")


def test_meta_first_team_is_home_even_if_players_listed_away_first(tmp_path):
    path = tmp_path / "m.xml"
    path.write_text(
        '<M><Team id="a"><Name>X</Name></Team><Team id="b"><Name>Y</Name></Team>'
        '<Player teamId="b"/><Player teamId="a"/></M>'
    )
    assert read_atd_meta(path)[0].tolist() == ["A", "H"]


@pytest.mark.parametrize(
    "xml",
    [
        '<M><Team id="a"><Name>X</Name></Team><Player teamId="a"/></M>',
        '<M><Team id="a"><Name>X</Name></Team><Team id="b"><Name>Y</Name></Team></M>',
    ],
)
def test_meta_errors(tmp_path, xml):
    path = tmp_path / "m.xml"
    path.write_text(xml)
    with pytest.raises(ValueError):
        read_atd_meta(path)


def test_check_bundle_flags_misaligned_ids(tmp_path):
    bundle = write_bundle(tmp_path, first_id=2)
    rep = check_bundle(bundle)
    assert rep["aligned"] is False


def test_check_bundle_report(tmp_path):
    bundle = write_bundle(tmp_path, n=40, ball="0.5,0.5,0,Alive")
    rep = check_bundle(bundle, expected_records=40)
    assert rep["aligned"] is True
    assert rep["n_records"] == 40 and rep["n_tags"] == 3
    assert rep["ball_fields"] == 4
    assert rep["unknown_tag_codes"] == ["GOAL"]  # fixture says "Goal", the table has "GOALS"
    assert rep["homography_rows"] == 40 and rep["homography_valid_pct"] == 75.0
    assert rep["ball_present_graded_pct"] is None  # official half blocks lie beyond 40 frames
    assert all(type(v) in (int, float, bool, type(None), list) for v in rep.values())


def test_check_bundle_ball_present_on_graded(tmp_path):
    bundle = write_bundle(tmp_path, n=11200)
    rep = check_bundle(bundle, expected_records=11200)
    assert rep["ball_present_graded_pct"] == 100.0


def test_check_bundle_missing_files_named(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        check_bundle(tmp_path)
    message = str(err.value)
    assert "atd.xml" in message and "atd.txt" in message and "homography.csv" in message
    assert str(tmp_path) in message
