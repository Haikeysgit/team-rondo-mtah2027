import pandas as pd
import pytest

from rondo.labeltool.expand import expand_transitions


def _marks(rows):
    """rows: (frame, label)."""
    return pd.DataFrame({"frame": [r[0] for r in rows], "label": [r[1] for r in rows]})


def test_expand_three_marks_over_a_window():
    out = expand_transitions(_marks([(10, "H"), (20, "D"), (25, "A")]), 10, 30, n_frames=50)
    assert len(out) == 50
    assert (out[10:20] == "H").all() and (out[20:25] == "D").all() and (out[25:31] == "A").all()
    assert (out[:10] == "X").all() and (out[31:] == "X").all()
    assert (out[10:20] == "H").sum() == 10
    assert (out[20:25] == "D").sum() == 5
    assert (out[25:31] == "A").sum() == 6


def test_expand_single_mark_fills_the_whole_window():
    out = expand_transitions(_marks([(5, "H")]), 5, 9, n_frames=20)
    assert (out[5:10] == "H").all()
    assert (out[:5] == "X").all() and (out[10:] == "X").all()


def test_expand_first_mark_must_equal_start():
    with pytest.raises(ValueError, match="window start"):
        expand_transitions(_marks([(11, "H")]), 10, 20, n_frames=50)


def test_expand_marks_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="increasing"):
        expand_transitions(_marks([(10, "H"), (10, "A")]), 10, 20, n_frames=50)
    with pytest.raises(ValueError, match="increasing"):
        expand_transitions(_marks([(10, "H"), (9, "A")]), 10, 20, n_frames=50)


def test_expand_marks_must_not_go_past_the_window_end():
    with pytest.raises(ValueError, match="window end"):
        expand_transitions(_marks([(10, "H"), (21, "A")]), 10, 20, n_frames=50)


def test_expand_rejects_a_bad_label():
    with pytest.raises(ValueError, match="H.*A.*D"):
        expand_transitions(_marks([(10, "X")]), 10, 20, n_frames=50)


def test_expand_rejects_start_after_end():
    with pytest.raises(ValueError, match="must not be after"):
        expand_transitions(_marks([(10, "H")]), 10, 5, n_frames=50)


def test_expand_rejects_end_past_n_frames():
    with pytest.raises(ValueError, match="n_frames"):
        expand_transitions(_marks([(10, "H")]), 10, 50, n_frames=50)


def test_expand_rejects_empty_marks():
    with pytest.raises(ValueError, match="at least one row"):
        expand_transitions(_marks([]), 10, 20, n_frames=50)


def test_expand_output_shape():
    out = expand_transitions(_marks([(0, "H")]), 0, 0, n_frames=5)
    assert out.name == "label" and out.index.name == "frame"
    assert list(out.index) == [0, 1, 2, 3, 4]
