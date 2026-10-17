import numpy as np
import pandas as pd
import pytest

from rondo import kit
from rondo.contracts import ContractError
from rondo.eval import harness, windows
from tests.synthetic import full_length

# Possessions long enough that a one-frame shift keeps every IoU far above 0.7, and short enough
# that no single possession covers 30% of the window (so a constant guess matches none).
SPELLS = [("H", 300), ("D", 50), ("A", 400), ("H", 250), ("D", 60), ("A", 350)]
PUBLIC_GT = "BAR-ATH_possession_gt_public.csv"
BASELINE = kit.KIT_DIR / "evaluation" / "results" / "baseline_smooth50.csv"


def window(start: int, spells=SPELLS, n: int = 6000) -> pd.Series:
    """Truth series of length n: X everywhere except the spells laid end to end from start."""
    labels = np.full(n, "X", dtype=object)
    at = start
    for label, length in spells:
        labels[at : at + length] = label
        at += length
    return pd.Series(labels, index=pd.RangeIndex(n, name="frame"), name="label")


def late_by_one(truth: pd.Series) -> pd.Series:
    pred = truth.copy()
    pred.iloc[1:] = truth.iloc[:-1].to_numpy()
    return pred


def write_labels(path, labels: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": labels.index, "label": labels.to_numpy()}).to_csv(path, index=False)


# --------------------------------------------------------------------------- score


def test_perfect_prediction_scores_one():
    truth = window(1000)
    out = harness.score(truth, truth)
    assert out["spell_f1_mean"] == 1.0 and out["macro_f1"] == 1.0
    assert out["n_gt_spells"] == out["n_pred_spells"] == 4


def test_constant_guess_scores_zero():
    truth = window(1000)
    pred = pd.Series("H", index=truth.index, name="label")
    assert harness.score(pred, truth)["spell_f1_mean"] == 0.0


def test_one_frame_late_copy_scores_one():
    truth = window(1000)
    assert harness.score(late_by_one(truth), truth)["spell_f1_mean"] == 1.0


def test_score_works_on_a_full_length_match():
    truth = full_length(window(1000, n=3000).reset_index(drop=True))
    out = harness.score(truth, truth)
    assert out["spell_f1_mean"] == 1.0 and out["n_graded"] == sum(n for _, n in SPELLS)


def test_score_reports_the_documented_keys_and_no_interval_without_bootstrap():
    out = harness.score(window(1000), window(1000))
    for key in (
        "spell_f1_mean",
        "macro_f1",
        "spell_bootstrap_low",
        "spell_bootstrap_high",
        "n_gt_spells",
        "n_pred_spells",
        "mean_spell_duration_s",
        "gt_mean_spell_duration_s",
        "n_graded",
    ):
        assert key in out
    assert out["spell_bootstrap_low"] is None and out["spell_bootstrap_high"] is None


def test_bootstrap_interval_brackets_the_score_and_is_deterministic():
    truth = window(1000, spells=SPELLS * 3, n=8000)
    pred = truth.copy()
    pred.iloc[1500:1600] = "H"  # split the first A possession in two
    first = harness.score(pred, truth, bootstrap=200)
    assert first["spell_bootstrap_low"] <= first["spell_f1_mean"] <= first["spell_bootstrap_high"]
    assert harness.score(pred, truth, bootstrap=200) == first


def test_score_rejects_misaligned_series():
    truth = window(1000)
    with pytest.raises(ValueError, match="frames"):
        harness.score(truth.iloc[:-1], truth)
    shifted = truth.copy()
    shifted.index = shifted.index + 1
    with pytest.raises(ValueError, match="frames"):
        harness.score(shifted, truth)


def test_score_rejects_unknown_labels():
    truth = window(1000)
    pred = truth.copy()
    pred.iloc[5] = "Z"
    with pytest.raises(ContractError):
        harness.score(pred, truth)


def test_score_matches_the_official_scorer_on_the_kit_baseline():
    # Golden numbers printed by vendor/mtah-kit/evaluation/run_evaluation.py --bootstrap 1000.
    truth = windows.load_windows(None)["public"]
    pred = windows.read_labels(BASELINE, n_frames=len(truth))
    out = harness.score(pred, truth, bootstrap=1000)
    assert round(out["spell_f1_mean"], 4) == 0.3256
    assert round(out["macro_f1"], 4) == 0.6207
    assert round(out["spell_bootstrap_low"], 4) == 0.2140
    assert round(out["spell_bootstrap_high"], 4) == 0.4465
    assert (out["n_gt_spells"], out["n_pred_spells"], out["n_graded"]) == (17, 26, 7500)


# --------------------------------------------------------------------------- score_all


def test_score_all_has_one_row_per_window_plus_combined():
    wins = {"w1": window(500), "w2": window(3500)}
    table = harness.score_all(wins["w1"].where(wins["w1"] != "X", wins["w2"]), wins)
    assert table["window"].tolist() == ["w1", "w2", "combined"]
    assert (table["spell_f1_mean"] == 1.0).all()


def test_score_all_combined_counts_the_union_of_graded_frames():
    wins = {"w1": window(500), "w2": window(3500)}
    pred = pd.Series("H", index=wins["w1"].index, name="label")
    table = harness.score_all(pred, wins).set_index("window")
    per_window = sum(n for _, n in SPELLS)
    assert table.loc["w1", "n_graded"] == table.loc["w2", "n_graded"] == per_window
    assert table.loc["combined", "n_graded"] == 2 * per_window
    assert table.loc["combined", "n_gt_spells"] == 8


def test_score_all_rejects_windows_that_disagree_on_a_frame():
    w1 = window(500)
    w2 = window(600, spells=[("A", 300)])
    with pytest.raises(ValueError, match="w1.*w2"):
        harness.score_all(w1, {"w1": w1, "w2": w2})


def test_score_all_accepts_windows_that_agree_where_they_overlap():
    w1 = window(500)
    w2 = window(500, spells=SPELLS[:2])
    table = harness.score_all(w1, {"w1": w1, "w2": w2}).set_index("window")
    assert table.loc["combined", "n_graded"] == table.loc["w1", "n_graded"]


# --------------------------------------------------------------------------- windows


def test_load_windows_without_hand_labels_gives_the_public_window(tmp_path):
    wins = windows.load_windows(tmp_path)
    assert list(wins) == ["public"]
    public = wins["public"]
    graded = public[public != "X"]
    assert len(public) == 199319 and len(graded) == 7500
    assert graded.index.min() == 11185 and graded.index.max() == 18684


def test_load_windows_adds_hand_labelled_files_in_name_order(tmp_path):
    n = 199319
    for name, start in (("w2", 150000), ("w1", 30000)):
        labels = pd.Series("X", index=pd.RangeIndex(n, name="frame"), name="label")
        labels.iloc[start : start + 500] = "H"
        write_labels(tmp_path / "labels" / f"{name}.csv", labels)
    wins = windows.load_windows(tmp_path)
    assert list(wins) == ["public", "w1", "w2"]
    assert (wins["w1"] == "H").sum() == 500 and wins["w1"].iloc[30000] == "H"


def test_load_windows_names_the_bad_file(tmp_path):
    write_labels(tmp_path / "labels" / "w1.csv", window(10, n=100))
    with pytest.raises(ValueError, match="w1.csv"):
        windows.load_windows(tmp_path)


def test_load_windows_rejects_a_window_with_no_graded_frames(tmp_path):
    labels = pd.Series("X", index=pd.RangeIndex(199319, name="frame"), name="label")
    write_labels(tmp_path / "labels" / "empty.csv", labels)
    with pytest.raises(ValueError, match="empty.csv"):
        windows.load_windows(tmp_path)


def test_load_windows_reserves_the_public_name(tmp_path):
    public = windows.load_windows(None)["public"]
    write_labels(tmp_path / "labels" / "public.csv", public)
    with pytest.raises(ValueError, match="public"):
        windows.load_windows(tmp_path)


def test_read_labels_normalises_case_and_whitespace(tmp_path):
    path = tmp_path / "pred.csv"
    path.write_text("frame,label\n0, h\n1,a \n2,D\n3,x\n")
    assert windows.read_labels(path, n_frames=4).tolist() == ["H", "A", "D", "X"]


# --------------------------------------------------------------------------- write_result


def test_write_result_follows_the_results_template(tmp_path):
    wins = {"public": window(1000)}
    table = harness.score_all(late_by_one(wins["public"]), wins, bootstrap=100)
    path = tmp_path / "2026-10-17-kiki-official-baselines.md"
    harness.write_result(path, "Kiki", "rondo score x.csv --bootstrap 100", table, "Looks right.")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# official baselines\n")
    assert "- Date: 2026-10-17" in text and "- Owner: Kiki" in text
    assert "- Command: `rondo score x.csv --bootstrap 100`" in text
    assert "- Commit: `" in text
    assert "| window | spellF1 | 90% interval | macroF1 | possessions (ours / true) |" in text
    assert "| public | 1.0000 | 1.0000–1.0000 | " in text and "| 4 / 4 |" in text
    assert "| combined |" in text
    assert text.rstrip().endswith("Looks right.")


def test_write_result_marks_a_missing_interval(tmp_path):
    wins = {"public": window(1000)}
    table = harness.score_all(wins["public"], wins)
    path = tmp_path / "2026-10-17-kiki-smoke.md"
    harness.write_result(path, "Kiki", "rondo score x.csv", table, "")
    assert "| public | 1.0000 | — |" in path.read_text(encoding="utf-8")


def test_write_result_marks_a_window_without_possessions(tmp_path):
    wins = {"public": window(1000), "dead": window(3000, spells=[("D", 500)])}
    table = harness.score_all(window(1000), wins, bootstrap=50)
    path = tmp_path / "2026-10-17-kiki-dead.md"
    harness.write_result(path, "Kiki", "rondo score x.csv", table, "")
    text = path.read_text(encoding="utf-8")
    assert "| dead | 0.0000 | — |" in text and "nan" not in text
