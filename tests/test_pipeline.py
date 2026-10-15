import importlib

import pandas as pd
import pytest

from rondo import pipeline, registry
from rondo.pipeline import run
from tests.synthetic import Step, make_match

STEPS = [Step("dribble", "H", 3), Step("interception", "A", 1), Step("dribble", "A", 3)]


def test_pipeline_is_deterministic(synthetic_config):
    m, _ = make_match(STEPS)
    assert run(m, synthetic_config).equals(run(m, synthetic_config))


def test_pipeline_output_is_a_full_submission_table(synthetic_config):
    m, _ = make_match(STEPS)
    out = run(m, synthetic_config)
    assert len(out) == m.meta.n_frames and set(out["label"]) <= {"H", "A", "D", "X"}
    graded = m.frames["graded"].to_numpy()
    assert (out["label"].to_numpy()[~graded] == "X").all()
    assert (out["label"].to_numpy()[graded] != "X").all()


def test_pipeline_fits_only_when_labels_given(synthetic_config, monkeypatch):
    m, truth = make_match(STEPS)
    seen = []
    cls = registry.EMISSIONS["stub_emission"]
    original = cls.fit

    def spy(self, signals, labels):
        seen.append(len(labels))
        return original(self, signals, labels)

    monkeypatch.setattr(cls, "fit", spy)
    run(m, synthetic_config)
    assert seen == []
    run(m, synthetic_config, labels=truth)
    assert seen == [len(truth)]


def test_pipeline_passes_emission_params(synthetic_config):
    m, _ = make_match(STEPS)
    cfg = {**synthetic_config, "emission_params": {"confidence": 0.9}}
    assert len(run(m, cfg)) == m.meta.n_frames


def test_missing_components_listed_with_tasks(synthetic_config, monkeypatch):
    # Pretend the not-yet-merged tasks are absent, whatever has been merged since.
    for name in ("ball_nearest", "tag_restarts"):
        monkeypatch.delitem(registry.SIGNALS, name, raising=False)
    for name in ("rules_v0",):
        monkeypatch.delitem(registry.EMISSIONS, name, raising=False)
    for name in ("viterbi",):
        monkeypatch.delitem(registry.DECODERS, name, raising=False)
    m, _ = make_match(STEPS)
    cfg = {
        **synthetic_config,
        "signals": ["stub_nearest", "ball_nearest", "tag_restarts", "nope"],
        "emission": "rules_v0",
        "decoder": "viterbi",
    }
    with pytest.raises(KeyError) as err:
        run(m, cfg)
    text = str(err.value)
    for needle in ("ball_nearest", "Task 11", "tag_restarts", "Task 12", "rules_v0", "Task 9"):
        assert needle in text
    assert "viterbi" in text and "Task 10" in text and "nope" in text


def test_signal_column_clash_raises(synthetic_config, monkeypatch):
    monkeypatch.setitem(registry.SIGNALS, "stub_clash", registry.SIGNALS["stub_nearest"])
    m, _ = make_match(STEPS)
    cfg = {**synthetic_config, "signals": ["stub_nearest", "stub_clash"]}
    with pytest.raises(ValueError, match="stub_nearest") as err:
        run(m, cfg)
    assert "stub_clash" in str(err.value)


def test_duplicate_registration_raises():
    with pytest.raises(ValueError):
        registry.register_signal("stub_nearest")(lambda m: m)


def test_load_components_skips_missing_packages_only(monkeypatch):
    pipeline.load_components()  # missing packages are skipped, present ones import cleanly
    real = importlib.import_module

    def broken(name, *args, **kwargs):
        if name == "rondo.signals":
            raise ModuleNotFoundError("No module named 'numpyy'", name="numpyy")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(pipeline.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError):
        pipeline.load_components()


def test_signal_validation_applied(synthetic_config, monkeypatch):
    monkeypatch.setitem(
        registry.SIGNALS, "stub_short", lambda m: pd.DataFrame({"frame": m.frames["frame"][:5]})
    )
    m, _ = make_match(STEPS)
    with pytest.raises(ValueError):
        run(m, {**synthetic_config, "signals": ["stub_short"]})
