"""Run the configured signals -> emission -> decoder chain and finalise the labels."""

from __future__ import annotations

import importlib

import pandas as pd

from rondo import registry
from rondo.contracts import (
    MatchData,
    validate_emissions,
    validate_labels,
    validate_signals,
)
from rondo.post import finalize

COMPONENT_PACKAGES = ("rondo.signals", "rondo.emissions", "rondo.decoder")

# Where the default components come from, for the error message when one is not merged yet.
PROVIDED_BY = {
    "ball_nearest": "Task 11",
    "ball_motion": "Task 11",
    "ball_bounds": "Task 11",
    "team_motion": "Task 11",
    "tag_restarts": "Task 12",
    "tag_team_prior": "Task 12",
    "rules_v0": "Task 9",
    "viterbi": "Task 10",
}


def load_components() -> None:
    """Import the component packages so their registration decorators run.

    A package that does not exist yet is skipped; any other import error is real and propagates.
    """
    for package in COMPONENT_PACKAGES:
        try:
            importlib.import_module(package)
        except ModuleNotFoundError as error:
            if error.name != package:
                raise


def _check_registered(config: dict) -> None:
    wanted = [("signal", name, registry.SIGNALS) for name in config["signals"]] + [
        ("emission", config["emission"], registry.EMISSIONS),
        ("decoder", config["decoder"], registry.DECODERS),
    ]
    missing = []
    for kind, name, table in wanted:
        if name not in table:
            origin = PROVIDED_BY.get(name)
            missing.append(f"{kind} '{name}'" + (f" (provided by {origin})" if origin else ""))
    if missing:
        raise KeyError("components not registered: " + "; ".join(missing))


def run(match: MatchData, config: dict, labels: pd.Series | None = None) -> pd.DataFrame:
    """Labels for every frame of ``match`` (``frame, label``, ``H/A/D/X``).

    ``labels`` (optional, indexed by frame) makes the emission model ``fit`` before it predicts.
    """
    load_components()
    _check_registered(config)
    meta = match.meta

    merged: pd.DataFrame | None = None
    owner: dict[str, str] = {}
    for name in config["signals"]:
        signal = registry.SIGNALS[name](match)
        validate_signals(signal, meta)
        if merged is None:
            merged = signal[["frame"]].copy()
        for column in signal.columns:
            if column == "frame":
                continue
            if column in owner:
                raise ValueError(
                    f"signal column '{column}' is produced by both '{owner[column]}' and '{name}'"
                )
            owner[column] = name
        merged = merged.merge(signal, on="frame", how="left", validate="one_to_one")
    if merged is None:
        raise ValueError("config lists no signals")

    model = registry.EMISSIONS[config["emission"]](**config.get("emission_params", {}))
    if labels is not None:
        model.fit(merged, labels)
    emissions = model.predict(merged)
    validate_emissions(emissions, meta)

    decoder = registry.DECODERS[config["decoder"]]
    decoded = decoder(emissions, match.tags, meta, config.get("decoder_params", {}))
    validate_labels(decoded, meta, allow_x=False)
    return finalize(decoded, meta)
