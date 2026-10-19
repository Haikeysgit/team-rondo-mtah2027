"""The emission interface (C3) and a helper to build typed parameters from the config."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Protocol, Self, TypeVar

import pandas as pd

P = TypeVar("P")


class Emission(Protocol):
    """Per-frame label probabilities. Built as ``Cls(**emission_params)`` by the pipeline."""

    def fit(self, signals: pd.DataFrame, labels: pd.Series) -> Self:
        """Learn from labelled frames; called only when the pipeline is given labels."""
        ...

    def predict(self, signals: pd.DataFrame) -> pd.DataFrame:
        """``frame, p_H, p_A, p_D`` with each row summing to 1."""
        ...


def build_params(cls: type[P], params: dict, name: str) -> P:
    """``cls(**params)`` where every value must be a finite number >= 0.

    An unknown parameter name is an error rather than silently ignored, so a typo in
    ``configs/default.toml`` cannot leave a default in place unnoticed.
    """
    known = {field.name for field in fields(cls)}
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(
            f"unknown {name} parameter(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}"
        )
    values = {key: float(value) for key, value in params.items()}
    for key, value in values.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} {key} must be finite and >= 0, got {value}")
    return cls(**values)


def require_columns(signals: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    """Raise ``KeyError`` naming every signal column the emission needs but did not get."""
    missing = sorted(set(columns) - set(signals.columns))
    if missing:
        raise KeyError(f"{name} needs signal column(s) {', '.join(missing)}")
