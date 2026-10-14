"""Name -> component registries filled by the signal, emission and decoder packages (C2-C4)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from rondo.contracts import MatchData

SIGNALS: dict[str, Callable[[MatchData], pd.DataFrame]] = {}
EMISSIONS: dict[str, type] = {}
DECODERS: dict[str, Callable] = {}


def _register(table: dict, kind: str, name: str):
    def decorator(obj):
        if name in table:
            raise ValueError(f"{kind} '{name}' is already registered")
        table[name] = obj
        return obj

    return decorator


def register_signal(name: str):
    return _register(SIGNALS, "signal", name)


def register_emission(name: str):
    return _register(EMISSIONS, "emission", name)


def register_decoder(name: str):
    return _register(DECODERS, "decoder", name)
