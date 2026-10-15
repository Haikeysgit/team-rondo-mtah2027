"""Signals (C2): importing the package registers every module in it.

Each submodule is imported in sorted name order, so a module dropped into this directory
registers its signals with no edit here.
"""

import importlib
import pkgutil

for _module in sorted(m.name for m in pkgutil.iter_modules(__path__)):
    importlib.import_module(f"{__name__}.{_module}")
