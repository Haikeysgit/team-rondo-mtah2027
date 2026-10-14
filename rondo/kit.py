"""Access to the official MTAH kit (pinned submodule in vendor/mtah-kit), used unchanged."""

import sys
from pathlib import Path

KIT_DIR = Path(__file__).resolve().parent.parent / "vendor" / "mtah-kit"

for _p in (KIT_DIR / "evaluation", KIT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import games
import possession_metrics as pm

__all__ = ["KIT_DIR", "games", "pm"]
