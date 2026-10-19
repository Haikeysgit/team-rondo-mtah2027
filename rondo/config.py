"""Runtime configuration: where the competition data lives."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
LAPTOP_DATA_DIR = Path(r"C:\rondo-data")  # the team's laptops keep the bundle outside the repo


def data_dir() -> Path:
    r"""The data bundle folder.

    ``MTAH_DATA_DIR`` (environment or ``.env``) if set; else ``C:\rondo-data`` if that folder
    exists; else ``<repo root>/data``, the official kit's convention.
    """
    value = os.environ.get("MTAH_DATA_DIR", "").strip()
    if value:
        return Path(value)
    if LAPTOP_DATA_DIR.is_dir():
        return LAPTOP_DATA_DIR
    return REPO_ROOT / "data"
