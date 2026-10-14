"""Runtime configuration: where the competition data lives."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def data_dir() -> Path:
    """Return the data directory from MTAH_DATA_DIR (env or .env), default C:\rondo-data."""
    return Path(os.environ.get("MTAH_DATA_DIR", r"C:\rondo-data"))
