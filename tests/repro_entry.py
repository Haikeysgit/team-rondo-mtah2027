"""``python -m tests.repro_entry run ...``: ``rondo run`` with the test stub components registered.

``rondo verify-repro`` starts each run in a fresh process, which would not see components that
the tests register in-process; its test points it at this module instead of ``rondo.cli``.
"""

import tests.conftest  # noqa: F401  (registers the stub components)
from rondo.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
