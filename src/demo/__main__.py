"""Command-line entry point: ``python -m src.demo``.

Generates the pre-generated demo artifacts. See :mod:`src.demo.generate`.
"""

from __future__ import annotations

import sys

from src.demo.generate import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
