#!/usr/bin/env python
"""Compatibility wrapper for the package CLI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_ran_kpi_forecasting.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
