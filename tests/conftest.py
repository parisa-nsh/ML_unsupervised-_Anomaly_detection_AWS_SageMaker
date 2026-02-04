"""Pytest configuration: add scripts directory to path so tests can import from scripts."""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_scripts = _repo_root / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
