"""Explicit boundary for annotation/legacy API modules not yet migrated."""
import sys
from pathlib import Path


def enable_legacy_modules():
    directory = str(Path(__file__).resolve().parents[2] / 'ml')
    if directory not in sys.path:
        sys.path.insert(0, directory)
