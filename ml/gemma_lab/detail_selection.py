"""Compatibility entrypoint; implementation lives in forte_vision.application.detail_selection."""
import sys
from pathlib import Path

# Historical script entrypoints may start with only ml/ on sys.path.
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)
from forte_vision.application import detail_selection as _implementation

sys.modules[__name__] = _implementation
