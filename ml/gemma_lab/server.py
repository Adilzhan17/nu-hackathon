"""Compatibility entrypoint; implementation lives in forte_vision.interfaces.analysis_http."""
import sys
from pathlib import Path

# Historical script entrypoints may start with only ml/ on sys.path.
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)
from forte_vision.interfaces import analysis_http as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
