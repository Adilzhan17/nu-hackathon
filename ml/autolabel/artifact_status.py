"""Small offline completion probe for bounded resumable shell workflows."""
import json
from pathlib import Path
import sys

if __name__=='__main__':
    try:
        data=json.loads(Path(sys.argv[1]).read_text())
        complete=data.get('completed') is True
    except (OSError,ValueError,IndexError):complete=False
    raise SystemExit(0 if complete else 1)
