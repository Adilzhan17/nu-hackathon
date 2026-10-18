"""Stable entrypoint for the team's existing annotation/review workspace."""
from ._legacy import enable_legacy_modules


def main():
    enable_legacy_modules()
    from ml.annotator import main as serve
    serve()


if __name__ == '__main__':
    main()
