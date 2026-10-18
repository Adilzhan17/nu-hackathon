"""Stable jury-service entrypoint; preserves the existing authenticated contract."""
from ._legacy import enable_legacy_modules


def main():
    enable_legacy_modules()
    from ml.serve_jury import main as serve
    serve()


if __name__ == '__main__':
    main()
