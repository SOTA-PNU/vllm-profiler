"""Module entry point for ``python -m perfetto_hetero_profiler``."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
