"""Legacy entry point for the download runner.

This script remains in place for backwards compatibility and now delegates to the
frameworks/drivers CLI implementation.
"""

from Training.frameworks_drivers.cli.download_runner_cli import main


if __name__ == "__main__":  # pragma: no cover - thin wrapper only used as script
    raise SystemExit(main())
