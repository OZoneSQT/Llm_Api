from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List


def _spinner(stop_event: threading.Event) -> None:
    for ch in itertools.cycle("|/-\\"):
        if stop_event.is_set():
            break
        print(f"\rRunning... {ch}", end="", flush=True)
        time.sleep(0.15)
    print("\r", end="", flush=True)


def run_in_acaonda(script: Path, script_args: List[str]) -> int:
    """Run a Python script inside the `acaonda` conda environment using `conda run`.

    Returns the subprocess return code.
    """
    # Ensure `conda` is available on PATH
    if shutil.which("conda") is None:
        print("Error: `conda` not found on PATH. Please install or add conda to PATH.", file=sys.stderr)
        return 2

    # Build command and run with a live spinner in a background thread
    cmd = ["conda", "run", "-n", "acaonda", "python", str(script)] + script_args

    stop_event = threading.Event()
    t = threading.Thread(target=_spinner, args=(stop_event,), daemon=True)
    t.start()

    try:
        proc = subprocess.run(cmd)
        rc = proc.returncode if proc is not None else 1
    finally:
        stop_event.set()
        t.join()

    return rc


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Python script inside the `acaonda` conda environment.")
    parser.add_argument("script", type=Path, help="Python script to run")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the target script")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    script = args.script
    script_args: List[str] = args.script_args or []

    if not script.exists():
        print(f"Warning: script '{script}' does not exist; the wrapper will still attempt to run it.")

    rc = run_in_acaonda(script=script, script_args=script_args)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
