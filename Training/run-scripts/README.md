Run-scripts
===========

This folder contains cross-platform launchers to run the Training CLIs and tests. See the top-level `README.md` for overall setup and architectural guidance.

- `run-tests.ps1` / `run-tests.cmd`: Run the unit test suite under `Training/tests`.
  - Use `--include-smoke` with the PowerShell script to include smoke tests (or run `run-smoke-tests.*`).
- `run-smoke-tests.ps1` / `run-smoke-tests.cmd`: Run the test suite with smoke tests enabled by setting `RUN_SMOKE_TESTS=1` for the process.

Examples
--------
PowerShell (recommended):

```powershell
# Run unit tests (non-smoke)
C:\GitHub\Llm_Api\Training\run-scripts\run-tests.ps1

# Run smoke tests too (sets RUN_SMOKE_TESTS for the process)
$env:RUN_SMOKE_TESTS = 1
C:\GitHub\Llm_Api\Training\run-scripts\run-smoke-tests.ps1
```

Command Prompt:

```
C:\GitHub\Llm_Api\Training\run-scripts\run-tests.cmd
C:\GitHub\Llm_Api\Training\run-scripts\run-smoke-tests.cmd
```

If your environment uses a specific Python executable, set the `PYTHON` environment variable to the full path before running these scripts.

Notes
-----
- Smoke tests often require larger runtime dependencies (e.g., `transformers`) and cached snapshots. The test runner will skip smoke test cases with an explanatory message if prerequisites are not met.
- Prefer creating a fresh virtual environment and installing `Training/requirements.txt` before running smoke tests to ensure deterministic behavior.
