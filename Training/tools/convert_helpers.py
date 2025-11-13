"""Detect common model formats and offer conversion helpers for GGUF/GGML/GPTQ artifacts.

The helper is intentionally conservative. It only executes converters when the user
passes ``--yes`` (with optional ``--dry-run`` previews) and otherwise prints concrete
commands the user can run manually. Supported tasks:

* Detect GGUF/GGML/GPTQ/safetensors artifacts under a directory.
* Run the official Transformers LLaMA GGML→HF converter when present in the repo or
  available via the installed ``transformers`` package.
* Suggest or execute ``auto_gptq`` conversions via ``python -m auto_gptq.convert``.
* Print llama.cpp conversion hints for manual workflows.
"""

from Training.frameworks_drivers.cli.convert_cli import main


if __name__ == '__main__':
    raise SystemExit(main())
