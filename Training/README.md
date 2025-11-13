# Training helpers

This folder contains small utilities and harnesses to run quick finetunes and to safely load community model checkpoints.

Environment setup
-----------------
1. Create or update the Anaconda environment using the bundled spec:

	```powershell
	powershell -NoProfile -ExecutionPolicy Bypass -File Training\scripts\setup-conda-env.ps1
	```

2. Activate the environment before running tools or tests:

	```powershell
	conda activate llm_api
	```

3. Run the maintained pytest suite at any time via:

	```powershell
	powershell -NoProfile -ExecutionPolicy Bypass -File Training\scripts\run-tests.ps1
	```

	To automate the full bootstrap (provision the env and immediately execute tests) pass `-RunTests` when invoking the setup script:

	```powershell
	powershell -NoProfile -ExecutionPolicy Bypass -File Training\scripts\setup-conda-env.ps1 -RunTests
	```

The environment specification lives in `Training/environment.yml`. Update that file when project dependencies change so local and CI environments stay aligned.

fixed_tokenizer snapshot
------------------------
If a community model's tokenizer fails to load due to remote-code/tokenizers incompatibilities, create a local "fixed_tokenizer" snapshot under the model folder.

1. Activate your training environment (example uses `llm_api` conda env):

```powershell
conda activate llm_api
```

2. Run the helper (example for Qwen model):

```powershell
# Example: use the configured MODEL_ROOT (see Training/domain/path_config.py) instead of hard-coded E:\AI
python Training\tools\make_fixed_tokenizer.py --model-path <MODEL_ROOT>\\Qwen_Qwen3-1.7B --force
```

This will try safe fallbacks (trust_remote_code + use_fast=False) and write `<MODEL_ROOT>\\Qwen_Qwen3-1.7B\\fixed_tokenizer` (MODEL_ROOT defaults to E:\\AI\\Models; see `Training/domain/path_config.py`) which the adapter and the quick finetune harness prefer automatically.

When to use `--use-adapter`
---------------------------
- Use `--use-adapter` when loading community checkpoints that may have config/tokenizer differences (rope_scaling variants, family-specific naming, or model types unrecognized by your installed `transformers`).
- The adapter applies small, conservative non-destructive patches (by default) and prefers `fixed_tokenizer` when present.
- Example: run the quick smoke finetune using the adapter:

```powershell
conda activate llm_api; python Training\tools\quick_finetune_and_sample.py --use-adapter --local-path <MODEL_ROOT>\\Qwen_Qwen3-1.7B --max-examples 2 --force-cpu
```

Notes
-----
- Adapter fallbacks (e.g. coercing qwen3 -> Qwen2Config) are heuristics to make community checkpoints loadable; they are not perfect. When possible, upgrade `transformers` to a version that natively supports the model family or obtain an official transformers-format checkpoint.
- Smoke tests in `Training/tests` are gated by `RUN_SMOKE_TESTS=1` because they may require large local model snapshots. Run them only on machines with the models downloaded.

Conversion helper
-----------------
If you find a model in a non-transformers format (GGUF/GGML/GPTQ), use `Training/tools/convert_helpers.py` to detect the format and (where available) run a converter.

Examples:

```powershell
# detect formats in a model folder (use configured MODEL_ROOT or a local path)
python Training\tools\convert_helpers.py --detect <MODEL_ROOT>\\some_model

# attempt to run the repo's transformers Llama converter (if present)
python Training\tools\convert_helpers.py --convert-llama-ggml-to-hf --input-dir C:\path\to\ggml_dir --output-dir C:\output\hf_dir --model-size 1B --llama-version 3.2
```

The helper will try to find a `convert_llama_weights_to_hf.py` script inside the repository or the installed `transformers` package and invoke it. If no converter is available the helper prints manual instructions and hints.

End-to-end example: LLama GGML/GGUF -> HF (recommended safe flow)
-------------------------------------------------------------
This walkthrough shows a conservative, reproducible flow to convert a community GGUF/GGML model into a Hugging Face-compatible folder and validate it with the adapter helpers. It assumes you have sufficient RAM and disk space and that you run commands in the `llm_api` environment when python modules like `transformers` are required.

1) Detect formats in the model folder

```powershell
python Training\tools\convert_helpers.py --detect C:\path\to\maybe_gguf_model
```

2) If the model is GGUF/GGML, prefer the Transformers LLama converter if present (the helper will attempt to run it). Work on a copy of the original folder and keep backups.

```powershell
# Dry-run to show planned command
python Training\tools\convert_helpers.py --convert-llama-ggml-to-hf --input-dir C:\path\to\ggml_dir --output-dir C:\tmp\hf_out --model-size 7B --dry-run

# To actually run (unsafe), pass --yes (only after confirming the planned command)
python Training\tools\convert_helpers.py --convert-llama-ggml-to-hf --input-dir C:\path\to\ggml_dir --output-dir C:\tmp\hf_out --model-size 7B --yes
```

3) Validate the output by trying to load locally (use the adapter if the model is community-sourced):

```powershell
conda activate llm_api
python Training\tools\check_load_model.py --model-path C:\tmp\hf_out --use-adapter
```

4) If tokenizer load fails, create a fixed tokenizer snapshot and re-run validation:

```powershell
python Training\tools\make_fixed_tokenizer.py --model-path C:\tmp\hf_out --force
python Training\tools\check_load_model.py --model-path C:\tmp\hf_out --use-adapter
```

5) Run a quick smoke finetune to ensure training end-to-end with the adapter:

```powershell
python Training\tools\quick_finetune_and_sample.py --use-adapter --local-path C:\tmp\hf_out --max-examples 2 --force-cpu
```

Notes:
- If you encounter GPTQ artifacts, consider using an `auto_gptq` toolchain to re-create HF-format weights, or use a GGUF-compatible runtime for inference.
- The helper provides suggestions for `auto_gptq` and `llama.cpp` workflows via `--auto-gptq-suggest` and `--llama-cpp-suggest`.
