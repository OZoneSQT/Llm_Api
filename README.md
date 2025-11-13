# Llm_Api

Llm_Api is a laboratory for experimenting with large language models, fine-tuning workflows, and retrieval augmented generation (RAG) services. The repository bundles:

- Training pipelines for downloading models/datasets, preparing adapters, and exporting artifacts.
- A RAG-oriented service layer for serving responses from local or remote models.
- Multi-language API examples for embedding LLM calls into existing applications.

The codebase is being migrated to a Clean Architecture stack so that domain logic remains portable while delivery layers (CLI, PowerShell, service daemons) evolve independently.


## Repository Layout

- `ApiCall/` – Minimal clients in C, C++, C#, Java, JavaScript, PowerShell, and Python for calling Ollama endpoints.
- `Service/` – RAG microservice, dataset ingestion helpers, and PowerShell utilities for provisioning dependencies.
- `Training/` – Clean Architecture implementation for model/tooling workflows (domain, use cases, controllers, and CLI drivers) alongside legacy `tools/` wrappers.
- `accelerate/`, `datasets/`, `diffusers/`, `llama.cpp/`, `peft/`, `transformers/` – Third-party projects vendored for reproducible experimentation.
- `wheelhouse/` – Prebuilt Python wheels consumed by the PowerShell setup scripts.


## Prerequisites and Setup

1. **Install Python 3.13** (matching the pinned tooling) and ensure it is on your `PATH`.
2. **Create and activate a virtual environment** (recommended to avoid polluting the global interpreter):
	 ```powershell
	 python -m venv .venv
	 .\.venv\Scripts\Activate.ps1
	 ```
3. **Install Python dependencies** for the Training stack:
	 ```powershell
	 python -m pip install --upgrade pip
	 python -m pip install -r Training\requirements.txt
	 ```
	 If you prefer the curated wheels, run `dep_install_traning.ps1` for Training or `dep_install_service.ps1` for the RAG service.
4. **Configure storage paths** (defaults suit an `E:\AI\…` layout). Override by exporting environment variables before running any CLI:
	 ```powershell
	 $env:HF_MODEL_ROOT = "D:\Models"
	 $env:HF_DATA_ROOT = "D:\Datasets"
	 $env:HF_CACHE_DIR = "D:\HFCache"
	 ```
5. **Authenticate with Hugging Face** when downloads are required:
	 ```powershell
	 huggingface-cli login
	 ```


## Running the Key Workflows

| Workflow | Command (PowerShell) | Notes |
| --- | --- | --- |
| Download curated models/datasets | `python -m Training.frameworks_drivers.cli.download_runner_cli --yes` | Respects `HF_*` path overrides and mirrors `Training\tools\download_runner_e.py` |
| Prepare dataset manifests | `python -m Training.frameworks_drivers.cli.prepare_datasets_cli --datasets example.yaml` | Delegates to the dataset preparation use case |
| Convert checkpoints to GGUF | `python -m Training.frameworks_drivers.cli.convert_cli --source <path> --target <path>` | Detects quantization strategy automatically |
| Launch RAG service | `pwsh Service\run.ps1` | Calls `Service\main.py` after ensuring the vector store exists |
| Build vector database | `pwsh Service\build_db.ps1` | Runs `Service\build_index.py` over `Service\docs_folder` |
| Quick fine-tune smoke test | `python Training\tools\quick_finetune_and_sample.py --use-adapter --local-path <MODEL_ROOT>\Qwen_Qwen3-1.7B --max-examples 2 --force-cpu` | Wrapper routes through Clean Architecture adapters |

All legacy scripts under `Training/tools/*.py` remain executable; they now forward to the new CLI modules so that external automation keeps working.


## Usage Examples

### 1. Invoke the download runner with a dry run
```powershell
python -m Training.frameworks_drivers.cli.download_runner_cli --dry-run
```
This prints the planned model/dataset downloads and highlights duplicate entries without fetching artifacts.

### 2. Convert a fine-tuned model to GGUF
```powershell
python -m Training.frameworks_drivers.cli.convert_cli \`
	--source "$env:HF_MODEL_ROOT\MyFineTune" \`
	--target "$env:HF_MODEL_ROOT\MyFineTune-GGUF" \`
	--quantization q4_0
```
The conversion controller validates the source folder, selects the correct quantizer, and reports the resulting GGUF files.

### 3. Issue a RAG request from the client
```powershell
python Service\client.py --prompt "Summarise the latest onboarding playbook" --top-k 4
```
Ensure `Service\vector_store` has been generated via `build_db.ps1` before running the client.


## Testing

- Run the targeted adapter regression suite:
	```powershell
	python -m pytest Training\tests\test_adapter_utils.py
	```
- To execute the entire unit-test surface once coverage expands:
	```powershell
	python -m pytest
	```
- Include `--maxfail=1 --disable-warnings` when triaging failures for faster feedback.


## Logging and Error Handling

- All CLI entry points emit status updates to stdout/stderr. To preserve the audit trail in the required `.csv` format, tee the output through PowerShell:
	```powershell
	$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
	python -m Training.frameworks_drivers.cli.download_runner_cli 2>&1 \|
		ForEach-Object { "$timestamp,download_runner,$(Get-Date -Format o),INFO,$_" } \|
		Out-File -FilePath logs\download_runner_$timestamp.csv -Encoding utf8
	```
	The resulting CSV rows follow the structure `run_id,component,timestamp,level,message`.
- Wrap service processes with the same pattern or extend the Python scripts with the standard `logging` module configured with a CSV formatter (e.g., `csv.DictWriter`).
- Handle exceptions centrally: Catch domain-specific errors in controllers, log them at `ERROR`, and re-raise or exit with a non-zero status for automation visibility.
- Never include secrets, PII, or access tokens in the log message payload; redact before persisting.


## Security Considerations

- Store Hugging Face tokens and API keys exclusively in environment variables or secure credential stores; the repo intentionally avoids hardcoding secrets.
- Review third-party datasets/models for licensing or content restrictions before downloading.
- Validate user-supplied prompts and dataset uploads in the Service layer to prevent prompt injection or malicious file ingestion.
- Run sandboxed fine-tuning jobs on isolated machines or containers; restrict filesystem permissions on `MODEL_ROOT`, `DATA_ROOT`, and `.cache` directories.
- Keep dependencies patched by rerunning the PowerShell `dep_update` scripts or upgrading `pip` packages regularly.


## Clean Architecture (Training)

- `Training/domain` – Environment-aware configuration (`PathConfig`) and core entities.
- `Training/app/use_cases` – Conversion, dataset preparation, migration, and model loading orchestration.
- `Training/interface_adapters` – Controllers translating IO for downstream tools.
- `Training/frameworks_drivers` – Command-line entry points and integration glue.

**Architecture Details**

This repo follows a simplified Clean Architecture. A single example request flow (dataset generation) is:

1. CLI: `Training/frameworks_drivers/cli/generate_dataset_cli.py` parses args and calls the controller.
2. Controller: `Training/interface_adapters/controllers/*` validates inputs and invokes the use case.
3. Use Case: `Training/app/use_cases/*` executes domain logic (download, sanitize, write metadata).
4. Domain: `Training/domain/*` provides `PathConfig` and core entities (no framework-specific logic).
5. Persistence/IO: The use case writes files to `HF_DATA_ROOT` / `HF_CACHE_DIR` via low-level helpers.

Diagram (ASCII):

	[CLI] -> [Controller] -> [Use Case] -> [Domain Entities / Helpers] -> [Filesystem / Cache]

Mapping of important modules:

- `PathConfig` -> `Training/domain/path_config.py`
- Dataset validation -> `Training/app/use_cases/validate_dataset.py`
- Dataset generation CLI -> `Training/frameworks_drivers/cli/generate_dataset_cli.py`
- Download runner -> `Training/frameworks_drivers/cli/download_runner_cli.py`

For contributors: prefer adding logic to `app/use_cases` and keep framework-specific code in `frameworks_drivers`.

**Vendoring and name-shadowing**

This repository contains vendored copies of several upstream projects under top-level folders (for reproducibility). When running Python tooling, avoid having a repo-local folder named `datasets` or `transformers` on your active `PYTHONPATH` because it can shadow the installed packages. Use the provided installer script or the `Training/requirements.txt` to install packages into a virtual environment, and remove or rename local vendor folders if you intend to use the PyPI-installed packages instead.

**Architecture Docs**

Detailed diagrams (architecture + request flow) are available at `docs/ARCHITECTURE.md` (Mermaid format). View them in a Markdown renderer that supports Mermaid or open the file directly.

**Scripts Index**

Summary of key setup and helper scripts (paths relative to repository root):

- `dep_install_traning.ps1`: Install Training dependencies from `Training/requirements.txt` (Windows PowerShell).
- `Service/dep_install_service.ps1`: Offline install entrypoint for Service dependencies using `wheelhouse`.
- `dep_download.ps1`: Helper script to prefetch specific artifacts (see script header for usage).
- `Installer.ps1` (if present): General installer wrapper (may be missing in some clones).
- `Training/run-scripts/*`: Cross-platform wrappers to run tests, smoke-tests, generate/train/sanitize workflows.

If a script is missing or you wish to run with a specific Python interpreter, set the `PYTHON` environment variable to the full path of the desired Python executable before calling the script.

**Testing and Run Guidance (expanded)**

- Create and activate a virtual environment:

	```powershell
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	```

- Install Training dependencies:

	```powershell
	python -m pip install --upgrade pip
	python -m pip install -r Training\requirements.txt
	```

- Run unit tests (non-smoke):

	```powershell
	C:\GitHub\Llm_Api\Training\run-scripts\run-tests.ps1
	```

- Run tests including smoke tests (set environment var for process):

	```powershell
	$env:RUN_SMOKE_TESTS = 1
	C:\GitHub\Llm_Api\Training\run-scripts\run-smoke-tests.ps1
	```

- If a smoke test requires `transformers` or large site caches, the test runner will skip with a clear message when the dependency or cached snapshots are not available.

Please see `Training/run-scripts/README.md` for examples and `Training/requirements.txt` for the canonical dependency list.
- `Training/tools` – Backwards-compatible shims that call the new CLI modules.


## Contributing and Next Steps

- Expand unit coverage across the newly introduced use cases (see `Training/tests`).
- Add structured logging helpers to remove the need for manual piping.
- Document additional service endpoints and roll out integration tests for the RAG pipeline.
