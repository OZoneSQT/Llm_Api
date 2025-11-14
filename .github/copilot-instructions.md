- [X] Verify that the copilot-instructions.md file in the .github directory is created.

- [X] Clarify Project Requirements  
	Ask for project type, language, and frameworks if not specified. Skip if already provided.

- [X] Scaffold the Project  
	Ensure that the previous step has been marked as completed.  
	Call project setup tool with projectType parameter.  
	Run scaffolding command to create project files and folders.  
	Use '.' as the working directory.  
	If no appropriate projectType is available, search documentation using available tools.  
	Otherwise, create the project structure manually using available file creation tools.  

- [X] Customize the Project  
	Verify that all previous steps have been completed successfully and you have marked the step as completed.  
	Develop a plan to modify codebase according to user requirements.  
	Apply modifications using appropriate tools and user-provided references.  
	Skip this step for "Hello World" projects.  

- [X] Install Required Extensions  
	ONLY install extensions provided mentioned in the get_project_setup_info. Skip this step otherwise and mark as completed.

- [X] Compile the Project  
	Verify that all previous steps have been completed.  
	Install any missing dependencies.  
	Run diagnostics and resolve any issues.  
	Check for markdown files in project folder for relevant instructions on how to do this.  

- [X] Create and Run Task  
	Verify that all previous steps have been completed.  
	Check https://code.visualstudio.com/docs/debugtest/tasks to determine if the project needs a task. If so, use the create_and_run_task to create and launch a task based on package.json, README.md, and project structure.  
	Skip this step otherwise.

- [X] Launch the Project   
	Verify that all previous steps have been completed.  
	Prompt user for debug mode, launch only if confirmed.  

- [X] Ensure Documentation is Complete  
	Verify that all previous steps have been completed.  
	Verify that README.md and the copilot-instructions.md file in the .github directory exist and contain current project information.  
	README.md must include:  
	- Project title and description  
	- Setup instructions  
	- Usage examples  
	- Testing instructions  
	- Logging and error handling details (including `.csv` format)  
	- Security considerations
	- Spelling and grammar check for clarity and professionalism.
	- Add Dependencies and installation instructions	
	- Add Contribution guidelines
	- Security considerations
	- Spelling and grammar check for clarity and professionalism.
	Clean up the copilot-instructions.md file in the .github directory by removing all HTML comments.

---

## UX Guidelines

Prioritize user experience in all code changes.

- Provide clear and informative feedback to users.
- Provide intuitive navigation and interaction flows if applicable.
- ensure process status visibility in terminal for users during long-running operations, unless running -Silent mode.

---

## Clean Code Standards

All code must follow the principles from *Clean Code* by Robert C. Martin:

- Use **descriptive, intention-revealing names** for all identifiers.  
- Keep **functions small and focused** on a single task.  
- Apply the **Single Responsibility Principle** to all classes and modules.  
- Avoid code duplication (DRY).  
- Write **self-explanatory code**; use comments only to explain *why*, not *what*.  
- Maintain **consistent formatting** and structure.  
- Validate inputs early and **fail fast** with meaningful errors.

---

## Clean Architecture Guidelines

Structure the project using Clean Architecture:

- **Entities**: Core business rules and logic.  
- **Use Cases**: Application-specific orchestration.  
- **Interface Adapters**: Controllers, presenters, gateways.  
- **Frameworks & Drivers**: UI, database, external services.

**Dependency Rule**: Inner layers must not depend on outer layers. Use interfaces and inversion of control.

---

## Secure Coding Requirements

Security is mandatory unless explicitly waived:

- Sanitize and validate all external inputs.  
- Never hardcode secrets—use environment variables or secure vaults.  
- Apply the **principle of least privilege**.  
- Avoid exposing sensitive data in logs or errors.  
- Protect against common vulnerabilities (e.g., XSS, SQL injection, CSRF).

---

## Unit Testing Policy

Unit tests are required for all business logic:

- Target >80% coverage on core logic.  
- Tests must be isolated and deterministic.  
- Use descriptive test names: `should_<behavior>_when_<condition>`.  
- Include tests for edge cases and failure scenarios.

---

## Error Logging Standards

All runtime errors must be logged:

- Use structured logging (e.g., `.csv` format).  
# copilot-instructions.md — Llm_Api (concise agent guide)

Purpose: provide immediate, actionable guidance for AI coding agents working in this repository — what to edit, how to run things, and where to look for conventions.

**Big Picture**
- **Architecture:** Clean Architecture for `Training/` (domain -> use_cases -> interface_adapters -> frameworks_drivers). See `docs/ARCHITECTURE.md` and `README.md` (section "Clean Architecture (Training)").
- **Service layer:** `Service/` implements the RAG microservice and CLI wrappers (`Service/main.py`, `Service/client.py`). Vector store lives under `Service/vector_store`.
- **Clients:** `ApiCall/` contains language-specific Ollama clients (`OllamaAPI.py`, `OllamaAPI.js`, `OllamaAPI.ps1`, etc.). Use these when integrating model endpoints.
- **Vendored libs:** Top-level folders like `lib/transformers`, `llama.cpp`, `diffusers` are vendored for reproducibility — avoid adding them to `PYTHONPATH` if you want the PyPI packages instead.

**Developer workflows (exact commands)**
- **Create venv & activate (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Install Training deps:** `python -m pip install -r Training\requirements.txt` or run `dep_install_traning.ps1` (Windows).
- **Install Service deps (offline):** `pwsh Service\dep_install_service.ps1` (uses `wheelhouse/`).
- **Build vector DB:** `pwsh Service\build_db.ps1` (runs `Service\build_index.py`).
- **Launch RAG service (PowerShell):** `pwsh Service\run.ps1` (calls `Service\main.py`).
- **Quick tests:** `python -m pytest Training\tests\test_adapter_utils.py` or `python -m pytest` for broader runs.

**Project-specific conventions & patterns**
- **Clean Architecture mapping:** Prefer placing business logic in `Training/app/use_cases` and keep IO in `frameworks_drivers`.
- **Vendored-transformers rules:** For files under `lib/transformers`, see `lib/transformers/.github/copilot-instructions.md` — follow its copy/`modular` patterns and use `make fixup` when editing those directories.
- **Logging format:** CLI output is formatted to CSV rows: `run_id,component,timestamp,level,message`. Example piping shown in `README.md` (search for "csv"). Use this format for new CLI entrypoints.
- **Secrets:** Never hardcode tokens — use `HF_*` env vars (`HF_MODEL_ROOT`, `HF_DATA_ROOT`, `HF_CACHE_DIR`) and `huggingface-cli login` when needed.

**Integration points & external dependencies**
- **Hugging Face:** downloads and model conversions are driven by `Training/frameworks_drivers/cli/*` and respect `HF_*` env vars.
- **Ollama endpoints:** client implementations in `ApiCall/` show how the service calls remote models; use them as examples for new language clients.
- **Prebuilt wheels:** `wheelhouse/` is used by `Service\dep_install_service.ps1` for reproducible installs.

**Where to look (key files to read first)**
- `README.md` — high-level workflows and exact PowerShell commands.
- `docs/ARCHITECTURE.md` — mermaid diagrams and flow details.
- `Training/frameworks_drivers/cli/*.py` — CLI entrypoints and invocation patterns.
- `Service/main.py`, `Service/client.py`, `Service/build_index.py` — RAG runtime and vector store build.
- `ApiCall/*` — cross-language Ollama client examples.

**Edit & test checklist (short)**
- Run the local venv and install `Training/requirements.txt`.
- If editing vendored code under `lib/transformers`, follow `lib/transformers/.github/copilot-instructions.md` and run `make fixup` where applicable.
- Update `README.md` and `docs/ARCHITECTURE.md` when modifying public APIs or workflows.
- Unit tests: add tests under `Training/tests` matching the `use_cases` you change; run `pytest` targeted first, then full run.

If anything in this summary is unclear or you'd like more examples (e.g., a sample PR checklist or a small runnable example that exercises `Service/main.py`), tell me which section to expand and I'll iterate.
