@echo off
setlocal

set "ENV_NAME=llm_api"
set "PYTEST_MARK=not smoke"
set "PYTHON_EXEC=%PYTHON%"
if "%PYTHON_EXEC%"=="" set "PYTHON_EXEC=python"









exit /b %ERRORLEVEL%"%PYTHON_EXEC%" -m pytest Training/tests %*)    shift    set RUN_SMOKE_TESTS=1if "%1"=="--include-smoke" (:: Default: run only non-smoke tests. Pass --include-smoke to run smoke tests (uses conda env helper if needed).