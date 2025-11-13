@echo off
setlocal

set "PYTHON_EXEC=%PYTHON%"
if "%PYTHON_EXEC%"=="" set "PYTHON_EXEC=python"

"%PYTHON_EXEC%" -m Training.frameworks_drivers.cli.tune_model_cli %*
exit /b %ERRORLEVEL%
