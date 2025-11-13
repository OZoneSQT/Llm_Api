@echo off
setlocal








exit /b %ERRORLEVEL%"%PYTHON_EXEC%" -m pytest Training/tests %*if "%PYTHON_EXEC%"=="" set "PYTHON_EXEC=python"set "PYTHON_EXEC=%PYTHON%"set RUN_SMOKE_TESTS=1:: Run smoke tests by setting RUN_SMOKE_TESTS=1 for the process.