@echo off
setlocal

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "VENV_DIR=%ROOT%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if exist "%VENV_PY%" (
  set "VIRTUAL_ENV=%VENV_DIR%"
  set "PATH=%VENV_DIR%\Scripts;%PATH%"
  set "PYTHON=%VENV_PY%"
)

set "LIBTORCH_USE_PYTORCH=1"
set "LIBTORCH_BYPASS_VERSION_CHECK=1"

if exist "%VENV_DIR%\Lib\site-packages\torch\lib" (
  set "PATH=%VENV_DIR%\Lib\site-packages\torch\lib;%PATH%"
)

if "%~1"=="" (
  echo Usage: %~nx0 command [args...]
  exit /b 1
)

%*
