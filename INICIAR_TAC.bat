@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Ambiente Python ausente. Execute INSTALAR_TAC.bat primeiro.
  pause
  exit /b 1
)
if "%~1"=="" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" -m tac
  exit /b 0
)
"%~dp0.venv\Scripts\python.exe" -m tac %*
exit /b %errorlevel%
