@echo off
setlocal
cd /d "%~dp0"
set TAC_TEST_MODE=1
set QT_QPA_PLATFORM=offscreen
".venv\Scripts\python.exe" -m pytest -q tests --junitxml=reports\tests.xml
set tac_exit=%errorlevel%
pause
exit /b %tac_exit%
