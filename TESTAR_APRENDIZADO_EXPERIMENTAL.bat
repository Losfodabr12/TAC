@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo TAC - teste separado da rede neural
echo Feche a janela principal da TAC antes de continuar.
echo.
echo 1 - Candidato treinado somente com suas 36 conversas
echo 2 - Modelo ativo original, para comparar
choice /C 12 /N /M "Escolha 1 ou 2: "
if errorlevel 2 goto ativo
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\chat_file_experiment.py" --model experimental
goto fim
:ativo
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\chat_file_experiment.py" --model ativo
:fim
echo.
pause
endlocal
