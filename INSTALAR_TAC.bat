@echo off
setlocal
cd /d "%~dp0"
echo TAC - Instalacao de bibliotecas. Nao baixa modelos nem pesos treinados.
if not exist ".venv\Scripts\python.exe" (
  py -3.13 -m venv .venv
  if errorlevel 1 goto erro
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
  echo CUDA nao instalada. Tentando a versao CPU...
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 goto erro
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
if errorlevel 1 goto erro
".venv\Scripts\python.exe" -m pip freeze > requirements-lock.txt
echo Instalacao concluida. Abra INICIAR_TAC.bat.
pause
exit /b 0
:erro
echo A instalacao falhou. Verifique internet, espaco livre e Python 3.13 de 64 bits.
pause
exit /b 1
