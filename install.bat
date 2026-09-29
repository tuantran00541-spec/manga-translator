@echo off
rem Install once, then open any cmd window and type: manga
setlocal
cd /d "%~dp0"
set "PY="
for %%V in (3.12 3.11 3.10) do (
  if not defined PY py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
)
if not defined PY python -c "import sys; sys.exit(not (3,10) <= sys.version_info[:2] <= (3,12))" >nul 2>&1 && set "PY=python"
if not defined PY (
  echo Can Python 3.12: tai tai https://www.python.org/downloads/ roi chay lai install.bat
  pause
  exit /b 1
)
%PY% scripts\install.py
if errorlevel 1 (
  echo Cai dat chua xong, xem loi o tren roi chay lai install.bat
  pause
  exit /b 1
)
pause
