@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 (
  echo.
  echo Cai dat chua xong, xem loi o tren roi chay lai install.bat
)
pause
