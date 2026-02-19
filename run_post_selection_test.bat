@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"

set KEEP_WINDOW=%ITANJI_KEEP_WINDOW%
if "%KEEP_WINDOW%"=="" set KEEP_WINDOW=1

echo ==================================================
echo  Post-scrape selection test flow
echo  Uses existing assets/data.json only
echo  Runs: Slack selection -> main.py
echo ==================================================

python test_post_selection_flow.py
if errorlevel 1 (
  echo [ERROR] Test flow failed.
  echo [INFO] ExitCode=%ERRORLEVEL%
  goto :fail
)

echo [OK] Test flow finished.
if "%KEEP_WINDOW%"=="1" (
  echo.
  echo Finished. Press any key to close this window.
  pause >nul
)
exit /b 0

:fail
if "%KEEP_WINDOW%"=="1" (
  echo.
  echo Failed. Check logs above, then press any key.
  pause >nul
)
exit /b 1
