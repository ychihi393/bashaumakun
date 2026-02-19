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
echo  Post image generation only (uses assets/data.json)
echo ==================================================

echo [1/2] Check dependencies
python -c "import cv2,requests,bs4,PIL,dotenv; print('deps_ok')"
if errorlevel 1 (
  echo [INFO] Installing missing dependencies...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    goto :fail
  )
)

echo [2/2] Run main.py
python main.py
if errorlevel 1 (
  echo [ERROR] main.py failed.
  goto :fail
)

echo [OK] Post image generation finished.
echo [Output] See Python logs above for exact saved file paths.
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
