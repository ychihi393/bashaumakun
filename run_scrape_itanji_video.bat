@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"

set KEEP_WINDOW=%ITANJI_KEEP_WINDOW%
if "%KEEP_WINDOW%"=="" set KEEP_WINDOW=1
set AUTO_INSTALL_CHROMIUM=%ITANJI_AUTO_INSTALL_CHROMIUM%
if "%AUTO_INSTALL_CHROMIUM%"=="" set AUTO_INSTALL_CHROMIUM=0
set ITANJI_VIDEO_MAX_PROPERTIES=%ITANJI_VIDEO_MAX_PROPERTIES%
if "%ITANJI_VIDEO_MAX_PROPERTIES%"=="" set ITANJI_VIDEO_MAX_PROPERTIES=0
set ITANJI_VIDEO_MAX_PAGES=%ITANJI_VIDEO_MAX_PAGES%
if "%ITANJI_VIDEO_MAX_PAGES%"=="" set ITANJI_VIDEO_MAX_PAGES=100

echo ==================================================
echo  Itanji scraping step
echo ==================================================
echo [Config] ITANJI_VIDEO_MAX_PROPERTIES=%ITANJI_VIDEO_MAX_PROPERTIES%  ITANJI_VIDEO_MAX_PAGES=%ITANJI_VIDEO_MAX_PAGES%

echo [1/3] Check Python
python --version
if errorlevel 1 (
  echo [ERROR] Python not found.
  goto :fail
)

echo [2/3] Check dependencies
python -c "import cv2,playwright,google.genai,pandas,requests,bs4,lxml,PIL,dotenv; print('deps_ok')"
if errorlevel 1 (
  echo [INFO] Installing missing dependencies...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    goto :fail
  )
)

echo [2.5/3] Check Playwright Chromium runtime files
python -c "from pathlib import Path; import os,sys; root=Path(os.getenv('LOCALAPPDATA',''))/'ms-playwright'; ok=root.exists() and any(p.name.startswith('chromium-') for p in root.iterdir()); print('chromium_ok' if ok else 'chromium_missing'); sys.exit(0 if ok else 2)"
if errorlevel 2 (
  echo [WARN] Chromium runtime files not found.
  if "%AUTO_INSTALL_CHROMIUM%"=="1" (
    echo [INFO] Installing Chromium...
    python -m playwright install chromium
    if errorlevel 1 (
      echo [WARN] Chromium install failed. Continue with current environment.
    )
  ) else (
    echo [INFO] Auto install is disabled. Set ITANJI_AUTO_INSTALL_CHROMIUM=1 to enable.
  )
)

echo [3/3] Run scraper
python scrape_itanji_video.py
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo [ERROR] Scraper failed. ExitCode=%EXIT_CODE%
  goto :fail_with_code
)

echo [OK] Scraper finished.
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

:fail_with_code
if "%KEEP_WINDOW%"=="1" (
  echo.
  echo Failed. Check logs above, then press any key.
  pause >nul
)
exit /b %EXIT_CODE%
