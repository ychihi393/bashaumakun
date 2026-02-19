@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"

if defined ITANJI_KEEP_WINDOW (
  set "KEEP_WINDOW=%ITANJI_KEEP_WINDOW%"
) else (
  set "KEEP_WINDOW=1"
)
if defined ITANJI_SKIP_SCRAPE (
  set "SKIP_SCRAPE=%ITANJI_SKIP_SCRAPE%"
) else (
  set "SKIP_SCRAPE=0"
)
if defined ITANJI_ALLOW_STALE_DATA (
  set "ALLOW_STALE_DATA=%ITANJI_ALLOW_STALE_DATA%"
) else (
  set "ALLOW_STALE_DATA=0"
)
if defined ITANJI_CLEAN_RUN (
  set "CLEAN_RUN=%ITANJI_CLEAN_RUN%"
) else (
  set "CLEAN_RUN=1"
)
REM .env の SLACK_BOT_TOKEN を確認（Python で読み込む）
python -c "from dotenv import load_dotenv; from pathlib import Path; load_dotenv(Path(r'%~dp0.env')); import os; need=('SLACK_BOT_TOKEN','SLACK_APP_TOKEN','SLACK_CHANNEL'); exit(0 if all(os.environ.get(k,'').strip() for k in need) else 1)" 2>nul
if errorlevel 1 (
  set "USE_SLACK=0"
) else (
  set "USE_SLACK=1"
)

set "KEEP_WINDOW=%KEEP_WINDOW: =%"
set "SKIP_SCRAPE=%SKIP_SCRAPE: =%"
set "ALLOW_STALE_DATA=%ALLOW_STALE_DATA: =%"
set "CLEAN_RUN=%CLEAN_RUN: =%"

if /i "%KEEP_WINDOW%"=="true" set "KEEP_WINDOW=1"
if /i "%KEEP_WINDOW%"=="yes" set "KEEP_WINDOW=1"
if /i "%KEEP_WINDOW%"=="on" set "KEEP_WINDOW=1"
if not "%KEEP_WINDOW%"=="1" set "KEEP_WINDOW=0"

if /i "%SKIP_SCRAPE%"=="true" set "SKIP_SCRAPE=1"
if /i "%SKIP_SCRAPE%"=="yes" set "SKIP_SCRAPE=1"
if /i "%SKIP_SCRAPE%"=="on" set "SKIP_SCRAPE=1"
if not "%SKIP_SCRAPE%"=="1" set "SKIP_SCRAPE=0"

if /i "%ALLOW_STALE_DATA%"=="true" set "ALLOW_STALE_DATA=1"
if /i "%ALLOW_STALE_DATA%"=="yes" set "ALLOW_STALE_DATA=1"
if /i "%ALLOW_STALE_DATA%"=="on" set "ALLOW_STALE_DATA=1"
if not "%ALLOW_STALE_DATA%"=="1" set "ALLOW_STALE_DATA=0"

if /i "%CLEAN_RUN%"=="true" set "CLEAN_RUN=1"
if /i "%CLEAN_RUN%"=="yes" set "CLEAN_RUN=1"
if /i "%CLEAN_RUN%"=="on" set "CLEAN_RUN=1"
if not "%CLEAN_RUN%"=="1" set "CLEAN_RUN=0"

echo ==================================================
echo  Full pipeline
echo  1) Scraping  2) Slack selection  3) Post image generation
echo ==================================================
echo [Config] SKIP_SCRAPE=%SKIP_SCRAPE%  CLEAN_RUN=%CLEAN_RUN%  ALLOW_STALE_DATA=%ALLOW_STALE_DATA%  USE_SLACK=%USE_SLACK%

if "%CLEAN_RUN%"=="1" (
  echo [0/5] Cleaning previous output data...
  python clean_pipeline_state.py
  if errorlevel 1 goto :fail
) else (
  echo [0/5] Skipping clean ^(ITANJI_CLEAN_RUN=0^)
)

if "%SKIP_SCRAPE%"=="1" goto :skip_scrape

echo [1/5] Running scraper...
call run_scrape_itanji_video.bat
if errorlevel 1 goto :scrape_failed
goto :after_scrape

:skip_scrape
echo [1/5] Skipping scrape ^(ITANJI_SKIP_SCRAPE=1^)
goto :after_scrape

:scrape_failed
echo [ERROR] Scraping step failed.
if "%ALLOW_STALE_DATA%"=="1" goto :stale_check
goto :fail

:stale_check
echo [INFO] ALLOW_STALE_DATA=1 - checking existing assets/data.json...
python check_assets_data.py
if errorlevel 1 (
  echo [ERROR] Existing assets/data.json is not usable.
  goto :fail
)
goto :after_scrape

:after_scrape
echo [2/5] Checking assets/data.json...
python check_assets_data.py
if errorlevel 3 (
  echo [OK] No adopted data. Skipping post image generation.
  goto :success
)
if errorlevel 1 (
  echo [ERROR] assets/data.json validation failed.
  goto :fail
)

if "%USE_SLACK%"=="1" (
  echo [3/5] Running Slack image selection ^(slack_selector.py^)...
  echo        Slack???????????Slack??????????????????
  python slack_selector.py
  if errorlevel 1 (
    echo [ERROR] Slack selection failed or timed out.
    echo         Stop pipeline to avoid non-manual cover selection.
    goto :fail
  )
) else (
  echo [3/5] Skipping Slack selection ^(.env ? SLACK_BOT_TOKEN/SLACK_APP_TOKEN/SLACK_CHANNEL ???????^)
  echo        Slack??3????????????????????????
)

echo [4/5] Running post image generation (main.py)...
python main.py
if errorlevel 1 (
  echo [ERROR] main.py failed.
  goto :fail
)

echo [5/5] All steps completed.
echo [Output] Check Python logs above for saved file paths.
goto :success

:success
if "%KEEP_WINDOW%"=="1" (
  echo.
  echo Done. Press any key to close.
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
