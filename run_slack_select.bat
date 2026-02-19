@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"

echo ==================================================
echo  Slack 画像選定
echo  scrape 後・main.py 前に実行してください
echo ==================================================

if not defined SLACK_BOT_TOKEN (
  echo [ERROR] SLACK_BOT_TOKEN が設定されていません。
  echo         .env に以下を追加してください:
  echo           SLACK_BOT_TOKEN=xoxb-...
  echo           SLACK_APP_TOKEN=xapp-...
  echo           SLACK_CHANNEL=C01234ABCDE
  pause >nul
  exit /b 1
)

python slack_selector.py
if errorlevel 1 (
  echo [ERROR] Slack選定に失敗しました。ログを確認してください。
  pause >nul
  exit /b 1
)

echo.
echo Slack選定が完了しました。次に main.py または run_full_pipeline.bat を実行してください。
pause >nul
exit /b 0
