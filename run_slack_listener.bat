@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ==================================================
echo  Slack 投稿完了リスナー
echo  投稿完了ボタンの押下を監視します
echo ==================================================
echo.

python slack_listener.py

pause
