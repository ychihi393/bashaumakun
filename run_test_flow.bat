@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ==================================================
echo  テストフロー: 選定リセット + フル実行
echo  Slack選定 - 画像生成 - Drive - Slack通知
echo ==================================================
echo.

python -X utf8 test_flow.py --reset

echo.
pause
