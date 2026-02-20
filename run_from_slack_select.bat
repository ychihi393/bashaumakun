@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"

echo ==================================================
echo  [テスト用] Slack選定 → 画像生成
echo  スクレイピングをスキップ / 既存 assets/data.json を使用
echo ==================================================
echo.

REM ─── 前回の選定をリセット（毎回やり直し）──────────
if exist "assets\slack_selections.json" (
  del "assets\slack_selections.json"
  echo [0] 前回のSlack選定をリセットしました
)

REM ─── assets/data.json の存在確認 ─────────────────
echo [1/3] assets/data.json 確認...
python check_assets_data.py
if errorlevel 3 (
  echo [OK] 採用データが0件です。処理をスキップします。
  goto :success
)
if errorlevel 1 (
  echo [ERROR] assets/data.json が見つからないか無効です。
  echo         先にスクレイピング ^(run_full_pipeline.bat^) を実行してください。
  goto :fail
)

REM ─── Slack 画像選定 ───────────────────────────────
echo.
echo [2/3] Slack 画像選定 ^(slack_selector.py^)...
echo        Slack にカタログ画像を送信するので、選択が終わるまで待機します。
python slack_selector.py
if errorlevel 1 (
  echo [ERROR] Slack選定に失敗しました。ログを確認してください。
  goto :fail
)

REM ─── 画像生成・Slack送信 ──────────────────────────
echo.
echo [3/3] 画像生成・Slack送信 ^(main.py^)...
python main.py
if errorlevel 1 (
  echo [ERROR] main.py に失敗しました。ログを確認してください。
  goto :fail
)

:success
echo.
echo [OK] 完了しました。
echo [出力先] output\投稿用出力\採用\ 以下に各物件フォルダが作成されています。
echo.
pause
exit /b 0

:fail
echo.
echo [ERROR] 処理に失敗しました。上のログを確認してください。
echo.
pause
exit /b 1
