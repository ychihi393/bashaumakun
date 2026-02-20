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

echo.
echo [OK] 画像生成が完了しました。
echo [出力先] output\投稿用出力\採用\  以下に各物件フォルダが作成されています。
echo.
pause
exit /b 0

:fail
echo.
echo [ERROR] 処理に失敗しました。上のログを確認してください。
echo.
pause
exit /b 1
