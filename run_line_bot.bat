@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ==================================================
echo  LINE Bot サーバー
echo  物件番号をLINEで送るとカルーセルで返信します
echo ==================================================
echo.

python -c "import fastapi, uvicorn, linebot" 2>nul
if errorlevel 1 (
  echo [INFO] 依存パッケージをインストール中...
  python -m pip install -r requirements_line.txt
  if errorlevel 1 (
    echo [ERROR] インストール失敗
    pause
    exit /b 1
  )
)

echo [INFO] サーバーを起動します (ポート: 8000)
echo [INFO] LINE Developer Console のウェブフック URL:
echo         LINE_SERVER_URL/webhook
echo.
echo [INFO] 外部公開が必要な場合は別ウィンドウで以下を実行:
echo   ngrok http 8000
echo   または: cloudflared tunnel --url http://localhost:8000
echo.
python line_bot.py

pause
