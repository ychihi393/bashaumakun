@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

echo ==================================================
echo  LINE Bot システム起動
echo ==================================================
echo.

set NGROK_EXE=C:\Users\yamag\AppData\Local\Microsoft\WinGet\Packages\Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe\ngrok.exe

rem ── 既存プロセスを停止 ───────────────────────────────────────────
taskkill /f /im ngrok.exe >nul 2>&1
echo [1] ngrok 起動中...
start "ngrok" /min "%NGROK_EXE%" http 8000

rem ── ngrok の公開URL を取得して .env に書き込む ─────────────────
timeout /t 4 /nobreak >nul
python "%~dp0_update_ngrok_url.py"
if errorlevel 1 (
    echo [ERROR] ngrok URL の取得に失敗しました
    pause
    exit /b 1
)

rem ── LINE Bot サーバー起動 ────────────────────────────────────────
echo [3] LINE Bot サーバー起動中...
start "LINE Bot" cmd /k "chcp 65001 >nul && cd /d %~dp0 && python line_bot.py"

timeout /t 2 /nobreak >nul
echo.
echo ==================================================
echo  起動完了！
echo.
for /f "tokens=*" %%i in ('python -c "from dotenv import dotenv_values; v=dotenv_values('.env'); print(v.get('LINE_SERVER_URL',''))"') do set SERVER_URL=%%i
echo  Webhook URL: %SERVER_URL%/webhook
echo.
echo  [重要] LINE Developer Console で上記 Webhook URL を設定してください
echo  https://developers.line.biz/console/
echo ==================================================
echo.
pause
