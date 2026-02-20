@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

echo ==================================================
echo  Stable Diffusion セットアップ
echo  torch / diffusers / transformers / accelerate
echo ==================================================
echo.

python -m pip install -r requirements_sd.txt
if errorlevel 1 (
  echo.
  echo [ERROR] インストールに失敗しました。ログを確認してください。
  pause
  exit /b 1
)

echo.
echo ==================================================
echo  インストール完了！
echo.
echo  次のステップ:
echo  .env ファイルに以下の1行を追加してください:
echo    USE_SD_OUTPAINTING=1
echo ==================================================
echo.
pause
