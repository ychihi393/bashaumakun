@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==================================================
echo  Google Drive 共有ドライブ セットアップ手順
echo ==================================================
echo.
echo サービスアカウントでアップロードするには
echo 「共有ドライブ」が必要です。
echo.
echo ---- 手順 ----------------------------------------
echo.
echo 1. ブラウザで Google Drive を開く
echo    https://drive.google.com
echo.
echo 2. 左メニューの「共有ドライブ」→「新規」をクリック
echo    名前: 投稿自動化 など（任意）
echo.
echo 3. 作成した共有ドライブを右クリック
echo    →「メンバーを管理」
echo    →「メンバーを追加」に以下のメールを入力:
echo.
echo    bashaumakun@bashauma.iam.gserviceaccount.com
echo.
echo    権限: 「コンテンツ管理者」または「投稿者」
echo.
echo 4. 共有ドライブの URL を確認する
echo    例: https://drive.google.com/drive/folders/XXXXXXXXXXXX
echo    この XXXXXXXXXXXX が「共有ドライブID」
echo.
echo 5. .env の GOOGLE_DRIVE_FOLDER_ID を更新する
echo    GOOGLE_DRIVE_FOLDER_ID=XXXXXXXXXXXX
echo.
echo --------------------------------------------------
echo.
pause
