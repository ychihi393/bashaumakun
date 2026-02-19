#!/usr/bin/env python3
"""
Google Drive アップロードモジュール

物件フォルダを Google Drive にアップロードし、共有リンクを取得する。
サービスアカウント認証を使用（OAuth の手動ログイン不要）。

.env 設定:
    GOOGLE_DRIVE_CREDENTIALS_JSON - サービスアカウントJSONファイルのパス
    GOOGLE_DRIVE_FOLDER_ID        - アップロード先の親フォルダID（省略時はマイドライブ直下）
"""

import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    _DRIVE_AVAILABLE = True
except ImportError:
    _DRIVE_AVAILABLE = False


def is_configured() -> bool:
    """認証が設定されているか"""
    if not _DRIVE_AVAILABLE:
        return False
    cred_path = os.getenv("GOOGLE_DRIVE_CREDENTIALS_JSON", "").strip()
    return bool(cred_path and Path(cred_path).expanduser().exists())


def create_run_folder(run_name: str) -> Optional[str]:
    """実行用の親フォルダを作成し、フォルダIDを返す。"""
    service = _get_drive_service()
    if not service:
        return None
    parent_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip() or None
    parents = [parent_id] if parent_id else []
    try:
        folder_meta = {
            "name": run_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parents:
            folder_meta["parents"] = parents
        root = service.files().create(body=folder_meta, fields="id").execute()
        return root.get("id")
    except Exception as e:
        logging.warning("Google Drive フォルダ作成失敗: %s", e)
        return None


def _get_drive_service():
    if not _DRIVE_AVAILABLE:
        return None
    cred_path = os.getenv("GOOGLE_DRIVE_CREDENTIALS_JSON", "").strip()
    if not cred_path:
        return None
    path = Path(cred_path).expanduser()
    if not path.exists():
        logging.warning("Google Drive認証ファイルが見つかりません: %s", cred_path)
        return None
    try:
        creds = service_account.Credentials.from_service_account_file(
            str(path),
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        logging.warning("Google Drive認証に失敗: %s", e)
        return None


def upload_folder_and_get_link(
    local_folder: Path,
    folder_name: str,
    parent_id: Optional[str] = None,
) -> Optional[str]:
    """
    ローカルフォルダを Drive にアップロードし、共有リンク（Anyone with link can view）を返す。
    """
    service = _get_drive_service()
    if not service:
        return None

    parent_id = parent_id or os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip() or None
    parents = [parent_id] if parent_id else []

    try:
        # ルートフォルダ作成
        folder_meta = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parents:
            folder_meta["parents"] = parents
        root = service.files().create(body=folder_meta, fields="id").execute()
        folder_id = root.get("id")
        if not folder_id:
            return None

        # ファイルをアップロード
        for f in local_folder.iterdir():
            if not f.is_file():
                continue
            mime = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
            file_meta = {"name": f.name, "parents": [folder_id]}
            media = MediaFileUpload(str(f), mimetype=mime, resumable=True)
            service.files().create(body=file_meta, media_body=media, fields="id").execute()

        # 共有設定（リンクを知っていれば誰でも閲覧可）
        service.permissions().create(
            fileId=folder_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()

        return f"https://drive.google.com/drive/folders/{folder_id}"
    except HttpError as e:
        logging.warning("Google Drive アップロード失敗: %s", e)
        return None
    except Exception as e:
        logging.warning("Google Drive アップロードエラー: %s", e)
        return None
