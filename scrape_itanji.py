"""
scrape_itanji.py
================
イタンジBBから物件データを取得し、ローカル保存するスクリプト。
Next.jsでのWEB公開用にデータを整形。

【修正版機能】
- 初回: 全件取得
- 2回目以降 (UPDATE_MODE=True):
  - リスト画面で「申込あり」は即スキップ
  - リスト画面で「取得済みURL」は即スキップ
  - 既存物件の更新は行わず、削除判定のみ実施（未検出/申込あり）
"""

import json
import os
import re
import requests
import time
import random
import signal
import sys
import shutil
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
from urllib.parse import urlparse
try:
    import boto3
    from botocore.config import Config
except Exception:
    boto3 = None
    Config = None

# Windows環境での絵文字表示対応
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# 環境変数読み込み
load_dotenv()

# ============================================================
# 設定
# ============================================================
OUTPUT_DIR = "output"
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
TOP_URL = "https://bukkakun.com/"
LIST_SEARCH_URL = "https://itandibb.com/rent_rooms/list"  # リスト検索ページのURL

# ログイン情報
ITANJI_EMAIL = os.getenv("ITANJI_EMAIL")
ITANJI_PASSWORD = os.getenv("ITANJI_PASSWORD")

# R2自動アップロード設定
R2_UPLOAD_ENABLED = os.getenv("R2_UPLOAD_ENABLED", "").lower() in ("1", "true", "yes", "on")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "heyamatch-properties")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else ""

# ソース識別（他スクレイパーとの共存のため）
SOURCE_ID = "itanji"

# インデックスファイルパス（スクレイパー専用 — 他スクレイパーとの競合を防ぐ）
INDEX_FILE = os.path.join(OUTPUT_DIR, "itanji_index.json")
# 旧共有ファイル（後方互換性用：起動時にマイグレーション）
LEGACY_SHARED_INDEX = os.path.join(OUTPUT_DIR, "properties_index.json")

# 画像圧縮設定
COMPRESS_IMAGES = os.getenv("ITANJI_COMPRESS_IMAGES", "0").lower() in ("1", "true", "yes", "on")
COMPRESS_MAX_SIDE = int(os.getenv("ITANJI_COMPRESS_MAX_SIDE", "1200"))
COMPRESS_QUALITY = int(os.getenv("ITANJI_COMPRESS_QUALITY", "78"))

# バックグラウンドR2アップロード用
_upload_executor = ThreadPoolExecutor(max_workers=2)
_upload_futures: list = []
_upload_executor_closed = False

# 誤検知による全削除事故を避けるため、空インデックスのR2反映は明示許可時のみ
ALLOW_EMPTY_ITANJI_INDEX_UPLOAD = os.getenv("ALLOW_EMPTY_ITANJI_INDEX_UPLOAD", "0").lower() in ("1", "true", "yes", "on")
# Safety guard: if area list detection fails and found URLs are zero, skip area cleanup.
SKIP_AREA_CLEANUP_WHEN_FOUND_EMPTY = os.getenv("SKIP_AREA_CLEANUP_WHEN_FOUND_EMPTY", "1").lower() in ("1", "true", "yes", "on")

# 自動コミット設定（物件JSONの差分のみ）
AUTO_GIT_ENABLED = os.getenv("AUTO_GIT_ENABLED", "").lower() in ("1", "true", "yes", "on")
AUTO_GIT_COMMIT_MESSAGE = os.getenv("AUTO_GIT_COMMIT_MESSAGE", "Update property data")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 取得する物件数の上限（0 = 制限なし）
MAX_PROPERTIES = 0

# 並列処理設定（高速化版）
CONCURRENT_TABS = 12
TAB_OPEN_DELAY_MIN = 0.02
TAB_OPEN_DELAY_MAX = 0.08
BATCH_WAIT_SECONDS = 0.2
PAGINATION_WAIT_MS = 800
SAVE_INTERVAL = 5
ENABLE_TIMING_LOG = os.getenv("ENABLE_TIMING_LOG", "1").lower() in ("1", "true", "yes", "on")

# 検索結果の上限（イタンジBBの制限）
MAX_SEARCH_RESULTS = 5000

# ページネーション設定
MAX_PAGES = 100  # 本番用: 100ページまで
CONCURRENT_AREAS = 3

# 差分更新モード設定
# Trueにすると、リスト取得時点で既存物件を弾くため超高速になります
UPDATE_MODE = True 

# 既存物件データ更新モード
# Trueにすると、既存物件も詳細ページにアクセスしてJSONデータを更新します（画像は再取得しない）
# 現在は更新を行わず、削除判定のみとする
UPDATE_EXISTING_DATA = False

# 既存物件のメタ情報のみ更新（高速）
# built_date / direction など、欠損のみを最小取得で補完
UPDATE_EXISTING_META_ONLY = True

# 速度重視モード（安全側の最適化のみ実施）
# safe: 従来の挙動 / balanced: 軽量最適化 / aggressive: さらに高速（非推奨）
SPEED_MODE = "balanced"

# 既存データ読込時、個別詳細JSONからのURL補完を行うか
# Trueだと精度は上がるが起動が遅くなる
LOAD_DETAIL_URL_FALLBACK = os.getenv("LOAD_DETAIL_URL_FALLBACK", "0").lower() in ("1", "true", "yes", "on")

# 東京都の23区と市区町村リスト
TOKYO_WARDS = [
    "千代田区", "中央区", "港区", "新宿区", "文京区", "台東区", "墨田区", "江東区",
    "品川区", "目黒区", "大田区", "世田谷区", "渋谷区", "中野区", "杉並区", "豊島区",
    "北区", "荒川区", "板橋区", "練馬区", "足立区", "葛飾区", "江戸川区"
]

# 東京都の市部
TOKYO_CITIES = [
    "八王子市", "立川市", "武蔵野市", "三鷹市", "青梅市", "府中市", "昭島市", "調布市",
    "町田市", "小金井市", "小平市", "日野市", "東村山市", "国分寺市", "国立市", "福生市",
    "狛江市", "東大和市", "清瀬市", "東久留米市", "武蔵村山市", "多摩市", "稲城市", "羽村市",
    "あきる野市", "西東京市"
]

# 東京都の町村部
TOKYO_TOWNS = [
    "瑞穂町", "日の出町", "檜原村", "奥多摩町"
]

# 東京都全域（23区 + 市部 + 町村部）
TOKYO_ALL_AREAS = TOKYO_WARDS + TOKYO_CITIES + TOKYO_TOWNS

# 検索対象
SEARCH_TARGETS = TOKYO_ALL_AREAS  # 本番用: 都内すべての市区町村

# グローバル変数: 中断フラグ
interrupt_flag = False

# 並列ワーカー設定（複数プロセス用）
WORKER_ID = int(os.getenv("SCRAPE_WORKER_ID", "0"))
WORKER_COUNT = int(os.getenv("SCRAPE_WORKER_COUNT", "1"))
OUTPUT_SUFFIX = os.getenv("SCRAPE_OUTPUT_SUFFIX", "").strip()


# ============================================================
# ユーティリティ関数
# ============================================================
def setup_dirs():
    """出力ディレクトリの作成"""
    for d in [OUTPUT_DIR, DATA_DIR, IMAGES_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

def migrate_from_shared_index():
    """旧共有 properties_index.json から itanji データを itanji_index.json に移行"""
    if os.path.exists(INDEX_FILE):
        return  # 既に専用ファイルがある
    if not os.path.exists(LEGACY_SHARED_INDEX):
        return  # 旧ファイルもない
    try:
        with open(LEGACY_SHARED_INDEX, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        if not isinstance(all_data, list):
            return
        my_data = [p for p in all_data if infer_source_from_item(p) in (SOURCE_ID, "")]
        if my_data:
            with open(INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(my_data, f, ensure_ascii=False, indent=2)
            print(f"[migrate] 旧共有インデックスから {len(my_data)}件 を itanji_index.json に移行しました")
    except Exception as e:
        print(f"[migrate] マイグレーション失敗: {e}")

def configure_output_dir(output_dir: str):
    global OUTPUT_DIR, DATA_DIR, IMAGES_DIR, INDEX_FILE, LEGACY_SHARED_INDEX
    OUTPUT_DIR = output_dir
    DATA_DIR = os.path.join(OUTPUT_DIR, "data")
    IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
    INDEX_FILE = os.path.join(OUTPUT_DIR, "itanji_index.json")
    LEGACY_SHARED_INDEX = os.path.join(OUTPUT_DIR, "properties_index.json")

_r2_client = None

def is_r2_ready() -> bool:
    if not R2_UPLOAD_ENABLED:
        return False
    if boto3 is None or Config is None:
        print("[R2] boto3未インストールのためアップロードをスキップします")
        return False
    missing = []
    if not R2_ACCOUNT_ID:
        missing.append("R2_ACCOUNT_ID")
    if not R2_ACCESS_KEY_ID:
        missing.append("R2_ACCESS_KEY_ID")
    if not R2_SECRET_ACCESS_KEY:
        missing.append("R2_SECRET_ACCESS_KEY")
    if missing:
        print(f"[R2] 環境変数が未設定のためアップロードをスキップします: {', '.join(missing)}")
        return False
    return True

def get_r2_client():
    global _r2_client
    if _r2_client is None:
        _r2_client = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
    return _r2_client

def get_content_type(filepath: str) -> str:
    """ファイル拡張子からContent-Typeを推定"""
    lower = filepath.lower()
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    elif lower.endswith(".png"):
        return "image/png"
    elif lower.endswith(".webp"):
        return "image/webp"
    elif lower.endswith(".gif"):
        return "image/gif"
    elif lower.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


def compress_image_file(filepath: str):
    """画像を圧縮してファイルを上書き（COMPRESS_IMAGES有効時のみ）"""
    if not COMPRESS_IMAGES:
        return
    try:
        from PIL import Image
        img = Image.open(filepath)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > COMPRESS_MAX_SIDE:
            ratio = COMPRESS_MAX_SIDE / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        ext = filepath.lower().rsplit(".", 1)[-1]
        if ext in ("jpg", "jpeg"):
            img.save(filepath, "JPEG", quality=COMPRESS_QUALITY, optimize=True)
        elif ext == "webp":
            img.save(filepath, "WEBP", quality=COMPRESS_QUALITY)
        elif ext == "png":
            img.save(filepath, "PNG", optimize=True)
    except Exception:
        pass


def upload_files_to_r2(rel_paths: list[str]):
    """汎用ファイルアップロード（画像/JSON問わず）"""
    if not rel_paths or not is_r2_ready():
        return
    client = get_r2_client()
    uploaded = 0
    error_count = 0
    for rel_path in rel_paths:
        rel_path = rel_path.replace("\\", "/")
        local_file = os.path.join(OUTPUT_DIR, rel_path)
        if not os.path.exists(local_file):
            continue
        content_type = get_content_type(rel_path)
        try:
            client.upload_file(
                local_file, R2_BUCKET_NAME, rel_path,
                ExtraArgs={"ContentType": content_type},
            )
            uploaded += 1
        except Exception as e:
            print(f"[R2] アップロード失敗: {rel_path} - {e}")
            error_count += 1
    if uploaded or error_count:
        print(f"[R2] アップロード完了: {uploaded}件 (エラー {error_count}件)")


def upload_files_to_r2_background(local_paths: list[str]):
    """R2アップロードをバックグラウンドで実行"""
    global _upload_executor_closed
    if not local_paths or interrupt_flag or not R2_UPLOAD_ENABLED:
        return
    if _upload_executor_closed:
        return
    try:
        future = _upload_executor.submit(upload_files_to_r2, local_paths)
        _upload_futures.append(future)
    except RuntimeError as e:
        # executor already shut down
        _upload_executor_closed = True
        if "shutdown" not in str(e).lower():
            print(f"[R2] バックグラウンドアップロード開始失敗: {e}")


def wait_all_uploads():
    """バックグラウンドアップロードの完了を待機"""
    if not _upload_futures:
        return
    pending = [f for f in _upload_futures if not f.done()]
    if not pending:
        _upload_futures.clear()
        return
    print(f"[R2] バックグラウンドアップロード完了待ち ({len(pending)}件)...")
    for f in _upload_futures:
        try:
            # 終了時に未完了を残すと interpreter shutdown で画像アップロード失敗が多発する
            f.result()
        except Exception as e:
            if "shutdown" not in str(e).lower():
                print(f"[R2] バックグラウンドアップロードエラー: {e}")
    _upload_futures.clear()
    print("[R2] バックグラウンドアップロード完了")


def shutdown_upload_executor(wait: bool = False):
    """アップロードキューを安全にシャットダウン"""
    global _upload_executor, _upload_executor_closed
    if _upload_executor_closed:
        return
    _upload_executor_closed = True
    try:
        _upload_executor.shutdown(wait=wait)
    except Exception:
        pass


def download_r2_index_raw() -> list[dict] | None:
    """R2からproperties_index.jsonをダウンロードし、全データを返す。404の場合は空リスト（未作成）として返す。"""
    if not is_r2_ready():
        return []
    try:
        import tempfile
        client = get_r2_client()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()
        client.download_file(R2_BUCKET_NAME, "properties_index.json", tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            data = json.load(f)
        os.remove(tmp.name)
        return data if isinstance(data, list) else []
    except Exception as e:
        err_msg = str(e).lower()
        if "404" in err_msg or "not found" in err_msg:
            print("[R2] インデックス未存在(404) → 新規作成として続行")
            return []
        print(f"[R2] インデックスDL失敗: {e}")
        return None


def infer_source_from_item(item: dict) -> str:
    """source欠損時は detail_url から推定。推定不能なら空文字を返す。"""
    if not isinstance(item, dict):
        return ""
    source = str(item.get("source") or "").strip()
    if source:
        return source
    detail_url = str(item.get("detail_url") or "").strip().lower()
    if not detail_url:
        return ""
    if "itandibb.com" in detail_url or "bukkakun.com" in detail_url or "/rent_rooms/" in detail_url:
        return SOURCE_ID
    if "rent.es-square.net" in detail_url:
        return "es_square"
    return ""


def split_index_by_source(items: list[dict] | None) -> tuple[list[dict], list[dict]]:
    """R2共通インデックスを [itanji, others] に分離（source欠損の推定含む）。"""
    mine: list[dict] = []
    others: list[dict] = []
    for raw in (items or []):
        if not isinstance(raw, dict):
            continue
        inferred = infer_source_from_item(raw)
        item = raw
        if inferred and not raw.get("source"):
            item = dict(raw)
            item["source"] = inferred
        if inferred == SOURCE_ID or item.get("source") == SOURCE_ID:
            mine.append(item)
        else:
            others.append(item)
    return mine, others


def download_index_from_r2() -> bool:
    """R2からインデックスをDLし、itanjiのデータのみローカルに保存"""
    all_data = download_r2_index_raw()
    if all_data is None:
        return False
    if not all_data:
        return False
    my_data, _ = split_index_by_source(all_data)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(my_data, f, ensure_ascii=False, indent=2)
    print(f"[R2] インデックスDL完了: 全体={len(all_data)}件, itanji={len(my_data)}件")
    return True


def upload_merged_index_to_r2() -> None:
    """ローカルのitanjiインデックスとR2上の他スクレイパーデータをマージしてアップロード"""
    if not is_r2_ready():
        return
    try:
        my_data: list[dict] = []
        if os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                local_data = json.load(f)
            if isinstance(local_data, list):
                # ローカルindexはitanji専用ファイルなので、source欠損分はitanjiとして補正
                for raw in local_data:
                    if not isinstance(raw, dict):
                        continue
                    if raw.get("source"):
                        my_data.append(raw)
                    else:
                        fixed = dict(raw)
                        fixed["source"] = SOURCE_ID
                        my_data.append(fixed)
        all_r2_data = download_r2_index_raw()
        if all_r2_data is None:
            print("[R2] マージ中止: R2インデックス取得失敗のため既存データ保護を優先")
            return
        r2_mine, other_data = split_index_by_source(all_r2_data)
        if (not my_data) and r2_mine and (not ALLOW_EMPTY_ITANJI_INDEX_UPLOAD):
            print(
                f"[R2] マージ中止: ローカルitanji=0件 かつ R2 itanji={len(r2_mine)}件。"
                "空アップロード事故防止のためスキップします "
                "(ALLOW_EMPTY_ITANJI_INDEX_UPLOAD=1 で許可)"
            )
            return
        # 重複排除（同一IDの物件は画像データが充実している方を優先）
        def score_entry(p: dict) -> int:
            s = 0
            if isinstance(p.get('local_images'), list) and len(p['local_images']) > 0:
                s += 100 + len(p['local_images'])
            if p.get('thumbnail'):
                s += 50
            ic = p.get('image_count', 0)
            if isinstance(ic, str):
                ic = int(ic) if ic.isdigit() else 0
            s += ic
            if p.get('layout') and str(p['layout']).strip() and str(p['layout']).strip() != '-':
                s += 20
            if p.get('area') and str(p['area']).strip():
                s += 10
            return s

        combined = other_data + my_data
        best_by_id: dict[str, dict] = {}
        for prop in combined:
            pid = prop.get('id')
            if not pid:
                continue
            existing = best_by_id.get(pid)
            if not existing or score_entry(prop) > score_entry(existing):
                best_by_id[pid] = prop
        merged = list(best_by_id.values())
        print(f"[R2] 重複排除: {len(combined)}件 → {len(merged)}件")

        merged_path = os.path.join(OUTPUT_DIR, "_merged_index.json")
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, separators=(',', ':'))
        client = get_r2_client()
        client.upload_file(
            merged_path, R2_BUCKET_NAME, "properties_index.json",
            ExtraArgs={"ContentType": "application/json"},
        )
        os.remove(merged_path)
        print(f"[R2] マージインデックス: itanji={len(my_data)}件 + 他={len(other_data)}件 = {len(merged)}件")
    except Exception as e:
        print(f"[R2] マージインデックスアップロード失敗: {e}")


def upload_images_to_r2(local_paths: list[str]):
    if not local_paths:
        return
    if not is_r2_ready():
        return
    client = get_r2_client()
    uploaded = 0
    error_count = 0
    for rel_path in local_paths:
        rel_path = rel_path.replace("\\", "/")
        local_file = os.path.join(OUTPUT_DIR, rel_path)
        if not os.path.exists(local_file):
            continue
        content_type = "image/jpeg" if rel_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        try:
            client.upload_file(
                local_file,
                R2_BUCKET_NAME,
                rel_path,
                ExtraArgs={"ContentType": content_type},
            )
            uploaded += 1
        except Exception as e:
            print(f"[R2] アップロード失敗: {rel_path} - {e}")
            error_count += 1
    print(f"[R2] アップロード完了: {uploaded}件 (エラー {error_count}件)")

def list_r2_keys(prefix: str) -> list[str]:
    if not is_r2_ready():
        return []
    client = get_r2_client()
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=R2_BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key:
                keys.append(key)
    return keys

def delete_r2_keys(keys: list[str]):
    if not keys:
        return
    if not is_r2_ready():
        return
    client = get_r2_client()
    deleted = 0
    for i in range(0, len(keys), 1000):
        chunk = keys[i:i + 1000]
        try:
            client.delete_objects(
                Bucket=R2_BUCKET_NAME,
                Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
            )
            deleted += len(chunk)
        except Exception as e:
            print(f"[R2] 削除失敗: {e}")
    print(f"[R2] 削除完了: {deleted}件")

def remove_local_image_dir(prop_id: str) -> int:
    if not prop_id:
        return 0
    local_dir = os.path.join(IMAGES_DIR, prop_id)
    if not os.path.exists(local_dir):
        return 0
    removed = 0
    for root, _, files in os.walk(local_dir):
        removed += len(files)
    shutil.rmtree(local_dir, ignore_errors=True)
    return removed


def purge_local_image_artifacts(prop_id: str, local_images=None) -> int:
    """
    local_images に列挙されたファイル削除 + 物件ディレクトリ全削除を必ず行う。
    local_images が古い/欠損していても、最終的に images/<id>/ を残さない。
    """
    removed = 0
    for rel_path in (local_images or []):
        rel_path = rel_path.replace("\\", "/")
        local_file = os.path.join(OUTPUT_DIR, rel_path)
        if os.path.exists(local_file):
            try:
                os.remove(local_file)
                removed += 1
            except Exception:
                pass

    # 残骸対策: 列挙漏れがあってもフォルダごと削除する
    removed += remove_local_image_dir(prop_id)
    return removed

def auto_commit_property_data():
    if not AUTO_GIT_ENABLED:
        return
    try:
        data_dir = os.path.join("scraping", "output", "data")
        if not os.path.exists(os.path.join(REPO_ROOT, data_dir)):
            print(f"[GIT] データディレクトリが見つかりません: {data_dir}")
            return
        subprocess.run(
            ["git", "-C", REPO_ROOT, "add", data_dir],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        status = subprocess.run(
            ["git", "-C", REPO_ROOT, "status", "--porcelain", data_dir],
            check=False,
            capture_output=True,
            text=True,
        )
        if not status.stdout.strip():
            print("[GIT] 変更なし（物件JSON）")
            return
        subprocess.run(
            ["git", "-C", REPO_ROOT, "commit", "-m", AUTO_GIT_COMMIT_MESSAGE],
            check=False,
        )
        subprocess.run(
            ["git", "-C", REPO_ROOT, "push"],
            check=False,
        )
        print("[GIT] 物件JSONをコミット＆プッシュしました")
    except Exception as e:
        print(f"[GIT] 自動コミット失敗: {e}")

def split_targets_for_worker(targets: list[str], worker_id: int, worker_count: int) -> list[str]:
    if worker_count <= 1:
        return targets
    return [t for i, t in enumerate(targets) if i % worker_count == worker_id]

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]', '_', name).strip('_')

def split_title(title: str) -> tuple[str, str]:
    title = title.strip()
    m = re.search(r'(.+?)\s*([0-9]{2,4})\s*号室?$', title)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m2 = re.search(r'(.+?)([0-9]{2,4})$', title)
    if m2:
        return m2.group(1).strip(), m2.group(2).strip()
    return title, ""

def normalize_url(url: str) -> str:
    """URLを厳密に正規化（http/https無視、クエリ無視、末尾スラッシュ無視）"""
    if not url: return ""
    try:
        # ドメイン以下を取得
        parsed = urlparse(url)
        path = parsed.path.rstrip('/') # 末尾スラッシュ削除
        return path # パスだけで比較する（https://itandibb.com 部分は無視）
    except:
        return url


def build_url_dict_from_local_detail_json(items: list[dict], by_url: dict):
    """detail_url欠損が多い時の補完。data/{id}.json の detail_url を使ってURL辞書を埋める。"""
    for item in items:
        item_id = item.get("id")
        if not item_id:
            continue
        full_path = os.path.join(DATA_DIR, f"{item_id}.json")
        if not os.path.exists(full_path):
            continue
        try:
            with open(full_path, "r", encoding="utf-8") as jf:
                full = json.load(jf)
            detail_url = full.get("detail_url")
            if detail_url:
                by_url[detail_url] = full
        except Exception:
            continue

def fully_render(page):
    try:
        page.evaluate("""
        [...document.querySelectorAll('button,[role="button"]')].forEach(b=>{
            const t=(b.innerText||'');
            if(/設備|費用|周辺|地図|備考|詳細|内見|所在地|その他|条件/.test(t)){
                try{ if(b.getAttribute('aria-expanded')==='false'){b.click();}}catch(e){}
            }
        });
        """)
    except:
        pass


def wait_for_detail_content(page, timeout_ms: int = 5000):
    """賃料・間取り等の詳細ブロックがDOMに出現するまで待つ（データ不足を減らす）"""
    try:
        page.wait_for_selector(
            ".DetailTable, .Block .ItemValue, span.BuildingName, [class*='ItemName']",
            timeout=timeout_ms,
            state="attached",
        )
    except Exception:
        pass
    try:
        page.wait_for_timeout(350)
    except Exception:
        pass

def configure_context(context):
    if SPEED_MODE == "safe":
        return

    def route_handler(route, request):
        rtype = request.resource_type
        if rtype in ("font", "media"):
            return route.abort()
        if SPEED_MODE == "aggressive" and rtype in ("image", "stylesheet"):
            return route.abort()
        return route.continue_()

    try:
        context.route("**/*", route_handler)
    except:
        pass

def is_valid_property_data(data: dict) -> bool:
    """タイトル・住所・賃料・間取りが揃っているか（保存可否の厳格判定用）"""
    if not data:
        return False
    title = data.get("title") or ""
    address = data.get("address") or ""
    rent = data.get("rent") or ""
    layout = data.get("layout") or ""
    if title.strip() in ("", "物件名不明"):
        return False
    if not address.strip():
        return False
    if not rent.strip():
        return False
    if not layout.strip():
        return False
    return True


def is_sufficient_for_save(data: dict) -> bool:
    """問題なし（家賃・画像・間取り・住所・駅がすべてある）場合のみTrue。これに満たない物件は保存しない。"""
    if not is_valid_property_data(data):
        return False
    stations = data.get("stations") or []
    if isinstance(stations, str) and stations.strip():
        stations = [s.strip() for s in re.split(r"[\n,、]", stations) if s.strip()]
    elif not isinstance(stations, list):
        stations = []
    if not stations:
        return False
    image_urls = data.get("image_urls") or []
    if not image_urls or len(image_urls) == 0:
        return False
    return True

def extract_meta_from_spans(page):
    """詳細画面のspanテキストから築年・向きを拾う（ラベルが無いケース対策）"""
    built_date = ""
    direction = ""
    try:
        spans = page.query_selector_all("span")
        for span in spans:
            text = span.inner_text().strip()
            if not text:
                continue
            if not built_date:
                m = re.search(r'(?:19|20)\d{2}年\d{1,2}月(?:\s*\(築\d{1,3}年\))?', text)
                if m:
                    built_date = m.group(0)
            if not direction:
                m2 = re.search(r'(南東|南西|北東|北西|南|北|東|西)向き', text)
                if m2:
                    direction = m2.group(0)
            if built_date and direction:
                break
    except:
        pass
    return built_date, direction


def build_detail_map(page) -> dict:
    """詳細テーブルを1回走査して {ラベル: 値} を作る（高速化用）"""
    try:
        pairs = page.evaluate(
            """
            () => {
              const out = [];

              // 方法1: .DetailTable コンテナから .ItemName/.ItemValue を取得
              // HTML構造: <div class="DetailTable"><div class="Block ItemName">...</div><div class="Block ItemValue">...</div></div>
              // 注意: .Block と .ItemName/.ItemValue は同じ要素のクラスなので、
              //       .DetailTable をコンテナとして選択し、その中の .ItemName/.ItemValue を探す
              const tables = document.querySelectorAll('.DetailTable');
              for (const table of tables) {
                const nameEl = table.querySelector('.ItemName');
                const valueEl = table.querySelector('.ItemValue');
                if (!nameEl || !valueEl) continue;
                const name = (nameEl.innerText || nameEl.textContent || '').replace(/\\s+/g, ' ').trim();
                // innerText を使用して block 要素間にスペースを挿入（所在地の住所・路線が結合されるのを防止）
                const value = (valueEl.innerText || valueEl.textContent || '').replace(/[\\t ]+/g, ' ').replace(/\\n\\s*\\n/g, '\\n').trim();
                if (!name) continue;
                out.push([name, value]);
              }

              // 方法1b: 所在地フィールドから交通情報（路線・駅）を別キーで抽出
              // HTML構造: 所在地の .ItemValue 内に .Flex 要素があり、駅情報が個別divに入っている
              for (const table of tables) {
                const nameEl = table.querySelector('.ItemName');
                if (!nameEl) continue;
                const label = (nameEl.textContent || '').trim();
                if (label !== '所在地' && label !== '住所') continue;
                const valueEl = table.querySelector('.ItemValue');
                if (!valueEl) continue;
                const flexDivs = valueEl.querySelectorAll('.Flex, [class*="Flex"]');
                const stationLines = [];
                for (const flex of flexDivs) {
                  const text = (flex.innerText || flex.textContent || '').trim();
                  if (text && (text.includes('駅') && (text.includes('分') || text.includes('線')))) {
                    stationLines.push(text);
                  }
                }
                if (stationLines.length > 0) {
                  out.push(['交通', stationLines.join('\\n')]);
                }
                break;
              }

              // 方法2: 設備・詳細セクション（itandi-bb-ui__Flex + itandi-bb-ui__Grid 構造）
              // HTML構造: <div class="itandi-bb-ui__Flex"><div class="itandi-bb-ui__Grid">ラベル</div><div class="itandi-bb-ui__Flex">値</div></div>
              const flexRows = document.querySelectorAll('[class*="itandi-bb-ui__Flex"]');
              for (const row of flexRows) {
                const children = row.children;
                if (children.length < 2) continue;
                const labelEl = children[0];
                const valueEl = children[1];
                const labelClass = labelEl.className || '';
                if (!labelClass.includes('itandi-bb-ui__Grid')) continue;
                const name = (labelEl.innerText || labelEl.textContent || '').replace(/\\s+/g, ' ').trim();
                const value = (valueEl.innerText || valueEl.textContent || '').replace(/\\s+/g, ' ').trim();
                if (!name || !value) continue;
                out.push([name, value]);
              }

              // 方法3: フォールバック - .Block 親要素パターン（旧構造互換）
              if (out.length === 0) {
                const blocks = document.querySelectorAll('.Block');
                for (const block of blocks) {
                  const nameEl = block.querySelector('.ItemName');
                  const valueEl = block.querySelector('.ItemValue');
                  if (!nameEl || !valueEl) continue;
                  const name = (nameEl.innerText || nameEl.textContent || '').replace(/\\s+/g, ' ').trim();
                  const value = (valueEl.innerText || valueEl.textContent || '').replace(/\\s+/g, ' ').trim();
                  if (!name) continue;
                  out.push([name, value]);
                }
              }

              // 方法4: 賃料・間取り・建物名を span/p/div から取得（長文要素内でもパターン検索）
              const keys = new Set(out.map(x => x[0]));
              const allEls = document.querySelectorAll('span, p, div, td, th');
              const rentRe = /賃料\\s*[:：]?\\s*([\\d.]+)\\s*万\\s*円?/;
              const layoutReExact = /^\\s*(\\d[SLDKR]+|ワンルーム|1R|スタジオ|[０-９][SLDKR]+)\\s*$/;
              const layoutReAny = /(\\d[SLDKR]+|ワンルーム|1R|スタジオ|[０-９][SLDKR]+)/;
              const buildingRe = /(建物名|物件名)\\s*[:：]?\\s*([^\\n]+)/;
              for (const el of allEls) {
                const txt = (el.textContent || '').trim();
                if (!txt) continue;
                if (!keys.has('賃料')) {
                  const m = txt.match(rentRe);
                  if (m) { out.push(['賃料', m[1] + '万円']); keys.add('賃料'); }
                }
                if (!keys.has('間取り')) {
                  let lm = txt.match(layoutReExact);
                  if (!lm && txt.length <= 150) lm = txt.match(layoutReAny);
                  if (lm) { out.push(['間取り', lm[1]]); keys.add('間取り'); }
                }
                if (!keys.has('建物名') && !keys.has('物件名') && txt.length <= 80) {
                  const bm = txt.match(buildingRe);
                  if (bm) { out.push(['建物名', bm[2].trim()]); keys.add('建物名'); }
                }
                if (!keys.has('構造')) {
                  const sm = txt.match(/構造\\s*[:：]?\\s*([^\\n/]+)/);
                  if (sm) { out.push(['構造', sm[1].trim()]); keys.add('構造'); }
                }
                if (!keys.has('所在階')) {
                  const fm = txt.match(/所在階\\s*[:：]?\\s*(\\d+階)/);
                  if (fm) { out.push(['所在階', fm[1].trim()]); keys.add('所在階'); }
                }
              }

              // 方法5: 「賃料: 14.5万円/管理費: 5,000円/共益費: 入力なし」形式の1要素から分割して取得
              if (!keys.has('賃料') || !keys.has('管理費')) {
                for (const el of allEls) {
                  const txt = (el.textContent || '').trim();
                  if (!txt || !txt.includes('賃料') || !txt.includes('/')) continue;
                  const parts = txt.split(/\\s*\\/\\s*/);
                  for (const part of parts) {
                    const p = part.trim();
                    if (!keys.has('賃料') && /賃料\\s*[:：]/.test(p)) {
                      const m = p.match(rentRe);
                      if (m) { out.push(['賃料', m[1] + (m[0].includes('円') ? '万円' : '万')]); keys.add('賃料'); break; }
                    }
                    if (!keys.has('管理費') && /管理費\\s*[:：]/.test(p)) {
                      const m = p.match(/管理費\\s*[:：]?\\s*([\\d,]+)\\s*円/);
                      if (m) { out.push(['管理費', m[1].replace(/,/g,'') + '円']); keys.add('管理費'); }
                    }
                    if (!keys.has('共益費') && /共益費\\s*[:：]/.test(p)) {
                      const m = p.match(/共益費\\s*[:：]?\\s*(.+)/);
                      if (m) { out.push(['共益費', m[1].trim()]); keys.add('共益費'); }
                    }
                  }
                  if (keys.has('賃料') && keys.has('管理費')) break;
                }
              }

              return out;
            }
            """
        )
        result = {}
        for name, value in pairs or []:
            if not name:
                continue
            if name in result and value:
                if value not in result[name]:
                    result[name] = f"{result[name]} / {value}"
            else:
                result[name] = value or ""

        # 複合フィールドを個別キーに分解
        # 例: "賃料: 14.5万円/管理費: 5,000円/共益費: 入力なし" → 賃料=14.5万円, 管理費=5,000円, 共益費=入力なし
        # 既存キーがあっても / 区切りの部分でラベル:値があれば上書き（正しい単一値を入れる）
        for _name, value in list(result.items()):
            if not value or (':' not in value and '：' not in value):
                continue
            parts = re.split(r'\s*/\s*', value)
            for part in parts:
                m = re.match(r'(.+?)\s*[:：]\s*(.+)', part.strip())
                if m:
                    sub_name = m.group(1).strip()
                    sub_value = m.group(2).strip()
                    if sub_name and sub_value:
                        result[sub_name] = sub_value

        return result
    except:
        return {}


# ============================================================
# 詳細情報抽出関数 (変更なし)
# ============================================================
def extract_layout_area_from_dom(page) -> tuple:
    """DOMから直接間取り・面積を取得（DetailTable構造に依存しない）"""
    try:
        result = page.evaluate(
            """
            () => {
              let layout = '';
              let area = '';
              // 全span, p, div要素から間取りパターンを検索
              const allEls = document.querySelectorAll('span, p, div, td, th');
              const layoutRe = /^\\s*(\\d[SLDKR]+|ワンルーム|1R|スタジオ|[０-９][SLDKR]+)\\s*$/;
              const layoutReInText = /(\\d[SLDKR]+|ワンルーム|1R|スタジオ|[０-９][SLDKR]+)/;
              const layoutAreaRe = /^\\s*(\\d[SLDKR]+|ワンルーム|1R|スタジオ|[０-９][SLDKR]+)\\s*[\/／]\\s*([\\d.]+)\\s*[㎡m²]/;
              const layoutAreaInText = /(\\d[SLDKR]+|ワンルーム|1R|スタジオ|[０-９][SLDKR]+)\\s*[\/／]\\s*([\\d.]+)\\s*[㎡m²]/;
              const areaRe = /^\\s*([\\d.]+)\\s*[㎡m²]\\s*$/;
              for (const el of allEls) {
                const txt = (el.textContent || '').trim();
                if (!txt) continue;
                const short = txt.length <= 50;
                if (short) {
                  const combo = txt.match(layoutAreaRe);
                  if (combo) { layout = combo[1]; area = combo[2]; break; }
                  const lm = txt.match(layoutRe);
                  if (lm) layout = lm[1];
                  const am = txt.match(areaRe);
                  if (am) area = am[1];
                } else {
                  if (!layout || !area) {
                    const combo = txt.match(layoutAreaInText);
                    if (combo) { layout = combo[1]; area = combo[2]; }
                    if (!layout) { const lm = txt.match(layoutReInText); if (lm) layout = lm[1]; }
                  }
                }
              }
              return [layout, area];
            }
            """
        )
        return (result[0] or '', result[1] or '') if result else ('', '')
    except:
        return ('', '')


def extract_property_details(page, body_text: str) -> dict:
    detail_map = build_detail_map(page)

    def clean_address_text(text: str) -> str:
        if not text:
            return ""
        lines = [ln.strip() for ln in re.split(r'[\r\n]+', text) if ln and ln.strip()]
        cleaned = []
        for line in lines:
            # UI補助テキストを除去
            line = re.sub(r'\s*地図\s*', ' ', line).strip()
            if not line:
                continue
            if "沿線駅" in line:
                continue
            # 駅・路線アクセス行は住所から除外（stationsで別管理）
            if re.search(r'駅', line) and re.search(r'徒歩|バス|線', line):
                continue
            cleaned.append(line)
        # 重複行を除去（順序維持）
        uniq = list(dict.fromkeys(cleaned))
        return "\n".join(uniq).strip()

    def get_detail_value(label_text: str) -> str:
        try:
            exact = detail_map.get(label_text)
            if exact:
                return exact
            for k, v in detail_map.items():
                if label_text in k:
                    return v
        except:
            pass
        return ""

    def get_detail_value_direct(label_text: str) -> str:
        """6日前実装のフォールバック: ラベル要素から同じDetailTable内の値を直接取得"""
        try:
            labels = page.query_selector_all(".ItemName")
            for label in labels:
                label_txt = (label.inner_text() or "").strip()
                if not label_txt or label_text not in label_txt:
                    continue
                parent = label.evaluate_handle("el => el.parentElement")
                parent_el = parent.as_element() if parent else None
                if not parent_el:
                    continue
                value_el = parent_el.query_selector(".ItemValue")
                if not value_el:
                    continue
                value_txt = (value_el.inner_text() or "").strip()
                if value_txt:
                    return value_txt
        except Exception:
            pass
        return ""

    def normalize_money_text(value: str) -> str:
        if not value:
            return ""
        normalized = value.replace('\u3000', ' ').strip()
        if normalized in ("-", "ー", "―", "入力なし", "未入力"):
            return ""
        return normalized

    def extract_money_value(label_text: str) -> str:
        raw = normalize_money_text(get_detail_value(label_text))
        if raw:
            return raw
        patterns = [
            rf"{label_text}\s*[:：]?\s*([0-9.]+)\s*ヶ?月",
            rf"{label_text}\s*[:：]?\s*([0-9,]+)\s*円",
            rf"{label_text}\s*[:：]?\s*(なし|無|不要|ゼロ|0円)",
        ]
        for pat in patterns:
            m = re.search(pat, body_text)
            if m:
                return m.group(1)
        return ""

    # タイトルは複数ソースで補完（物件名不明の抑制）
    # 優先: <span class="BuildingName HeaderText">ヴィラ和田 301</span> → .building-name → h1
    title = ""
    try:
        for selector in ["span.BuildingName", "span[class*='BuildingName']", ".building-name", "h1"]:
            title_elem = page.query_selector(selector)
            if title_elem:
                t = (title_elem.inner_text() or "").strip()
                if t:
                    title = t
                    break
    except Exception:
        pass
    if not title:
        title = (
            get_detail_value("物件名")
            or get_detail_value("建物名")
            or get_detail_value("部屋番号")
            or ""
        ).strip()
    if not title:
        m_title = re.search(r"([^\n]+(?:マンション|アパート|レジデンス|ハイツ|コーポ|コート|タワー)[^\n]*)", body_text)
        if m_title:
            title = m_title.group(1).strip()
    if not title:
        try:
            page_title = (page.title() or "").strip()
            for sep in [" | ", " - ", "｜", "－", " – "]:
                if sep in page_title:
                    page_title = page_title.split(sep)[0].strip()
            if page_title and re.search(r"(?:マンション|アパート|レジデンス|ハイツ|コーポ|コート|タワー|号室)", page_title) and len(page_title) <= 80:
                title = page_title
        except Exception:
            pass
    if not title:
        try:
            h1_text = page.evaluate("() => { const h = document.querySelector('h1'); return h ? (h.innerText || h.textContent || '').trim() : ''; }")
            if h1_text and len(h1_text) <= 80:
                title = h1_text
        except Exception:
            pass
    if not title:
        title = "物件名不明"
    building_name, room_number = split_title(title)
    address = get_detail_value("所在地") or get_detail_value("住所") or ""
    # 6日前実装の取得方式を併用（detail_mapが外すケースを救済）
    direct_address = get_detail_value_direct("所在地") or get_detail_value_direct("住所") or ""
    if direct_address and (
        not address or
        (re.match(r'^[〒\s\d\-]+$', address.strip()) and len(direct_address) > len(address))
    ):
        address = direct_address
    # 所在地が郵便番号のみ（〒と数字・ハイフンのみ）のときは、他ラベル・detail_map・body_text から住所を補う
    if address and re.match(r'^[〒\s\d\-]+$', address.strip()):
        addr_alt = get_detail_value("住所") or get_detail_value("所在地") or ""
        if not addr_alt or addr_alt == address:
            for _k, _v in (detail_map.items() if detail_map else []):
                if not _v or _v.strip() == address.strip():
                    continue
                if _k in ("交通", "アクセス"):
                    continue
                if re.search(r'[都道府県]', _v) and re.search(r'[市区町村丁目]', _v) and "駅" not in _v:
                    addr_alt = _v.strip()
                    break
        if not addr_alt or addr_alt == address:
            # ページ全文から: 〒の直後の行、または都道府県+市区町村を含む行を1行取得
            m_zip_line = re.search(r'〒\s*\d{3}-?\d{4}\s*([^\n]+)', body_text)
            if m_zip_line:
                rest = m_zip_line.group(1).strip()
                if re.search(r'[都道府県]', rest):
                    addr_alt = rest
            if not addr_alt:
                m_addr_line = re.search(r'([都道府県][^\n]*(?:区|市|町|村)[^\n]*(?:丁目|番地|\d+-?\d*)[^\n]*)', body_text)
                if m_addr_line:
                    addr_alt = m_addr_line.group(1).strip()
                elif body_text:
                    for line in body_text.split('\n'):
                        line = line.strip()
                        if re.search(r'^[都道府県]', line) and re.search(r'[区市町村]', line) and '駅' not in line and len(line) < 80:
                            addr_alt = line
                            break
        if not addr_alt or addr_alt == address:
            # 最終フォールバック: DOM の所在地テーブルから住所行を直接拾う
            try:
                addr_from_dom = page.evaluate(
                    """
                    () => {
                      const tables = document.querySelectorAll('.DetailTable');
                      for (const table of tables) {
                        const nameEl = table.querySelector('.ItemName');
                        const valueEl = table.querySelector('.ItemValue');
                        if (!nameEl || !valueEl) continue;
                        const label = (nameEl.innerText || nameEl.textContent || '').trim();
                        if (label !== '所在地' && label !== '住所') continue;

                        const rows = valueEl.querySelectorAll('.Flex');
                        let fallback = '';
                        for (const row of rows) {
                          const text = (row.innerText || row.textContent || '')
                            .replace(/\\s*地図\\s*/g, ' ')
                            .replace(/\\s+/g, ' ')
                            .trim();
                          if (!text) continue;

                          const noZip = text.replace(/〒\\s*\\d{3}-?\\d{4}/g, '').trim();
                          if (!noZip) continue;
                          if (/駅|徒歩|バス|線/.test(noZip)) continue;
                          if (!fallback) fallback = noZip;
                          if (/[都道府県]/.test(noZip)) return noZip;
                        }
                        if (fallback) return fallback;
                      }
                      return '';
                    }
                    """
                )
                if addr_from_dom:
                    addr_alt = str(addr_from_dom).strip()
            except Exception:
                pass
        if addr_alt and addr_alt != address:
            address = f"{address.strip()} {addr_alt.strip()}".strip()
    if not address:
        m_addr = re.search(r'所在地\s*[:：]?\s*([^\n]+)', body_text) or re.search(r'住所\s*[:：]?\s*([^\n]+)', body_text)
        if m_addr:
            address = m_addr.group(1).strip()
        if not address:
            m_zip = re.search(r'〒\s*\d{3}-?\d{4}\s*([^\n]+)', body_text)
            if m_zip:
                address = m_zip.group(1).strip()
    # 住所には地図・沿線駅・駅アクセス行を含めない
    address = clean_address_text(address)
    
    stations = []
    access_raw = get_detail_value("交通") or get_detail_value("アクセス") or ""
    if access_raw:
        for line in access_raw.split('\n'):
            line = line.strip()
            if line and ('駅' in line or '線' in line) and len(stations) < 3:
                stations.append(line)
    if not stations and body_text:
        for line in body_text.split('\n'):
            line = line.strip()
            if line and ('駅' in line and ('線' in line or '分' in line)) and len(stations) < 3:
                stations.append(line)
    
    rent = ""
    rent_raw = get_detail_value("賃料")
    if rent_raw:
        man_match = re.search(r'([\d.]+)\s*万円', rent_raw) or re.search(r'([\d.]+)\s*万\s*$', rent_raw) or re.search(r'([\d.]+)\s*万(?!\s*円)', rent_raw)
        if man_match:
            rent = str(int(float(man_match.group(1)) * 10000))
        else:
            yen_match = re.search(r'([\d,]+)\s*円', rent_raw)
            if yen_match:
                rent = yen_match.group(1).replace(',', '')
    if not rent:
        m_man = (
            re.search(r'賃料\s*[:：]?\s*([\d.]+)\s*万\s*円', body_text)
            or re.search(r'賃料\s*[:：]?\s*([\d.]+)\s*万円', body_text)
            or re.search(r'家賃\s*[:：]?\s*([\d.]+)\s*万\s*円', body_text)
            or re.search(r'家賃\s*[:：]?\s*([\d.]+)\s*万円', body_text)
        )
        if m_man:
            rent = str(int(float(m_man.group(1)) * 10000))
        else:
            m_yen = re.search(r'賃料\s*[:：]?\s*([0-9,]+)\s*円', body_text) or re.search(r'家賃\s*[:：]?\s*([0-9,]+)\s*円', body_text)
            if m_yen:
                rent = m_yen.group(1).replace(',', '')
        if not rent:
            m_early = re.search(r'([\d.]+)\s*万\s*円', body_text[:3500]) or re.search(r'([\d.]+)\s*万円', body_text[:3500])
            if m_early:
                val = float(m_early.group(1))
                if 0.5 <= val <= 500:
                    rent = str(int(val * 10000))

    management_fee = ""
    mgmt_raw = get_detail_value("管理費") or get_detail_value("共益費")
    if mgmt_raw:
        man_match = re.search(r'([\d.]+)\s*万円', mgmt_raw)
        if man_match: management_fee = str(int(float(man_match.group(1)) * 10000))
        else:
            yen_match = re.search(r'([\d,]+)\s*円', mgmt_raw)
            if yen_match: management_fee = yen_match.group(1).replace(',', '')
        if not management_fee and "入力なし" in mgmt_raw:
            management_fee = ""
    if not management_fee and body_text:
        m_mgmt = re.search(r'管理費\s*[:：]?\s*([0-9,]+)\s*円', body_text) or re.search(r'共益費\s*[:：]?\s*([0-9,]+)\s*円', body_text)
        if m_mgmt: management_fee = m_mgmt.group(1).replace(',', '')

    try:
        image_urls = page.evaluate(
            """
            () => {
              const urls = new Set();
              // 方法1: 従来の property-images パス
              const nodes = document.querySelectorAll("img[src*='property-images']");
              for (const img of nodes) {
                const src = img.getAttribute('src');
                if (src) urls.add(src);
              }
              // 方法2: フォールバック - property / image を含むsrc、または data-src（遅延読込）
              if (urls.size === 0) {
                const all = document.querySelectorAll("img[src*='property'], img[src*='image'], img[data-src]");
                for (const img of all) {
                  const src = img.getAttribute('src') || img.getAttribute('data-src');
                  if (src && (src.startsWith('http') || src.startsWith('//'))) urls.add(src);
                }
              }
              return Array.from(urls);
            }
            """
        ) or []
    except Exception:
        image_urls = []

    # 築年月/向きの表記ゆれを補完（ラベル取得に失敗するケース対策）
    built_date = (
        get_detail_value("築年月") or
        get_detail_value("築年") or
        get_detail_value("建築年月") or
        get_detail_value("築年数") or
        ""
    )
    if not built_date:
        m = re.search(r'\b(20\d{2}|19\d{2})年\d{1,2}月\b', body_text)
        if m:
            built_date = m.group(0)
        else:
            m2 = re.search(r'築\d{1,2}年', body_text)
            if m2:
                built_date = m2.group(0)
    if not built_date:
        span_built_date, _ = extract_meta_from_spans(page)
        if span_built_date:
            built_date = span_built_date

    direction = get_detail_value("向き") or get_detail_value("方角") or get_detail_value("主要採光面") or ""
    if not direction:
        m3 = re.search(r'(南東|南西|北東|北西|南|北|東|西)向き', body_text)
        if m3:
            direction = m3.group(0)
    if not direction:
        _, span_direction = extract_meta_from_spans(page)
        if span_direction:
            direction = span_direction

    deposit = extract_money_value("敷金")
    key_money = extract_money_value("礼金")

    # 間取り・面積をDOM直接取得（DetailTable外のspan等から取得）
    dom_layout, dom_area = extract_layout_area_from_dom(page)

    # 間取り: 1K/1DK/1LDK/2K/2DK/2LDK/3LDK, ワンルーム, 1R, スタジオ, 全角１２３ 等をすべて対象
    _layout_pat = r'([0-9０-９][SLDK]+[0-9０-９]*|ワンルーム|1R|１R|スタジオ)'
    layout = (
        re.search(_layout_pat, get_detail_value("間取り") or "")
        or re.search(_layout_pat, get_detail_value("間取") or "")
        or re.search(_layout_pat, get_detail_value("タイプ") or "")
        or (re.search(_layout_pat, dom_layout) if dom_layout else None)
        or re.search(r'間取り\s*[:：]?\s*' + _layout_pat, body_text)
        or re.search(_layout_pat, body_text[:8000])
        or re.match("", "")
    ).group(0)

    # 面積: DetailMap → DOM直接 → body_text の優先順でフォールバック
    area = (
        re.search(r'([\d.]+)', get_detail_value("専有面積") or "")
        or re.search(r'([\d.]+)', get_detail_value("面積") or "")
        or (re.search(r'([\d.]+)', dom_area) if dom_area else None)
        or re.search(r'専有面積\s*[:：]?\s*([\d.]+)', body_text)
        or re.search(r'([0-9０-９][SLDK]+[0-9０-９]*|ワンルーム|1R|１R|スタジオ)\s*[\/／]\s*([\d.]+)\s*[㎡m²]', body_text)
        or re.match("", "")
    ).group(0) if True else ""

    floor_val = get_detail_value("所在階") or get_detail_value("階") or ""
    if not floor_val and body_text:
        m_floor = re.search(r'所在階\s*[:：]?\s*(\d+階)', body_text)
        if m_floor: floor_val = m_floor.group(1).strip()
    structure_val = get_detail_value("構造") or ""
    if not structure_val and body_text:
        m_struct = re.search(r'構造\s*[:：]?\s*([^\n/]+)', body_text)
        if m_struct: structure_val = m_struct.group(1).strip()

    return {
        "title": title, "building_name": building_name, "room_number": room_number,
        "address": address, "stations": stations, "rent": rent,
        "management_fee": management_fee,
        "deposit": deposit, "key_money": key_money,
        "renewal_fee": get_detail_value("更新料") or "",
        "insurance": get_detail_value("火災保険") or get_detail_value("保険") or "",
        "layout": layout,
        "area": area,
        "built_date": built_date,
        "floor": floor_val,
        "structure": structure_val,
        "direction": direction or "",
        "available_date": get_detail_value("入居可能時期") or get_detail_value("入居可能日") or "",
        "contract_period": get_detail_value("賃貸借契約期間") or get_detail_value("契約期間") or "",
        "parking": get_detail_value("駐車場") or "",
        "building_type": get_detail_value("建物種別") or "",
        "total_units": get_detail_value("総戸数") or "",
        "contract_type": get_detail_value("賃貸借契約区分") or "",
        "facilities": [k for k in [
            # セキュリティ
            'オートロック', 'モニター付きインターホン', 'TVモニターホン', '防犯カメラ', 'ディンプルキー',
            # 水回り
            'バス・トイレ別', 'バストイレ別', '独立洗面台', '温水洗浄便座', 'ウォシュレット', '追い焚き', '追焚き', '浴室乾燥機', '洗面化粧台',
            # キッチン
            'システムキッチン', 'ガスコンロ', 'IHコンロ', '2口コンロ', '3口コンロ', 'カウンターキッチン', '食洗機', '食器洗い乾燥機',
            # 収納・室内設備
            'ウォークインクローゼット', 'WIC', 'クローゼット', 'シューズボックス', 'ロフト', '床下収納',
            # 設備
            'エアコン', '室内洗濯機置場', '浴室乾燥機', 'インターネット無料', 'ネット無料', '光ファイバー', 'CATV', 'BS', 'CS',
            # 構造・環境
            'フローリング', 'バルコニー', 'ベランダ', 'ルーフバルコニー', '専用庭', '南向き', '角部屋', '最上階', '2階以上',
            # その他
            'ペット可', 'ペット相談', '楽器可', '楽器相談', 'エレベーター', 'EV', '宅配ボックス', '駐輪場', '駐車場',
            '24時間ゴミ出し', '管理人', 'コンシェルジュ', 'オール電化', 'プロパンガス', '都市ガス', 
            'デザイナーズ', 'リノベーション', 'リフォーム済', '新築', '築浅',
            '女性限定', '単身者限定', '二人入居可', 'ルームシェア可', '事務所利用可', 'SOHO可'
        ] if k in body_text],
        "ad_fee": get_detail_value("広告費") or get_detail_value("AD") or "",
        "transaction_type": get_detail_value("取引態様") or "",
        "guarantee_company": get_detail_value("保証会社") or "",
        "remarks": (get_detail_value("備考") or "")[:500],
        # 希望条件（大家の希望・入居条件など。サイトのラベルに応じて複数試行）
        "preferred_conditions": (
            get_detail_value("希望条件") or get_detail_value("大家の希望条件") or
            get_detail_value("大家の希望") or get_detail_value("入居条件") or
            get_detail_value("貸主の希望") or get_detail_value("希望する入居者") or ""
        )[:1000],
        # PDF構成に合わせた追加項目
        "viewing_start_date": get_detail_value("内見開始日") or get_detail_value("内見開始") or "",
        "viewing_notes": get_detail_value("内見時注意事項") or "",
        "image_urls": image_urls, "image_count": len(image_urls),
    }

def extract_property_meta_light(page, body_text: str) -> dict:
    """既存物件のメタ情報のみを軽量取得（高速更新用）"""
    detail_map = build_detail_map(page)

    def get_detail_value(label_text: str) -> str:
        try:
            exact = detail_map.get(label_text)
            if exact:
                return exact
            for k, v in detail_map.items():
                if label_text in k:
                    return v
        except:
            pass
        return ""

    built_date = (
        get_detail_value("築年月") or
        get_detail_value("築年") or
        get_detail_value("建築年月") or
        get_detail_value("築年数") or
        ""
    )
    if not built_date:
        m = re.search(r'\b(20\d{2}|19\d{2})年\d{1,2}月\b', body_text)
        if m:
            built_date = m.group(0)
        else:
            m2 = re.search(r'築\d{1,2}年', body_text)
            if m2:
                built_date = m2.group(0)
    if not built_date:
        span_built_date, _ = extract_meta_from_spans(page)
        if span_built_date:
            built_date = span_built_date

    direction = get_detail_value("向き") or get_detail_value("方角") or get_detail_value("主要採光面") or ""
    if not direction:
        m3 = re.search(r'(南東|南西|北東|北西|南|北|東|西)向き', body_text)
        if m3:
            direction = m3.group(0)
    if not direction:
        _, span_direction = extract_meta_from_spans(page)
        if span_direction:
            direction = span_direction

    return {
        "built_date": built_date,
        "direction": direction
    }

def download_single_image(args):
    url, property_id, idx = args
    if not url or url.startswith('data:'): return None
    try:
        if url.startswith('//'): url = 'https:' + url
        elif url.startswith('/'): url = "https://itandibb.com" + url
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            ext = '.png' if 'png' in response.headers.get('Content-Type', '') else '.jpg'
            property_dir = os.path.join(IMAGES_DIR, property_id)
            os.makedirs(property_dir, exist_ok=True)
            filename = f"{str(idx+1).zfill(2)}{ext}"
            filepath = os.path.join(property_dir, filename)
            with open(filepath, "wb") as f: f.write(response.content)
            compress_image_file(filepath)
            return f"images/{property_id}/{filename}"
    except: pass
    return None

def download_images(property_id: str, image_urls: list[str]) -> tuple[list[str], list[str]]:
    local_paths = []
    property_dir = os.path.join(IMAGES_DIR, property_id)
    
    # 既存の画像ファイルをチェック（フォルダが存在し、かつ画像ファイルが実在する場合のみスキップ）
    if os.path.exists(property_dir):
        existing_files = [f for f in os.listdir(property_dir) if f.endswith(('jpg','png','webp'))]
        if existing_files:
            existing = [f"images/{property_id}/{f}" for f in sorted(existing_files)]
            return existing, []
    
    if not image_urls:
        return [], []
    
    # 高速化: max_workers 8→15（ネットワークI/Oなので安全に増やせる）
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(download_single_image, [(u, property_id, i) for i, u in enumerate(image_urls)]))
    local_paths = [r for r in results if r]
    return local_paths, local_paths


# ============================================================
# ログイン・検索処理
# ============================================================
def auto_login(page):
    """イタンジBBに自動ログイン（既存コードと同じ手順）"""
    if not ITANJI_EMAIL or not ITANJI_PASSWORD:
        print("エラー: .env ファイルにITANJI_EMAIL, ITANJI_PASSWORDを設定してください")
        return False

    try:
        def _safe_title_and_content():
            # Navigation can still be in progress right after click.
            for _ in range(4):
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=3000)
                    return page.title(), page.content()
                except Exception:
                    try:
                        page.wait_for_timeout(350)
                    except Exception:
                        pass
            try:
                return page.title(), ""
            except Exception:
                return "", ""

        print("[START] ログイン開始")

        # 1. bukkakun.comにアクセス
        print(f"トップページにアクセス: {TOP_URL}")
        page.goto(TOP_URL, wait_until="load", timeout=60000)
        page.wait_for_timeout(2000)

        # 2. ログインリンクを探してクリック
        print("ログインリンクを検索中...")
        login_link_selectors = [
            'a:has-text("ログイン")',
            'a[href*="login"]',
            'a[href*="itandi-accounts"]',
        ]
        
        login_link = None
        login_href = None
        for selector in login_link_selectors:
            try:
                links = page.locator(selector)
                if links.count() > 0:
                    login_link = links.first
                    login_href = login_link.get_attribute("href")
                    print(f"  ログインリンクを発見: {login_href}")
                    break
            except:
                continue

        if not login_link:
            print("[ERROR] ログインリンクが見つかりません")
            page.screenshot(path=os.path.join(OUTPUT_DIR, "debug_no_login_link.png"))
            return False

        # ログインリンクをクリック（URL遷移を待つ）
        print("ログインリンクをクリック...")
        try:
            with page.expect_navigation(timeout=30000, wait_until="domcontentloaded") as navigation_info:
                login_link.click()
        except:
            # ナビゲーションが発生しない場合も続行
            login_link.click()
        
        page.wait_for_timeout(2000)
        current_url = page.url
        print(f"  遷移後のURL: {current_url}")

        # 400エラーチェック
        page_title, page_content = _safe_title_and_content()
        if "400" in page_title or "エラー" in page_title or "Bad Request" in page_content or "ページが表示できませんでした" in page_content:
            print("[WARN] 400エラーが検出されました。リトライします...")
            page.screenshot(path=os.path.join(OUTPUT_DIR, "debug_400_error.png"))
            
            # トップページに戻って再試行
            print("トップページに戻って再試行...")
            page.goto(TOP_URL, wait_until="load", timeout=60000)
            page.wait_for_timeout(2000)
            
            # 再度ログインリンクをクリック
            login_link = page.locator('a:has-text("ログイン")').first
            try:
                with page.expect_navigation(timeout=30000, wait_until="domcontentloaded"):
                    login_link.click()
            except:
                login_link.click()
            page.wait_for_timeout(2000)
            
            # 再度エラーチェック
            page_title, page_content = _safe_title_and_content()
            if "400" in page_title or "エラー" in page_title or "Bad Request" in page_content:
                print("[ERROR] 400エラーが解消されませんでした")
                page.screenshot(path=os.path.join(OUTPUT_DIR, "debug_400_error_retry.png"))
                return False

        # 3. メールアドレス入力
        print("メールアドレス入力中...")
        try:
            page.wait_for_load_state("domcontentloaded", timeout=15000)
            page.wait_for_timeout(1200)
        except:
            pass

        email_selectors = [
            '#email',
            '#identifierId',
            'input[name="email"]',
            'input[name="username"]',
            'input[name="loginId"]',
            'input[name="identifier"]',
            'input[autocomplete="username"]',
            'input[type="email"]',
            'input[type="text"][name*="mail" i]',
            'input[type="text"][id*="mail" i]',
            'input[type="text"][name*="user" i]',
            'input[type="text"][id*="user" i]',
        ]
        email_input = None
        for selector in email_selectors:
            try:
                email_input = page.locator(selector)
                if email_input.count() > 0:
                    email_input = email_input.first
                    email_input.wait_for(state="visible", timeout=10000)
                    break
            except:
                continue

        if not email_input or email_input.count() == 0:
            try:
                fallback = page.locator(
                    'input[type="email"], input[autocomplete="username"], input[name*="mail" i], input[id*="mail" i], input[name*="user" i], input[id*="user" i], input[type="text"]'
                )
                if fallback.count() > 0:
                    email_input = fallback.first
            except:
                pass

        if not email_input or email_input.count() == 0:
            print("[ERROR] メールアドレス入力欄が見つかりません")
            page.screenshot(path=os.path.join(OUTPUT_DIR, "debug_no_email_input.png"))
            return False

        email_input.fill(ITANJI_EMAIL)
        page.wait_for_timeout(500)

        # 4. パスワード入力
        print("パスワード入力中...")
        password_selectors = [
            '#password',
            'input[name="password"]',
            'input[name="passwd"]',
            'input[autocomplete="current-password"]',
            'input[type="password"]',
        ]
        password_input = None
        for selector in password_selectors:
            try:
                password_input = page.locator(selector)
                if password_input.count() > 0:
                    password_input = password_input.first
                    password_input.wait_for(state="visible", timeout=10000)
                    break
            except:
                continue

        if not password_input or password_input.count() == 0:
            try:
                fallback_pw = page.locator('input[type="password"]')
                if fallback_pw.count() > 0:
                    password_input = fallback_pw.first
            except:
                pass

        if not password_input or password_input.count() == 0:
            print("[ERROR] パスワード入力欄が見つかりません")
            page.screenshot(path=os.path.join(OUTPUT_DIR, "debug_no_password_input.png"))
            return False

        password_input.fill(ITANJI_PASSWORD)
        page.wait_for_timeout(500)

        # 5. ログインボタンをクリック
        print("ログインボタンをクリック...")
        login_button_selectors = [
            'input.filled-button[value="ログイン"]',
            'button:has-text("ログイン")',
            'button:has-text("次へ")',
            'button:has-text("続行")',
            'input[type="submit"]',
            'button[type="submit"]',
        ]
        
        login_button = None
        for selector in login_button_selectors:
            try:
                login_button = page.locator(selector)
                if login_button.count() > 0:
                    login_button = login_button.first
                    break
            except:
                continue

        if not login_button or login_button.count() == 0:
            print("[ERROR] ログインボタンが見つかりません")
            page.screenshot(path=os.path.join(OUTPUT_DIR, "debug_no_login_button.png"))
            return False

        # ログインボタンクリック後のリダイレクトを待つ
        try:
            with page.expect_navigation(timeout=30000, wait_until="domcontentloaded"):
                login_button.click()
        except:
            # ナビゲーションが発生しない場合も続行
            login_button.click()

        page.wait_for_timeout(3000)

        # 6. ログイン成功確認
        current_url = page.url
        print(f"ログイン後のURL: {current_url}")

        # エラーページチェック
        page_title, page_content = _safe_title_and_content()
        if "400" in page_title or "エラー" in page_title or "Bad Request" in page_content or "ページが表示できませんでした" in page_content:
            print("[ERROR] ログイン後にエラーページが表示されました")
            page.screenshot(path=os.path.join(OUTPUT_DIR, "debug_login_400_error.png"))
            return False

        if "itandibb.com" in current_url or "bukkakun.com" in current_url:
            print("[OK] ログイン完了")
            # ページが完全に読み込まれるまで待機
            page.wait_for_timeout(2000)
            # ログインページに戻っていないか確認
            if "login" in current_url.lower() or "itandi-accounts" in current_url.lower():
                print("[WARN] ログインページに戻っています。トップページに移動します...")
                page.goto(TOP_URL, wait_until="load", timeout=60000)
                page.wait_for_timeout(2000)
            return True
        else:
            print("[WARN] 予期しないURLに遷移しました")
            print(f"  現在のページタイトル: {page_title}")
            
            # デバッグ用スクリーンショット
            page.screenshot(path=os.path.join(OUTPUT_DIR, "debug_login_failed.png"))
            print(f"  スクリーンショット保存: output/debug_login_failed.png")
            
            # URLにitandiが含まれていれば成功とみなす
            if "itandi" in current_url.lower():
                print("[OK] ログイン成功（URL判定）")
                return True

            return False

    except Exception as e:
        print(f"[ERROR] ログインエラー: {e}")
        import traceback
        traceback.print_exc()
        try:
            page.screenshot(path=os.path.join(OUTPUT_DIR, "debug_login_error.png"))
            print(f"エラー時スクリーンショット: output/debug_login_error.png")
        except:
            pass
        return False


def auto_search(page, target_area: str = "板橋区", skip_list_search: bool = False):
    """リスト検索を自動実行（指定された市区町村を選択）"""
    try:
        print("\n" + "=" * 50)
        print(f"【{target_area}】のリスト検索を開始...")
        print("=" * 50)

        # ステップ1: リスト検索ボタンをクリック（スキップフラグがFalseの場合のみ）
        if not skip_list_search:
            print("\n[ステップ1] リスト検索ボタンをクリック...")
            page.wait_for_timeout(2000)
            
            list_search_selectors = [
                'button:has(p:has-text("リスト検索"))',
                'button:has-text("リスト検索")',
                'p:has-text("リスト検索")',
            ]
            
            list_search_btn = None
            for selector in list_search_selectors:
                try:
                    btn = page.locator(selector)
                    if btn.count() > 0:
                        list_search_btn = btn.first
                        if selector.startswith('p:'):
                            parent_btn = btn.first.locator('xpath=ancestor::button')
                            if parent_btn.count() > 0:
                                list_search_btn = parent_btn.first
                            else:
                                list_search_btn = page.locator('button:has(p:has-text("リスト検索"))').first
                        break
                except:
                    continue

            if not list_search_btn:
                print("[ERROR] リスト検索ボタンが見つかりません")
                page.screenshot(path=os.path.join(OUTPUT_DIR, "debug_no_list_search_btn.png"))
                return False

            list_search_btn.click()
            page.wait_for_timeout(3000)
            print("[OK] リスト検索画面に移動")
        else:
            print("\n[スキップ] 既にリスト検索画面にいるため、リスト検索ボタンのクリックをスキップします")
            page.wait_for_timeout(2000)

        # ステップ2: 「所在地で絞り込み」をクリック
        print("\n[ステップ2] 所在地で絞り込みをクリック...")
        location_filter_selectors = [
            'div.itandi-bb-ui__Button__Text:has-text("所在地で絞り込み")',
            'div:has-text("所在地で絞り込み")',
            'button:has-text("所在地で絞り込み")',
        ]

        location_filter_btn = None
        for selector in location_filter_selectors:
            try:
                btn = page.locator(selector)
                if btn.count() > 0:
                    location_filter_btn = btn.first
                    # 親要素がボタンの場合
                    if selector.startswith('div:'):
                        # 親のbutton要素を探す
                        parent_btn = btn.first.locator('xpath=ancestor::button')
                        if parent_btn.count() > 0:
                            location_filter_btn = parent_btn.first
                        else:
                            # 別の方法: 親要素を探す
                            location_filter_btn = page.locator('button:has(div:has-text("所在地で絞り込み"))').first
                    break
            except:
                continue

        if not location_filter_btn:
            print("[ERROR] 所在地で絞り込みボタンが見つかりません")
            page.screenshot(path=os.path.join(OUTPUT_DIR, "debug_no_location_filter.png"))
            return False

        location_filter_btn.click()
        page.wait_for_timeout(2000)
        print("[OK] 所在地で絞り込み画面を開きました")

        # ステップ3: 指定された市区町村を選択
        print(f"\n[ステップ3] {target_area}を選択中...")
        area_selectors = [
            f'p.css-j5f0i8:has-text("{target_area}")',
            f'p:has-text("{target_area}")',
            f'div:has-text("{target_area}")',
            f'label:has-text("{target_area}")',
            f'span:has-text("{target_area}")',
        ]

        area_found = False
        for selector in area_selectors:
            try:
                area_element = page.locator(selector)
                if area_element.count() > 0:
                    print(f"  パターン '{selector}' で{target_area}を検出")
                    area_element.first.click()
                    page.wait_for_timeout(500)
                    area_found = True
                    print(f"[OK] {target_area}を選択しました")
                    break
            except Exception as e:
                print(f"  パターン '{selector}' でエラー: {e}")
                continue

        allow_manual_prompt = os.getenv("ITANJI_ALLOW_MANUAL_PROMPT", "0").lower() in ("1", "true", "yes")

        if not area_found:
            print(f"[WARN] {target_area}が見つかりませんでした。手動で選択してください。")
            page.screenshot(path=os.path.join(OUTPUT_DIR, f"debug_no_{target_area}.png"))
            if allow_manual_prompt:
                input(f"\n>>> {target_area}を選択してEnterを押してください... ")
            else:
                print("[INFO] ITANJI_ALLOW_MANUAL_PROMPT=0 のため手動待機をスキップして続行します。")

        # ステップ4: 「確定」ボタンをクリック
        print("\n[ステップ4] 確定ボタンをクリック...")
        confirm_selectors = [
            'div.itandi-bb-ui__Button__Text:has-text("確定")',
            'div:has-text("確定")',
            'button:has-text("確定")',
        ]

        confirm_btn = None
        for selector in confirm_selectors:
            try:
                btn = page.locator(selector)
                if btn.count() > 0:
                    confirm_btn = btn.first
                    # 親要素がボタンの場合
                    if selector.startswith('div:'):
                        # 親のbutton要素を探す
                        parent_btn = btn.first.locator('xpath=ancestor::button')
                        if parent_btn.count() > 0:
                            confirm_btn = parent_btn.first
                        else:
                            # 別の方法: 親要素を探す
                            confirm_btn = page.locator('button:has(div:has-text("確定"))').first
                    break
            except:
                continue

        if not confirm_btn:
            print("[WARN] 確定ボタンが見つかりませんでした。スキップします。")
        else:
            confirm_btn.click()
            page.wait_for_timeout(2000)
            print("[OK] 確定しました")

        # ステップ4.5: 広告可否は条件に使わない（スキップ）
        print("\n[ステップ4.5] 広告可否チェックはスキップ（条件から除外）")

        # ステップ5: 検索実行ボタンをクリック
        print("\n[ステップ5] 検索を実行中...")
        search_selectors = [
            'button.ListSearchButton[type="submit"]',
            'button:has-text("検索")[type="submit"]',
            'button:has-text("検索")',
            'button[type="submit"]:has-text("検索")',
        ]

        search_executed = False
        for selector in search_selectors:
            try:
                search_btn = page.locator(selector)
                if search_btn.count() > 0:
                    print(f"  検索ボタンを発見: {selector}")
                    search_btn.first.click()
                    page.wait_for_timeout(3000)
                    search_executed = True
                    print("[OK] 検索を実行しました")
                    break
            except Exception as e:
                print(f"  パターン '{selector}' でエラー: {e}")
                continue

        if not search_executed:
            print("[WARN] 検索ボタンが見つかりませんでした。手動で検索してください。")
            page.screenshot(path=os.path.join(OUTPUT_DIR, "debug_no_search_btn.png"))
            if allow_manual_prompt:
                input("\n>>> 検索を実行してEnterを押してください... ")
            else:
                print("[INFO] ITANJI_ALLOW_MANUAL_PROMPT=0 のため手動待機をスキップして続行します。")

        # 検索結果が表示されるまで待機
        page.wait_for_timeout(3000)
        
        print("\n[OK] リスト検索の設定が完了しました")
        return True

    except Exception as e:
        print(f"[ERROR] 検索エラー: {e}")
        import traceback
        traceback.print_exc()
        try:
            page.screenshot(path=os.path.join(OUTPUT_DIR, "debug_search_error.png"))
        except:
            pass
        return False


# ============================================================
# 【重要】URL収集・フィルタリング処理 (大幅改良)
# ============================================================

# グローバル変数: スクレイピング中に見つかった全URLと申込ありURLを記録
all_found_urls = set()  # 今回のスクレイピングで一覧に出現した全URL
applied_urls = set()    # 「申込あり」としてスキップしたURL

def collect_property_urls_from_page(page, existing_normalized_urls: set = None) -> tuple[list[str], list[str], dict]:
    """1ページ分の物件URLを収集し、既存・申込済みを即フィルタリング
    
    Returns:
        tuple[list[str], list[str], dict]:
        (新規物件のURLリスト, 既存物件のURLリスト, 収集メトリクス)
    """
    collect_start = time.perf_counter()

    stats = {
        "cards": 0,
        "skip_no_url": 0,
        "skip_no_img": 0,
        "skip_not_recruiting": 0,
        "skip_applied": 0,
        "img_unknown": 0,
        "new_count": 0,
        "existing_count": 0,
        "collect_sec": 0.0,
    }

    global all_found_urls, applied_urls
    new_urls = []
    existing_urls = []
    
    # 高速化: カード情報をブラウザ側で一括抽出（Python↔Browser往復を最小化）
    try:
        extracted = page.evaluate(
            """
            () => {
              const cardSelectors = [
                "div.itandi-bb-ui__Box.css-asttbz",
                "[data-testid='property-card']",
                ".property-card",
                "a[href*='/rent_rooms/']"
              ];
              let cards = [];
              for (const sel of cardSelectors) {
                const found = Array.from(document.querySelectorAll(sel));
                if (found.length > 0) {
                  cards = found;
                  break;
                }
              }
              const textOf = (el) => ((el?.innerText || el?.textContent || "").trim());
              const items = cards.map((card) => {
                let href = "";
                try {
                  if (card.matches && card.matches("a[href*='/rent_rooms/']")) {
                    href = card.getAttribute("href") || "";
                  }
                } catch {}
                if (!href) {
                  const link = card.querySelector("a[href*='/rent_rooms/']");
                  href = link ? (link.getAttribute("href") || "") : "";
                }

                const cardText = textOf(card);
                const statusTexts = Array.from(card.querySelectorAll("div.itandi-bb-ui__Flex"))
                  .map((el) => textOf(el))
                  .filter(Boolean);

                let imageCount = null;
                const m = cardText.match(/(\\d+)\\s*枚/);
                if (m) {
                  imageCount = Number.parseInt(m[1], 10);
                }
                if (imageCount === null && statusTexts.length > 0) {
                  for (const t of statusTexts) {
                    const mm = t.match(/(\\d+)\\s*枚/);
                    if (mm) { imageCount = Number.parseInt(mm[1], 10); break; }
                  }
                }

                return { href, cardText, statusTexts, imageCount };
              });

              return { cards: cards.length, items };
            }
            """
        )
    except:
        extracted = {"cards": 0, "items": []}

    stats["cards"] = int(extracted.get("cards", 0) or 0)
    cards_data = extracted.get("items", []) or []

    if not cards_data:
        stats["collect_sec"] = time.perf_counter() - collect_start
        return [], [], stats

    for card_data in cards_data:
        try:
            # 1. URL取得（先にURLを取得して記録用に使う）
            href = (card_data.get("href") or "").strip()
            
            if not href or "/rent_rooms/" not in href:
                stats["skip_no_url"] += 1
                continue

            # フルURL化
            detail_url = href if href.startswith("http") else "https://itandibb.com" + href
            card_text = card_data.get("cardText") or ""
            # 一覧に出現したURLは募集終了判定用に先に記録しておく
            all_found_urls.add(detail_url)

            # 画像1枚以上の物件のみ対象（0枚はスキップ、枚数取得できない場合も対象外）
            image_count = card_data.get("imageCount")
            if image_count is not None and int(image_count) <= 0:
                stats["skip_no_img"] += 1
                continue
            if image_count is None:
                stats["img_unknown"] += 1
                continue

            # 2. 募集中の物件のみ（申込あり・紹介不可等はスキップ）
            status_texts = card_data.get("statusTexts") or []
            if status_texts:
                is_recruiting = any("募集中" in s for s in status_texts)
                has_application = any(
                    ("申込あり" in s) or ("申し込み" in s) or ("紹介不可" in s)
                    for s in status_texts
                )
            else:
                is_recruiting = "募集中" in card_text
                has_application = (
                    "申込あり" in card_text or
                    "申し込み" in card_text or
                    "紹介不可" in card_text
                )
            if not is_recruiting or has_application:
                if not is_recruiting:
                    stats["skip_not_recruiting"] += 1
                if has_application:
                    stats["skip_applied"] += 1
                    # ★ 申込ありURLを記録（分析・デバッグ用）
                    applied_urls.add(detail_url)
                continue
            
            # 3. 既存URLチェック (正規化して比較)
            if existing_normalized_urls:
                norm_url = normalize_url(detail_url)
                if norm_url in existing_normalized_urls:
                    # 既存物件（UPDATE_EXISTING_DATAがTrueなら後で更新）
                    existing_urls.append(detail_url)
                    stats["existing_count"] += 1
                    continue

            # ここまで来たら「新規」かつ「募集中」
            new_urls.append(detail_url)
            stats["new_count"] += 1

        except:
            continue

    stats["collect_sec"] = time.perf_counter() - collect_start
    return new_urls, existing_urls, stats

def collect_property_urls_all_pages(page, existing_data: dict = None) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """全ページを巡回し、新規URLと既存URLを収集して返す
    
    Returns:
        tuple: (新規URLリスト, 既存URLリスト) - 各要素は (url, index) のタプル
    """
    global interrupt_flag
    
    all_new_urls = []
    all_existing_urls = []
    current_page = 1
    update_existing_any = UPDATE_EXISTING_DATA or UPDATE_EXISTING_META_ONLY
    
    # 既存URLの正規化セットを作成（高速検索用）
    existing_normalized_urls = set()
    if existing_data and 'by_url' in existing_data:
        for url in existing_data['by_url'].keys():
            existing_normalized_urls.add(normalize_url(url))
        if update_existing_any:
            print(f"  既存物件 {len(existing_normalized_urls)} 件をメモリにロード済み（データ更新対象）")
        else:
            print(f"  既存物件 {len(existing_normalized_urls)} 件をメモリにロード済み（スキップ対象）")

    while current_page <= MAX_PAGES:
        if interrupt_flag: break
        
        print(f"  ページ {current_page} をスキャン中...")
        
        page_start = time.perf_counter()

        # ページ内のURLを取得（新規と既存を分離）
        page_new_urls, page_existing_urls, page_stats = collect_property_urls_from_page(page, existing_normalized_urls)
        
        # 新規URLを追加
        offset = len(all_new_urls)
        for i, url in enumerate(page_new_urls):
            all_new_urls.append((url, offset + i))
        
        # 既存URLを追加（既存更新が有効な場合のみ）
        if update_existing_any:
            offset_existing = len(all_existing_urls)
            for i, url in enumerate(page_existing_urls):
                all_existing_urls.append((url, offset_existing + i))
            
        print(f"    -> 新規: {len(page_new_urls)}件, 既存: {len(page_existing_urls)}件 (累計 新規: {len(all_new_urls)}件, 既存: {len(all_existing_urls)}件)")
        if ENABLE_TIMING_LOG:
            print(
                "       [一覧計測] "
                f"cards={page_stats.get('cards', 0)}, "
                f"skip:no_url={page_stats.get('skip_no_url', 0)}, "
                f"skip:no_img={page_stats.get('skip_no_img', 0)}, "
                f"skip:not_recruiting={page_stats.get('skip_not_recruiting', 0)}, "
                f"skip:applied={page_stats.get('skip_applied', 0)}, "
                f"img_unknown={page_stats.get('img_unknown', 0)}, "
                f"collect={page_stats.get('collect_sec', 0.0):.2f}s"
            )

        # 次へボタン
        try:
            next_btn = page.locator('button:has-text("次へ"), a:has-text("次へ"), [data-testid="next-page"]').first
            if next_btn.count() > 0 and not next_btn.is_disabled():
                next_start = time.perf_counter()
                next_btn.click()
                page.wait_for_timeout(PAGINATION_WAIT_MS) # 遷移待ち
                if ENABLE_TIMING_LOG:
                    print(f"       [一覧計測] next-page={time.perf_counter() - next_start:.2f}s total={time.perf_counter() - page_start:.2f}s")
                current_page += 1
            else:
                if ENABLE_TIMING_LOG:
                    print(f"       [一覧計測] total={time.perf_counter() - page_start:.2f}s")
                print("  最終ページ到達")
                break
        except Exception as e:
            err_name = type(e).__name__
            if "TargetClosed" in err_name or "closed" in str(e).lower():
                print(f"  [WARN] ブラウザが閉じられました。ここまでの結果を使用します。")
            else:
                print(f"  [WARN] ページ遷移エラー: {err_name}: {e}")
            break

    return all_new_urls, all_existing_urls

# ============================================================
# スクレイピング実行部 (微調整)
# ============================================================


def scrape_properties_parallel_multi_tabs(context, property_urls: list[tuple[str, int]], area_name: str = "") -> list[dict]:
    """新規URLを並列タブで取得し、詳細情報抽出と画像DLを実行"""
    global interrupt_flag

    if not property_urls:
        print("  新規対象の物件はありません")
        return []

    print(f"\n[開始] 新規物件 {len(property_urls)} 件の詳細情報を取得します")

    properties = []
    start_time = time.time()
    total = len(property_urls)

    for batch_start in range(0, len(property_urls), CONCURRENT_TABS):
        if interrupt_flag:
            break

        batch = property_urls[batch_start : batch_start + CONCURRENT_TABS]
        tabs = []

        open_phase_start = time.perf_counter()
        for i, (url, idx) in enumerate(batch):
            try:
                tab = context.new_page()
                t_goto = time.perf_counter()
                tab.goto(url, wait_until="domcontentloaded", timeout=15000)
                goto_sec = time.perf_counter() - t_goto
                tabs.append((tab, url, idx, goto_sec))
                time.sleep(random.uniform(TAB_OPEN_DELAY_MIN, TAB_OPEN_DELAY_MAX))
            except:
                pass
        if ENABLE_TIMING_LOG:
            print(f"    [計測][新規][batch {batch_start // CONCURRENT_TABS + 1}] open={time.perf_counter() - open_phase_start:.2f}s tabs={len(tabs)}")

        t_wait = time.perf_counter()
        time.sleep(BATCH_WAIT_SECONDS)
        if ENABLE_TIMING_LOG:
            print(f"    [計測][新規][batch {batch_start // CONCURRENT_TABS + 1}] batch_wait={time.perf_counter() - t_wait:.2f}s")

        for i, (tab, url, idx, goto_sec) in enumerate(tabs):
            try:
                item_start = time.perf_counter()
                render_sec = 0.0
                eval_sec = 0.0
                extract_sec = 0.0
                reload_sec = 0.0
                retried = 0

                t_render = time.perf_counter()
                fully_render(tab)
                wait_for_detail_content(tab, timeout_ms=6000)
                render_sec += time.perf_counter() - t_render

                t_eval = time.perf_counter()
                body = tab.evaluate("document.body.innerText")
                eval_sec += time.perf_counter() - t_eval

                t_extract = time.perf_counter()
                data = extract_property_details(tab, body)
                extract_sec += time.perf_counter() - t_extract

                if not is_valid_property_data(data):
                    try:
                        retried = 1
                        t_reload = time.perf_counter()
                        tab.reload(wait_until="load", timeout=12000)
                        tab.wait_for_timeout(600)
                        reload_sec += time.perf_counter() - t_reload

                        t_render = time.perf_counter()
                        fully_render(tab)
                        wait_for_detail_content(tab, timeout_ms=6000)
                        render_sec += time.perf_counter() - t_render

                        t_eval = time.perf_counter()
                        body = tab.evaluate("document.body.innerText")
                        eval_sec += time.perf_counter() - t_eval

                        t_extract = time.perf_counter()
                        data = extract_property_details(tab, body)
                        extract_sec += time.perf_counter() - t_extract
                    except Exception:
                        pass

                data["detail_url"] = url
                data["scraped_at"] = datetime.now().isoformat()
                if area_name:
                    data["listed_area"] = area_name

                if not is_sufficient_for_save(data):
                    if ENABLE_TIMING_LOG:
                        lacks = []
                        if not (data.get("stations")): lacks.append("駅")
                        if not (data.get("image_urls")): lacks.append("画像")
                        if not (data.get("rent")): lacks.append("家賃")
                        if not (data.get("address")): lacks.append("住所")
                        if not (data.get("layout")): lacks.append("間取り")
                        print(f"    [{batch_start+i+1}/{len(property_urls)}] スキップ（問題あり: {', '.join(lacks)}欠け）: {data.get('title', '')}")
                    continue

                pid = sanitize_filename(f"{data['building_name']}_{data['room_number']}")
                if not pid or pid == "_":
                    pid = f"prop_{batch_start+i}"
                data["id"] = pid
                data["local_images"] = []

                properties.append(data)
                print(f"    [{batch_start+i+1}/{len(property_urls)}] 取得: {data['title']}")
                if ENABLE_TIMING_LOG:
                    total_sec = time.perf_counter() - item_start
                    print(
                        f"      [計測][新規][{batch_start+i+1}/{len(property_urls)}] "
                        f"goto={goto_sec:.2f}s render={render_sec:.2f}s eval={eval_sec:.2f}s "
                        f"extract={extract_sec:.2f}s reload={reload_sec:.2f}s retry={retried} total={total_sec:.2f}s"
                    )

            except Exception as e:
                print(f"    [ERROR] {url}: {e}")
            finally:
                tab.close()

        if len(properties) % SAVE_INTERVAL == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / len(properties) if len(properties) > 0 else 0
            remaining = (total - (batch_start + len(batch))) * avg_time / CONCURRENT_TABS
            print(f"    [途中保存] {len(properties)}件 (平均 {avg_time:.1f}秒/件, 残り約 {remaining/60:.1f}分)")
            save_results(properties, area_name)

    if properties and not interrupt_flag:
        props_with_images = [p for p in properties if p.get("image_urls")]
        total_image_urls = sum(len(p.get("image_urls", [])) for p in props_with_images)
        print(f"\n[開始] 画像を一括ダウンロード（画像URLあり: {len(props_with_images)}件の物件・合計{total_image_urls}枚）...")
        if not props_with_images and R2_UPLOAD_ENABLED:
            print("  [注意] 詳細ページから画像URLを取得できた物件が0件です（img[src*='property-images'] が無い可能性）")
        download_start = time.time()

        download_tasks = [(p["id"], p["image_urls"]) for p in properties if p.get("image_urls")]

        def download_for_property(args):
            pid, urls = args
            try:
                if interrupt_flag:
                    return (pid, [], [])
                local_paths, new_paths = download_images(pid, urls)
                return (pid, local_paths, new_paths)
            except:
                return (pid, [], [])

        with ThreadPoolExecutor(max_workers=15) as executor:
            image_results = list(executor.map(download_for_property, download_tasks))

        image_dict = {pid: paths for pid, paths, _ in image_results}
        for p_item in properties:
            p_item["local_images"] = image_dict.get(p_item["id"], [])

        download_time = time.time() - download_start
        total_downloaded = sum(len(paths) for _, paths, _ in image_results)
        print(f"  画像ダウンロード完了: {total_downloaded}枚 ({download_time:.1f}秒)")

        # 今回取得した全画像をR2にアップロード（ローカル既存分だけだとR2に届かないため、バッチ全件を送る）
        all_upload_paths = [path for _, paths, _ in image_results for path in paths]
        if all_upload_paths:
            print(f"  [R2] 画像アップロード開始: {len(all_upload_paths)}件（バックグラウンド）")
            upload_files_to_r2_background(all_upload_paths)
        elif R2_UPLOAD_ENABLED:
            if not download_tasks:
                print("  [R2] 画像アップロード対象なし（取得した画像URLが0件）")
            else:
                print("  [R2] 画像アップロード対象なし")

        save_results(properties, area_name)
    elif properties and interrupt_flag:
        print("\n[中断] ここまでの結果を保存して終了します")

    elapsed = time.time() - start_time
    if properties:
        print(f"\n[完了] {len(properties)}件取得 (総所要時間: {elapsed:.1f}秒, 平均: {elapsed/len(properties):.2f}秒/件)")

    return properties


def update_existing_properties(context, existing_urls: list[tuple[str, int]], existing_data: dict, meta_only: bool = False) -> int:
    """既存物件のJSONを更新（必要最小限の項目のみ更新可）"""
    global interrupt_flag

    if not existing_urls:
        return 0

    mode_label = "メタ情報のみ" if meta_only else "全情報"
    print(f"\n[開始] 既存物件の更新: 合計 {len(existing_urls)}件")
    print(f"  更新内容: {mode_label}（画像/IDは維持）")

    updated_count = 0
    failed_count = 0
    skipped_count = 0
    updated_index_payloads = []
    start_time = time.time()

    for batch_start in range(0, len(existing_urls), CONCURRENT_TABS):
        if interrupt_flag:
            break

        batch = existing_urls[batch_start : batch_start + CONCURRENT_TABS]
        tabs = []

        open_phase_start = time.perf_counter()
        for i, (url, idx) in enumerate(batch):
            try:
                tab = context.new_page()
                t_goto = time.perf_counter()
                tab.goto(url, wait_until="domcontentloaded", timeout=15000)
                goto_sec = time.perf_counter() - t_goto
                tabs.append((tab, url, idx, goto_sec))
                time.sleep(random.uniform(TAB_OPEN_DELAY_MIN, TAB_OPEN_DELAY_MAX))
            except:
                pass
        if ENABLE_TIMING_LOG:
            print(f"    [計測][更新][batch {batch_start // CONCURRENT_TABS + 1}] open={time.perf_counter() - open_phase_start:.2f}s tabs={len(tabs)}")

        t_wait = time.perf_counter()
        time.sleep(BATCH_WAIT_SECONDS)
        if ENABLE_TIMING_LOG:
            print(f"    [計測][更新][batch {batch_start // CONCURRENT_TABS + 1}] batch_wait={time.perf_counter() - t_wait:.2f}s")

        for i, (tab, url, idx, goto_sec) in enumerate(tabs):
            try:
                item_start = time.perf_counter()
                eval_sec = 0.0
                render_sec = 0.0
                extract_sec = 0.0
                reload_sec = 0.0
                io_sec = 0.0
                index_sec = 0.0
                retried = 0

                t_eval = time.perf_counter()
                body = tab.evaluate("document.body.innerText")
                eval_sec += time.perf_counter() - t_eval

                if meta_only:
                    t_extract = time.perf_counter()
                    new_data = extract_property_meta_light(tab, body)
                    extract_sec += time.perf_counter() - t_extract
                else:
                    t_render = time.perf_counter()
                    fully_render(tab)
                    wait_for_detail_content(tab, timeout_ms=6000)
                    render_sec += time.perf_counter() - t_render

                    t_eval = time.perf_counter()
                    body = tab.evaluate("document.body.innerText")
                    eval_sec += time.perf_counter() - t_eval

                    t_extract = time.perf_counter()
                    new_data = extract_property_details(tab, body)
                    extract_sec += time.perf_counter() - t_extract

                    if not is_valid_property_data(new_data):
                        try:
                            retried = 1
                            t_reload = time.perf_counter()
                            tab.reload(wait_until="load", timeout=12000)
                            tab.wait_for_timeout(600)
                            reload_sec += time.perf_counter() - t_reload

                            t_render = time.perf_counter()
                            fully_render(tab)
                            wait_for_detail_content(tab, timeout_ms=6000)
                            render_sec += time.perf_counter() - t_render

                            t_eval = time.perf_counter()
                            body = tab.evaluate("document.body.innerText")
                            eval_sec += time.perf_counter() - t_eval

                            t_extract = time.perf_counter()
                            new_data = extract_property_details(tab, body)
                            extract_sec += time.perf_counter() - t_extract
                        except Exception:
                            pass

                existing_prop = existing_data['by_url'].get(url)
                if not existing_prop:
                    norm_url = normalize_url(url)
                    for ex_url, ex_prop in existing_data['by_url'].items():
                        if normalize_url(ex_url) == norm_url:
                            existing_prop = ex_prop
                            break

                if not existing_prop:
                    print(f"    [SKIP] 既存データに一致なし: {url}")
                    skipped_count += 1
                    continue

                prop_id = existing_prop.get('id', '')

                t_io = time.perf_counter()
                detail_json_path = os.path.join(DATA_DIR, f"{prop_id}.json")
                if os.path.exists(detail_json_path):
                    with open(detail_json_path, "r", encoding="utf-8") as f:
                        full_data = json.load(f)
                else:
                    full_data = existing_prop.copy()

                old_images = full_data.get('local_images', [])
                old_image_urls = full_data.get('image_urls', [])
                old_id = full_data.get('id', prop_id)
                old_scraped_at = full_data.get('scraped_at', '')

                if meta_only:
                    if new_data.get('built_date') and not full_data.get('built_date'):
                        full_data['built_date'] = new_data['built_date']
                    if new_data.get('direction') and not full_data.get('direction'):
                        full_data['direction'] = new_data['direction']
                else:
                    full_data.update(new_data)

                full_data['id'] = old_id
                full_data['local_images'] = old_images
                full_data['detail_url'] = url
                full_data['updated_at'] = datetime.now().isoformat()
                full_data['original_scraped_at'] = old_scraped_at

                if not full_data.get('image_urls'):
                    full_data['image_urls'] = old_image_urls

                with open(detail_json_path, "w", encoding="utf-8") as f:
                    json.dump(full_data, f, ensure_ascii=False, indent=2)
                io_sec += time.perf_counter() - t_io

                # インデックス更新は最後に一括で実施（I/O削減）
                updated_index_payloads.append(full_data)

                updated_count += 1
                print(f"    [{batch_start+i+1}/{len(existing_urls)}] 更新: {full_data.get('title', prop_id)}")
                if ENABLE_TIMING_LOG:
                    total_sec = time.perf_counter() - item_start
                    print(
                        f"      [計測][更新][{batch_start+i+1}/{len(existing_urls)}] "
                        f"goto={goto_sec:.2f}s render={render_sec:.2f}s eval={eval_sec:.2f}s "
                        f"extract={extract_sec:.2f}s reload={reload_sec:.2f}s io={io_sec:.2f}s index={index_sec:.2f}s "
                        f"retry={retried} total={total_sec:.2f}s"
                    )

            except Exception as e:
                print(f"    [ERROR] {url}: {e}")
                failed_count += 1
            finally:
                tab.close()

    t_index_bulk = time.perf_counter()
    if updated_index_payloads:
        update_property_indexes_bulk(updated_index_payloads)
    index_bulk_sec = time.perf_counter() - t_index_bulk

    elapsed = time.time() - start_time
    print(f"\n[完了] 更新 {updated_count}件 (総所要時間: {elapsed:.1f}秒)")
    if updated_index_payloads:
        print(f"  - インデックス一括更新: {len(updated_index_payloads)}件 ({index_bulk_sec:.2f}秒)")
    if failed_count or skipped_count:
        print(f"  - 失敗: {failed_count}件")
        print(f"  - スキップ: {skipped_count}件")

    return updated_count


def update_property_index(prop_data: dict):
    """properties_index.jsonの特定物件を更新"""
    if not os.path.exists(INDEX_FILE):
        return
    
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            index_list = json.load(f)
    except:
        return
    
    # IDで検索して更新
    prop_id = prop_data.get('id', '')
    updated = False
    
    for i, item in enumerate(index_list):
        if item.get('id') == prop_id:
            # サマリー情報を更新
            index_list[i] = {
                "id": prop_data.get("id", ""),
                "source": SOURCE_ID,
                "title": prop_data.get("title", ""),
                "rent": prop_data.get("rent", ""),
                "management_fee": prop_data.get("management_fee", ""),
                "deposit": prop_data.get("deposit", ""),
                "key_money": prop_data.get("key_money", ""),
                "address": prop_data.get("address", ""),
                "stations": prop_data.get("stations", []),
                "layout": prop_data.get("layout", ""),
                "area": prop_data.get("area", ""),
                "floor": prop_data.get("floor", ""),
                "built_date": prop_data.get("built_date", ""),
                "structure": prop_data.get("structure", ""),
                "direction": prop_data.get("direction", ""),
                "available_date": prop_data.get("available_date", ""),
                "contract_period": prop_data.get("contract_period", ""),
                "parking": prop_data.get("parking", ""),
                "renewal_fee": prop_data.get("renewal_fee", ""),
                "insurance": prop_data.get("insurance", ""),
                "ad_fee": prop_data.get("ad_fee", ""),
                "transaction_type": prop_data.get("transaction_type", ""),
                "guarantee_company": prop_data.get("guarantee_company", ""),
                "remarks": prop_data.get("remarks", ""),
                "preferred_conditions": prop_data.get("preferred_conditions", ""),
                "viewing_start_date": prop_data.get("viewing_start_date", ""),
                "viewing_notes": prop_data.get("viewing_notes", ""),
                "facilities": prop_data.get("facilities", []),
                "building_name": prop_data.get("building_name", ""),
                "room_number": prop_data.get("room_number", ""),
                "local_images": prop_data.get("local_images", item.get("local_images", [])),
                "thumbnail": prop_data.get("local_images", [None])[0] if prop_data.get("local_images") else item.get("thumbnail"),
                "image_count": len(prop_data.get("local_images", [])) or len(prop_data.get("image_urls", [])) or item.get("image_count", 0),
                "detail_url": prop_data.get("detail_url", ""),
                "scraped_at": prop_data.get("original_scraped_at", ""),
                "updated_at": prop_data.get("updated_at", "")
            }
            updated = True
            break
    
    if updated:
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index_list, f, ensure_ascii=False, indent=2)


def update_property_indexes_bulk(updated_props: list[dict]):
    """properties_index.json をまとめて更新（高速）"""
    if not updated_props:
        return

    if not os.path.exists(INDEX_FILE):
        return

    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            index_list = json.load(f)
    except:
        return

    by_id = {p.get("id", ""): p for p in updated_props if p.get("id")}
    if not by_id:
        return

    for i, item in enumerate(index_list):
        prop_id = item.get("id", "")
        src = by_id.get(prop_id)
        if not src:
            continue
        index_list[i] = {
            "id": src.get("id", ""),
            "source": SOURCE_ID,
            "title": src.get("title", ""),
            "rent": src.get("rent", ""),
            "management_fee": src.get("management_fee", ""),
            "deposit": src.get("deposit", ""),
            "key_money": src.get("key_money", ""),
            "address": src.get("address", ""),
            "stations": src.get("stations", []),
            "layout": src.get("layout", ""),
            "area": src.get("area", ""),
            "floor": src.get("floor", ""),
            "built_date": src.get("built_date", ""),
            "structure": src.get("structure", ""),
            "direction": src.get("direction", ""),
            "available_date": src.get("available_date", ""),
            "contract_period": src.get("contract_period", ""),
            "parking": src.get("parking", ""),
            "renewal_fee": src.get("renewal_fee", ""),
            "insurance": src.get("insurance", ""),
            "ad_fee": src.get("ad_fee", ""),
            "transaction_type": src.get("transaction_type", ""),
            "guarantee_company": src.get("guarantee_company", ""),
            "remarks": src.get("remarks", ""),
            "preferred_conditions": src.get("preferred_conditions", ""),
            "viewing_start_date": src.get("viewing_start_date", ""),
            "viewing_notes": src.get("viewing_notes", ""),
            "facilities": src.get("facilities", []),
            "building_name": src.get("building_name", ""),
            "room_number": src.get("room_number", ""),
            "local_images": src.get("local_images", item.get("local_images", [])),
            "thumbnail": src["local_images"][0] if src.get("local_images") else item.get("thumbnail"),
            "image_count": len(src["local_images"]) if src.get("local_images") else (len(src.get("image_urls", [])) or item.get("image_count", 0)),
            "detail_url": src.get("detail_url", item.get("detail_url", "")),
            "scraped_at": src.get("original_scraped_at", item.get("scraped_at", "")),
            "updated_at": src.get("updated_at", item.get("updated_at", "")),
        }

    try:
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index_list, f, ensure_ascii=False, indent=2)
        # R2にもマージアップロード
        if R2_UPLOAD_ENABLED:
            upload_merged_index_to_r2()
    except:
        pass


def save_results(properties: list[dict], area_name: str = "", deleted_count: int = 0):
    """保存処理（追記モード）+ R2アップロード"""
    if not properties and deleted_count == 0: return
    setup_dirs()
    
    existing = []
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except: pass
        
    # 重複排除してマージ（同一idがpropertiesに複数含まれる場合も1件だけ採用）
    existing_ids = {p["id"] for p in existing}
    seen_new_ids = {}
    for p in properties:
        pid = p.get("id")
        if not pid or pid in existing_ids:
            continue
        if pid not in seen_new_ids:
            seen_new_ids[pid] = p
    new_items = list(seen_new_ids.values())
    
    # R2アップロード用のパスリスト
    upload_targets = []

    # 個別JSON保存
    for p in new_items:
        json_filename = f"{p['id']}.json"
        with open(os.path.join(DATA_DIR, json_filename), "w", encoding="utf-8") as f:
            json.dump(p, f, ensure_ascii=False, indent=2)
        upload_targets.append(f"data/{json_filename}")
            
    # インデックス保存（サマリー - source, local_images含む）
    def _norm_stations(stations):
        if isinstance(stations, list):
            return [str(s).strip() for s in stations if str(s).strip()]
        if isinstance(stations, str) and stations.strip():
            return [s.strip() for s in re.split(r'[\n,、]', stations) if s.strip()]
        return []
    summary_new = [{
        "id": p["id"],
        "source": SOURCE_ID,
        "title": p["title"], 
        "rent": p.get("rent") if p.get("rent") is not None else "",
        "management_fee": p.get("management_fee", ""),
        "deposit": p.get("deposit", ""),
        "key_money": p.get("key_money", ""),
        "address": p.get("address") or "", 
        "stations": _norm_stations(p.get("stations")),
        "layout": p["layout"], 
        "area": p["area"],
        "floor": p.get("floor", ""),
        "built_date": p.get("built_date", ""),
        "structure": p.get("structure", ""),
        "direction": p.get("direction", ""),
        "available_date": p.get("available_date", ""),
        "contract_period": p.get("contract_period", ""),
        "parking": p.get("parking", ""),
        "renewal_fee": p.get("renewal_fee", ""),
        "insurance": p.get("insurance", ""),
        "ad_fee": p.get("ad_fee", ""),
        "transaction_type": p.get("transaction_type", ""),
        "guarantee_company": p.get("guarantee_company", ""),
        "remarks": p.get("remarks", ""),
        "preferred_conditions": p.get("preferred_conditions", ""),
        "viewing_start_date": p.get("viewing_start_date", ""),
        "viewing_notes": p.get("viewing_notes", ""),
        "facilities": p.get("facilities", []),
        "building_name": p["building_name"], 
        "room_number": p["room_number"],
        "local_images": p.get("local_images", []),
        "thumbnail": p["local_images"][0] if p["local_images"] else None,
        "image_count": len(p["local_images"]) if p["local_images"] else len(p.get("image_urls", [])),
        "detail_url": p.get("detail_url", ""),
        "listed_area": p.get("listed_area", area_name),
        "scraped_at": p.get("scraped_at", "")
    } for p in new_items]
    
    final_list = existing + summary_new
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)
        
    print(f"  [保存完了] 新規追加: {len(new_items)}件 (現在の総数: {len(final_list)}件)")

    # R2へアップロード（個別JSONは直接、インデックスはマージしてアップロード）
    if R2_UPLOAD_ENABLED and upload_targets:
        print(f"[R2] JSONデータのアップロードを開始します ({len(upload_targets)}ファイル)")
        upload_files_to_r2(upload_targets)
        upload_merged_index_to_r2()


def load_existing_properties() -> dict:
    """既存データをロードしてURL辞書を作成。R2優先だが失敗時はローカルへフォールバック。"""
    res = {'by_url': {}, 'all_ids': set()}

    if R2_UPLOAD_ENABLED:
        try:
            all_data = download_r2_index_raw()
            if all_data is not None:
                my_data, _ = split_index_by_source(all_data)
                for item in my_data:
                    item_id = item.get('id')
                    if item_id:
                        res['all_ids'].add(item_id)
                    if item.get('detail_url'):
                        res['by_url'][item['detail_url']] = item
                if my_data:
                    with open(INDEX_FILE, "w", encoding="utf-8") as f:
                        json.dump(my_data, f, ensure_ascii=False, indent=2)
                need_fallback_urls = bool(res['all_ids']) and (not res['by_url'])
                if LOAD_DETAIL_URL_FALLBACK or need_fallback_urls:
                    build_url_dict_from_local_detail_json(my_data, res['by_url'])
                if ENABLE_TIMING_LOG:
                    print(f"  [計測][既存読込] R2優先参照 ids={len(res['all_ids'])} urls={len(res['by_url'])}")
                if res['all_ids'] or res['by_url']:
                    return res
                print("[WARN] R2にitanji既存データが見つからないためローカルへフォールバックします")
            else:
                print("[WARN] R2インデックス取得失敗のためローカルへフォールバックします")
        except Exception as e:
            print(f"[WARN] R2からのインデックス取得に失敗: {e} → ローカルへフォールバック")

    if os.path.exists(INDEX_FILE):
        try:
            t0 = time.perf_counter()
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                item_id = item.get('id')
                if item_id:
                    res['all_ids'].add(item_id)
                if item.get('detail_url'):
                    res['by_url'][item['detail_url']] = item
            if ENABLE_TIMING_LOG:
                print(f"  [計測][既存読込] ローカル indexロード={time.perf_counter() - t0:.2f}s ids={len(res['all_ids'])} urls={len(res['by_url'])}")
            need_fallback_urls = bool(res['all_ids']) and (not res['by_url'])
            if LOAD_DETAIL_URL_FALLBACK or need_fallback_urls:
                t1 = time.perf_counter()
                build_url_dict_from_local_detail_json(data, res['by_url'])
                if ENABLE_TIMING_LOG:
                    print(f"  [計測][既存読込] detail補完={time.perf_counter() - t1:.2f}s urls={len(res['by_url'])}")
        except Exception:
            pass
    return res


def signal_handler(sig, frame):
    global interrupt_flag
    print("\n[中断] 処理を停止します...")
    interrupt_flag = True
    # 新規キュー投入のみ止める（進行中タスクは wait_all_uploads() で待機して整合性を保つ）
    shutdown_upload_executor(wait=False)


# ============================================================
# メイン処理
# ============================================================

def main():
    global interrupt_flag
    signal.signal(signal.SIGINT, signal_handler)
    if OUTPUT_SUFFIX:
        configure_output_dir(os.path.join("output", OUTPUT_SUFFIX))
    setup_dirs()
    migrate_from_shared_index()

    print("="*60)
    print("【高速化・差分更新版】イタンジBB スクレイピング")
    print(f"モード: {'差分更新 (新規のみ取得)' if UPDATE_MODE else '全件取得'}")
    if UPDATE_EXISTING_DATA:
        print("      + 既存物件データ更新: ON（詳細情報を再取得）")
    if WORKER_COUNT > 1:
        print(f"      + 並列ワーカー: {WORKER_ID + 1}/{WORKER_COUNT}（分割取得）")
    print("="*60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        configure_context(context)
        page = context.new_page()

        # 1. ログイン
        if not auto_login(page):
            return

        # 2. エリア分割（並列ワーカー用）
        search_targets = split_targets_for_worker(SEARCH_TARGETS, WORKER_ID, WORKER_COUNT)

        # 2. 既存データ読み込み
        existing_data = None
        allow_cleanup_for_run = True
        if UPDATE_MODE:
            print("\n[準備] 既存データを読み込んでいます...")
            existing_data = load_existing_properties()
            print(f"  -> {len(existing_data['all_ids'])} 件の既存データを認識")
            if len(existing_data['all_ids']) == 0:
                allow_cleanup_for_run = False
                print("[SAFEGUARD] 既存データ0件のため、この実行では削除判定を無効化します")

        # 3. 各エリアを処理
        for idx, area in enumerate(search_targets):
            if interrupt_flag: break
            
            # 区ごと削除判定用：この区スキャン前のURL集合を保存
            prev_found = set(all_found_urls)

            print(f"\n--- 【{area}】({idx + 1}/{len(search_targets)}) の処理開始 ---")
            
            # 2番目以降のエリアはリスト検索ページに直接移動
            skip_list_search = False
            if idx > 0:
                print("リスト検索ページに移動中...")
                page.goto(LIST_SEARCH_URL, wait_until="load", timeout=60000)
                page.wait_for_timeout(2000)
                skip_list_search = True  # 既にリスト検索画面にいるので、リスト検索ボタンのクリックをスキップ
            
            # 検索実行
            if not auto_search(page, area, skip_list_search=skip_list_search):
                print(f"[ERROR] 【{area}】の検索に失敗しました")
                continue
            
            # ★ URL収集 (新規と既存を分離)
            print(f"  URL収集中...")
            new_urls, existing_urls = collect_property_urls_all_pages(page, existing_data)
            
            if not new_urls and not existing_urls:
                print(f"  【{area}】処理対象の物件はありませんでした。")
                area_found = set(all_found_urls) - prev_found
                print(
                    f"  [市区町村完了] 【{area}】"
                    f"一覧出現URL: {len(area_found)}件 / 新規URL: 0件 / 既存URL: 0件"
                )
                # ※ 区ごとの削除判定は廃止。全区スキャン後に最終一括クリーンアップで実施する。
                # 次のエリアのためにリスト検索ページに戻る
                if idx < len(search_targets) - 1:
                    page.goto(LIST_SEARCH_URL, wait_until="load", timeout=60000)
                    page.wait_for_timeout(2000)
                continue
            
            # ★ 新規物件の詳細取得（画像もDL）
            if new_urls:
                scrape_properties_parallel_multi_tabs(context, new_urls, area)
            else:
                print(f"  【{area}】新規物件はありませんでした。")
            
            # ★ 既存物件のデータ更新（画像はスキップ）
            if existing_urls and (UPDATE_EXISTING_DATA or UPDATE_EXISTING_META_ONLY):
                print(f"  更新対象: {len(existing_urls)}件")
                update_existing_properties(context, existing_urls, existing_data, meta_only=UPDATE_EXISTING_META_ONLY and not UPDATE_EXISTING_DATA)
            elif UPDATE_EXISTING_DATA or UPDATE_EXISTING_META_ONLY:
                print("  更新対象: 0件")
            
            # ★ 区ごとの進捗ログ（削除判定は全区完了後に一括で実施）
            area_found = set(all_found_urls) - prev_found
            print(
                f"  [市区町村完了] 【{area}】"
                f"一覧出現URL: {len(area_found)}件 / 新規URL: {len(new_urls)}件 / 既存URL: {len(existing_urls)}件"
            )
            
            # 次のエリアのためにリスト検索ページに戻る（最後のエリア以外）
            if idx < len(search_targets) - 1:
                print(f"\n次のエリアのため、リスト検索ページに戻ります...")
                page.goto(LIST_SEARCH_URL, wait_until="load", timeout=60000)
                page.wait_for_timeout(2000)

        browser.close()

        # ★ バックグラウンドアップロードの完了を待機
        wait_all_uploads()
        shutdown_upload_executor(wait=True)

        # ★ スクレイピング結果を保存（募集終了判定用・マージ用）
        save_scraping_results()

        # ★ 全区スキャン完了後に一括で募集終了判定（区ごとの中間削除は廃止済み）
        if WORKER_COUNT <= 1 and not interrupt_flag and allow_cleanup_for_run:
            print("\n[cleanup] Running final full cleanup for single-worker mode...")
            cleanup_ended_properties()
        elif WORKER_COUNT <= 1 and not interrupt_flag:
            print("\n[cleanup] skip final cleanup (safeguard: existing data was empty at startup)")

        if WORKER_COUNT <= 1:
            auto_commit_property_data()

        print("\n[完了] 全ての処理が終了しました。")


def save_scraping_results():
    """スクレイピングで見つかったURL情報を保存（募集終了判定用）"""
    global all_found_urls, applied_urls

    # 今回のスクレイピングで見つかった募集中URL
    scraped_urls_file = os.path.join(OUTPUT_DIR, "scraped_urls.json")
    with open(scraped_urls_file, "w", encoding="utf-8") as f:
        json.dump(list(all_found_urls), f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 募集中URL: {len(all_found_urls)}件 -> {scraped_urls_file}")

    # 申込ありとしてスキップしたURL
    applied_urls_file = os.path.join(OUTPUT_DIR, "applied_urls.json")
    with open(applied_urls_file, "w", encoding="utf-8") as f:
        json.dump(list(applied_urls), f, ensure_ascii=False, indent=2)
    print(f"[保存] 申込ありURL: {len(applied_urls)}件 -> {applied_urls_file}")

    # R2にもURL情報をアップロード
    if R2_UPLOAD_ENABLED:
        upload_files_to_r2(["scraped_urls.json", "applied_urls.json"])


def cleanup_ended_properties_for_area(area_name: str, area_found_urls: set):
    """指定市区町村の募集終了物件のみクリーンアップ（市区町村スキャン完了直後に呼ぶ）"""
    BACKUP_DIR = os.path.join(OUTPUT_DIR, "backup")
    ENDED_DIR = os.path.join(OUTPUT_DIR, "ended_properties")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    os.makedirs(ENDED_DIR, exist_ok=True)

    print(f"\n[削除判定] 【{area_name}】市区町村ブロックの判定を開始")
    print(f"  - 今回一覧に出現したURL: {len(area_found_urls)}件")
    if SKIP_AREA_CLEANUP_WHEN_FOUND_EMPTY and len(area_found_urls) == 0:
        print(f"[削除判定] 【{area_name}】スキップ: 一覧出現URLが0件のため安全停止（誤削除防止）")
        return

    if not os.path.exists(INDEX_FILE):
        print("  - 判定スキップ: properties_index.json が存在しません")
        return
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            properties_index = json.load(f)
    except Exception:
        print("  - 判定スキップ: properties_index.json の読み込みに失敗しました")
        return

    area_found_normalized = {normalize_url(url) for url in area_found_urls}
    area_props = []
    by_address = 0
    by_listed_area = 0
    for p in properties_index:
        address = p.get("address") or ""
        listed_area = p.get("listed_area") or ""
        if area_name in address:
            area_props.append(p)
            by_address += 1
            continue
        if listed_area == area_name:
            area_props.append(p)
            by_listed_area += 1
    print(
        f"  - 既存インデックス内（住所一致: {by_address}件 / listed_area一致: {by_listed_area}件）: "
        f"{len(area_props)}件"
    )
    if not area_props:
        print(f"[削除判定] 【{area_name}】対象物件なし（削除0件）")
        return

    ended_properties = []
    for prop in area_props:
        prop_url = prop.get("detail_url", "")
        prop_norm_url = normalize_url(prop_url)
        reason = ""
        if prop_url and prop_norm_url not in area_found_normalized:
            reason = "今回のスクレイピングで未検出"
        if reason:
            ended_properties.append((prop, reason))

    if not ended_properties:
        print(f"[削除判定] 【{area_name}】未検出なし（削除0件 / 継続{len(area_props)}件）")
        return

    # URL優先で除外対象を確定（id重複時の過剰削除を防ぐ）
    ended_url_keys = {
        normalize_url(prop.get("detail_url", ""))
        for prop, _ in ended_properties
        if prop.get("detail_url")
    }
    ended_id_keys = {
        prop.get("id", "")
        for prop, _ in ended_properties
        if not prop.get("detail_url")
    }
    active_properties = []
    for p in properties_index:
        p_url_key = normalize_url(p.get("detail_url", ""))
        if p_url_key and p_url_key in ended_url_keys:
            continue
        if (not p_url_key) and p.get("id", "") in ended_id_keys:
            continue
        active_properties.append(p)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"properties_index.json.{area_name}_{timestamp}.bak")
    try:
        shutil.copy2(INDEX_FILE, backup_path)
    except Exception:
        pass

    deleted_local_images = 0
    deleted_detail_json = 0
    r2_delete_keys = set()
    r2_delete_prefixes = set()
    ended_details = []
    for prop, reason in ended_properties:
        prop_id = prop.get("id", "")
        ended_details.append({**prop, "ended_reason": reason, "ended_at": datetime.now().isoformat()})
        local_images = prop.get("local_images", []) or []
        deleted_local_images += purge_local_image_artifacts(prop_id, local_images)
        for rel_path in local_images:
            rel_path = rel_path.replace("\\", "/")
            r2_delete_keys.add(rel_path)
        if prop_id:
            r2_delete_prefixes.add(f"images/{prop_id}/")
        detail_json_path = os.path.join(DATA_DIR, f"{prop_id}.json")
        if os.path.exists(detail_json_path):
            ended_json_path = os.path.join(ENDED_DIR, f"{prop_id}.json")
            try:
                with open(detail_json_path, "r", encoding="utf-8") as f:
                    detail_data = json.load(f)
                detail_data["ended_reason"] = reason
                detail_data["ended_at"] = datetime.now().isoformat()
                with open(ended_json_path, "w", encoding="utf-8") as f:
                    json.dump(detail_data, f, ensure_ascii=False, indent=2)
                os.remove(detail_json_path)
                deleted_detail_json += 1
            except Exception:
                pass

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(active_properties, f, ensure_ascii=False, indent=2)

    if r2_delete_prefixes:
        for prefix in sorted(r2_delete_prefixes):
            for k in list_r2_keys(prefix):
                r2_delete_keys.add(k)
    if r2_delete_keys:
        delete_r2_keys(sorted(r2_delete_keys))
    if R2_UPLOAD_ENABLED:
        upload_merged_index_to_r2()

    ended_list_file = os.path.join(ENDED_DIR, f"ended_{area_name}_{timestamp}.json")
    try:
        with open(ended_list_file, "w", encoding="utf-8") as f:
            json.dump(ended_details, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    area_active_count = max(0, len(area_props) - len(ended_properties))
    print(
        f"[削除判定] 【{area_name}】完了: 削除{len(ended_properties)}件 / 継続{area_active_count}件 "
        f"(理由: 未検出)"
    )
    print(
        f"  - 付随削除: 詳細JSON {deleted_detail_json}件, "
        f"ローカル画像 {deleted_local_images}件, R2キー {len(r2_delete_keys)}件"
    )


def cleanup_ended_properties():
    """募集終了物件のクリーンアップ（スクレイピング後に自動実行）※マージスクリプト等から呼ばれる一括用"""
    global all_found_urls

    SCRAPED_URLS_FILE = os.path.join(OUTPUT_DIR, "scraped_urls.json")
    BACKUP_DIR = os.path.join(OUTPUT_DIR, "backup")
    ENDED_DIR = os.path.join(OUTPUT_DIR, "ended_properties")

    print("\n" + "=" * 60)
    print("募集終了物件のクリーンアップ")
    print("=" * 60)

    # ディレクトリ作成
    os.makedirs(BACKUP_DIR, exist_ok=True)
    os.makedirs(ENDED_DIR, exist_ok=True)

    # 既存データ読み込み
    if not os.path.exists(INDEX_FILE):
        print("[WARN] properties_index.json が見つかりません")
        return

    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            properties_index = json.load(f)
    except:
        print("[ERROR] properties_index.json の読み込みに失敗")
        return

    print(f"\n現在の物件数: {len(properties_index)}件")

    # 正規化したURL
    scraped_normalized = {normalize_url(url) for url in all_found_urls}

    # 募集終了物件を判定（itanji以外のソースは対象外）
    active_properties = []
    ended_properties = []

    for prop in properties_index:
        prop_source = prop.get("source", "")
        # itanji以外のソース（es_squareなど）はスキップして残す
        if prop_source and prop_source != SOURCE_ID:
            active_properties.append(prop)
            continue

        prop_url = prop.get('detail_url', '')
        prop_norm_url = normalize_url(prop_url)

        is_ended = False
        reason = ""

        if prop_url and prop_norm_url not in scraped_normalized:
            is_ended = True
            reason = "今回のスクレイピングで未検出"

        if is_ended:
            ended_properties.append((prop, reason))
        else:
            active_properties.append(prop)

    print(f"募集中物件: {len(active_properties)}件")
    print(f"募集終了物件: {len(ended_properties)}件")
    print(f"削除対象: {len(ended_properties)}件")

    if not ended_properties:
        print("\n[完了] 削除する物件はありません")
        return

    # 安全ガード: itanji物件の50%以上を削除しようとした場合は中止
    itanji_total = len(ended_properties) + len([p for p in active_properties if (p.get("source", "") or SOURCE_ID) == SOURCE_ID])
    if itanji_total > 0 and len(ended_properties) / itanji_total > 0.5:
        print(f"\n[SAFEGUARD] 削除対象が多すぎます ({len(ended_properties)}/{itanji_total} = {len(ended_properties)/itanji_total:.0%})")
        print("  スクレイピングが不完全だった可能性があります。削除をスキップします。")
        print("  手動で削除する場合は ALLOW_EMPTY_ITANJI_INDEX_UPLOAD=1 を設定してください。")
        return

    # バックアップ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"properties_index.json.{timestamp}.bak")
    shutil.copy2(INDEX_FILE, backup_path)
    print(f"\n[BACKUP] {backup_path}")

    # 募集終了物件を保存
    ended_details = []
    deleted_local_images = 0
    deleted_detail_json = 0
    r2_delete_keys = set()
    r2_delete_prefixes = set()
    for prop, reason in ended_properties:
        prop_id = prop.get('id', '')
        ended_details.append({
            **prop,
            "ended_reason": reason,
            "ended_at": datetime.now().isoformat()
        })

        # 画像削除（ローカル + R2）
        local_images = prop.get("local_images", []) or []
        deleted_local_images += purge_local_image_artifacts(prop_id, local_images)
        for rel_path in local_images:
            rel_path = rel_path.replace("\\", "/")
            r2_delete_keys.add(rel_path)
        if prop_id:
            r2_delete_prefixes.add(f"images/{prop_id}/")
        detail_json_path = os.path.join(DATA_DIR, f"{prop_id}.json")
        if os.path.exists(detail_json_path):
            ended_json_path = os.path.join(ENDED_DIR, f"{prop_id}.json")
            try:
                with open(detail_json_path, "r", encoding="utf-8") as f:
                    detail_data = json.load(f)
                detail_data["ended_reason"] = reason
                detail_data["ended_at"] = datetime.now().isoformat()

                with open(ended_json_path, "w", encoding="utf-8") as f:
                    json.dump(detail_data, f, ensure_ascii=False, indent=2)

                os.remove(detail_json_path)
                deleted_detail_json += 1
            except:
                pass

        # 念のため空フォルダは削除

    # 募集終了リスト保存
    ended_list_file = os.path.join(ENDED_DIR, f"ended_{timestamp}.json")
    with open(ended_list_file, "w", encoding="utf-8") as f:
        json.dump(ended_details, f, ensure_ascii=False, indent=2)

    # properties_index.json 更新
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(active_properties, f, ensure_ascii=False, indent=2)

    print(f"[OK] 更新完了")
    print(f"  - 削除: {len(ended_properties)}件")
    print(f"  - 残存: {len(active_properties)}件")

    # R2削除（prefix指定の場合は一覧取得して削除）
    if r2_delete_prefixes:
        for prefix in sorted(r2_delete_prefixes):
            keys = list_r2_keys(prefix)
            for k in keys:
                r2_delete_keys.add(k)
    if r2_delete_keys:
        delete_r2_keys(sorted(r2_delete_keys))
    if deleted_local_images or r2_delete_keys:
        print(f"[画像削除] ローカル: {deleted_local_images}枚 / R2: {len(r2_delete_keys)}枚")
    print(f"[削除実績] 詳細JSON: {deleted_detail_json}件")

    # サマリー
    reason_counts = {}
    for _, reason in ended_properties:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print("\n削除された物件の内訳（理由別）:")
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count}件")

    # R2のマージインデックスを更新
    if R2_UPLOAD_ENABLED:
        upload_merged_index_to_r2()


if __name__ == "__main__":
    main()
