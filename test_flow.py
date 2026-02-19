#!/usr/bin/env python3
"""
test_flow.py  ――  スクレイピング後フローのテストスクリプト

既存のスクレイピングデータ（assets/data.json / saved_images）を使って
「Slack選定 → 投稿画像生成 → Google Drive → Slack通知」のフローを
何度でも繰り返しテストできる。

─────────────────────────────────────────────
使い方
─────────────────────────────────────────────
  python test_flow.py               # 状態確認 → Slack選定 → 画像生成（フル実行）
  python test_flow.py --skip-slack  # 選定済みデータを使って画像生成のみ
  python test_flow.py --reset       # 選定・出力をリセット → フル実行
  python test_flow.py --status      # 現在の状態を表示して終了
  python test_flow.py --no-drive    # Drive アップロードをスキップして実行
  python test_flow.py --reset --skip-slack  # リセット後、選定なしで画像生成のみ

─────────────────────────────────────────────
環境変数（.env）
─────────────────────────────────────────────
  SLACK_BOT_TOKEN / SLACK_APP_TOKEN / SLACK_CHANNEL  … Slack選定に必要
  GOOGLE_DRIVE_CREDENTIALS_JSON / GOOGLE_DRIVE_FOLDER_ID … Drive連携に必要
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

# ── パス定数 ─────────────────────────────────────────
DATA_PATH        = ROOT / "assets" / "data.json"
SELECTIONS_PATH  = ROOT / "assets" / "slack_selections.json"
OUTPUT_ROOT      = ROOT / "output" / "投稿用出力"
ADOPTED_FOLDER   = OUTPUT_ROOT / "採用"
BOTS_FOLDER      = OUTPUT_ROOT / "ボツ"
SAVED_IMAGES_ROOT = ROOT / "output" / "itanji_video" / "saved_images"
IMAGE_EXTS       = {".jpg", ".jpeg", ".png", ".webp"}


# ── ログ設定 ─────────────────────────────────────────
def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ── 状態表示 ─────────────────────────────────────────
def show_status() -> None:
    sep = "─" * 52
    print(f"\n{sep}")
    print("  現在の状態")
    print(sep)

    # assets/data.json
    if DATA_PATH.exists():
        try:
            records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
            print(f"  物件データ        : {len(records)} 件  ({DATA_PATH})")
        except Exception:
            print(f"  物件データ        : 読み込みエラー")
    else:
        print(f"  物件データ        : なし  ← スクレイピングが必要")

    # saved_images
    img_found = 0
    if SAVED_IMAGES_ROOT.exists():
        for d in SAVED_IMAGES_ROOT.rglob("*"):
            if d.is_file() and d.suffix.lower() in IMAGE_EXTS:
                img_found += 1
    print(f"  保存済み画像      : {img_found} ファイル  ({SAVED_IMAGES_ROOT})")

    # slack_selections.json
    if SELECTIONS_PATH.exists():
        try:
            sels = json.loads(SELECTIONS_PATH.read_text(encoding="utf-8"))
            adopted = sum(1 for v in sels.values() if _to_int(v, -99) >= 0)
            bots    = sum(1 for v in sels.values() if _to_int(v, -99) == -1)
            print(f"  Slack選定         : 採用={adopted} 件 / ボツ={bots} 件  ({SELECTIONS_PATH})")
        except Exception:
            print(f"  Slack選定         : 読み込みエラー")
    else:
        print(f"  Slack選定         : 未実施")

    # 出力フォルダ
    adopted_cnt = len(list(ADOPTED_FOLDER.iterdir())) if ADOPTED_FOLDER.exists() else 0
    bots_cnt    = len(list(BOTS_FOLDER.iterdir()))    if BOTS_FOLDER.exists() else 0
    print(f"  採用フォルダ      : {adopted_cnt} 件  ({ADOPTED_FOLDER})")
    print(f"  ボツフォルダ      : {bots_cnt} 件  ({BOTS_FOLDER})")

    # 環境変数チェック
    slack_ok = all(os.getenv(k, "").strip() for k in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_CHANNEL"))
    drive_ok = bool(os.getenv("GOOGLE_DRIVE_CREDENTIALS_JSON", "").strip() and
                    Path(os.getenv("GOOGLE_DRIVE_CREDENTIALS_JSON", "")).expanduser().exists())
    print(f"  Slack設定         : {'[OK] 有効' if slack_ok else '[NG] 未設定（.env を確認）'}")
    print(f"  Google Drive設定  : {'[OK] 有効' if drive_ok else '[NG] 未設定（.env を確認）'}")
    print(sep + "\n")


def _to_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


# ── データ確認 ────────────────────────────────────────
def check_data() -> bool:
    if not DATA_PATH.exists():
        logging.error("assets/data.json が見つかりません。先にスクレイピングを実行してください。")
        return False
    try:
        records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logging.error("data.json の読み込みエラー: %s", e)
        return False
    if not records:
        logging.error("物件データが 0 件です。")
        return False

    # 画像ファイル確認
    img_ok = 0
    for rec in records:
        pid = str(rec.get("id") or "")
        for root in (SAVED_IMAGES_ROOT / "adopted" / pid, SAVED_IMAGES_ROOT / "bots" / pid):
            if root.exists() and any(f.suffix.lower() in IMAGE_EXTS for f in root.iterdir() if f.is_file()):
                img_ok += 1
                break

    if img_ok == 0:
        logging.error("画像ファイルが 1 件も見つかりません。")
        logging.error("ITANJI_VIDEO_SAVE_IMAGES=1 でスクレイピングを再実行してください。")
        return False

    logging.info("データ確認OK: %d 件 / 画像あり %d 件", len(records), img_ok)
    return True


# ── リセット ─────────────────────────────────────────
def reset_selections(backup: bool = True) -> None:
    """Slack選定ファイルをリセット（バックアップ付き）"""
    if not SELECTIONS_PATH.exists():
        logging.info("リセット: 選定ファイルなし（スキップ）")
        return
    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = SELECTIONS_PATH.with_name(f"slack_selections.bak_{stamp}.json")
        shutil.copy2(SELECTIONS_PATH, bak)
        logging.info("選定バックアップ: %s", bak.name)
    SELECTIONS_PATH.unlink()
    logging.info("Slack選定をリセットしました")


def reset_output() -> None:
    """出力フォルダをリセット（スクレイピングデータは保持）"""
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
        logging.info("出力フォルダを削除しました: %s", OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logging.info("出力フォルダを再作成しました")


# ── サブプロセス実行 ──────────────────────────────────
def run_script(script: str, env: dict) -> bool:
    logging.info("実行: %s", script)
    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=str(ROOT),
        env=env,
    )
    return result.returncode == 0


# ── メイン ────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="スクレイピング後フロー テストスクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python test_flow.py               Slack選定 → 画像生成（フル実行）
  python test_flow.py --skip-slack  既存の選定で画像生成のみ
  python test_flow.py --reset       リセット → フル実行
  python test_flow.py --status      状態表示のみ
  python test_flow.py --no-drive    Driveアップロードをスキップ
        """,
    )
    parser.add_argument("--reset",       action="store_true", help="Slack選定と出力フォルダをリセットしてから実行")
    parser.add_argument("--skip-slack",  action="store_true", help="Slack選定をスキップ（既存の選定を使用）")
    parser.add_argument("--no-drive",    action="store_true", help="Google Driveアップロードをスキップ")
    parser.add_argument("--status",      action="store_true", help="現在の状態を表示して終了")
    args = parser.parse_args()

    setup_logger()
    show_status()

    if args.status:
        return

    # データ確認
    if not check_data():
        sys.exit(1)

    # リセット
    if args.reset:
        reset_selections(backup=True)
        reset_output()
        print()

    # 環境変数セット
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Drive スキップ
    if args.no_drive:
        env["GOOGLE_DRIVE_CREDENTIALS_JSON"] = ""
        logging.info("--no-drive: Google Driveアップロードをスキップします")

    # ── Step 1: Slack選定 ────────────────────────────
    if args.skip_slack:
        logging.info("[Step 1/2] Slack選定をスキップ（--skip-slack）")
        if not SELECTIONS_PATH.exists():
            logging.warning("  選定ファイルがありません。デフォルト選定で画像生成します。")
    else:
        slack_ok = all(env.get(k, "").strip() for k in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_CHANNEL"))
        if not slack_ok:
            logging.warning("[Step 1/2] Slackトークン未設定 → Slack選定をスキップ")
            logging.warning("  .env に SLACK_BOT_TOKEN / SLACK_APP_TOKEN / SLACK_CHANNEL を設定してください")
        else:
            # 毎回全物件を再選定させる（テスト用）
            env["SLACK_SKIP_ALREADY_SELECTED"] = "0"
            logging.info("[Step 1/2] Slack選定を開始します...")
            ok = run_script("slack_selector.py", env)
            if not ok:
                logging.error("Slack選定が失敗またはタイムアウトしました")
                ans = input("  続けて画像生成を実行しますか？ [y/N]: ").strip().lower()
                if ans != "y":
                    sys.exit(1)

    # ── Step 2: 画像生成 + Drive + Slack通知 ─────────
    logging.info("[Step 2/2] 投稿画像生成を開始します...")
    ok = run_script("main.py", env)
    if not ok:
        logging.error("main.py が失敗しました")
        sys.exit(1)

    # 完了後の状態表示
    show_status()
    logging.info("テスト完了")


if __name__ == "__main__":
    main()
