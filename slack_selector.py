#!/usr/bin/env python3
"""
Slack 画像選定スクリプト

スクレイピング後・main.py実行前に実行する。
各物件の候補画像を1枚のカタログ画像にまとめてSlackへ送信し、
ユーザーがボタンをタップして選定番号を送り返す。
選定結果は assets/slack_selections.json に保存される。

使い方:
    python slack_selector.py

必要な環境変数 (.env に設定):
    SLACK_BOT_TOKEN   - xoxb- から始まるボットトークン
    SLACK_APP_TOKEN   - xapp- から始まるアプリトークン（Socket Mode用）
    SLACK_CHANNEL     - 投稿先チャンネルID（例: C01234ABCDE）

任意環境変数:
    SLACK_SELECTION_TIMEOUT      - 選定待ちタイムアウト秒数（デフォルト: 600）
    SLACK_SKIP_ALREADY_SELECTED  - 既選定済み物件をスキップ (1=有効, デフォルト: 1)
    POSTGEN_GEMINI_COVER_PICK_MAX_IMAGES - カタログに載せる最大画像枚数（デフォルト: 10）
"""

import json
import logging
import os
import re
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler
except ImportError:
    print("ERROR: slack_bolt がインストールされていません。")
    print("実行: pip install slack_bolt slack_sdk")
    sys.exit(1)

# プロジェクトルートをパスに追加してmain.pyをインポート
sys.path.insert(0, str(Path(__file__).parent))

try:
    from main import (
        DATA_PATH,
        GEMINI_COVER_PICK_MAX_IMAGES,
        create_candidate_catalog,
        find_local_cached_image,
        is_likely_floorplan_image_file,
        is_likely_floorplan_ref,
        load_records,
        ordered_cover_candidates,
        reorder_by_portrait_4x5,
        sanitize_filename,
    )
except ImportError as e:
    print(f"ERROR: main.py のインポートに失敗しました: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────
SELECTIONS_PATH = Path("assets/slack_selections.json")

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "").strip()
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN", "").strip()
SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL", "").strip()

SELECTION_TIMEOUT = int(os.getenv("SLACK_SELECTION_TIMEOUT", "600"))
SKIP_ALREADY_SELECTED = os.getenv("SLACK_SKIP_ALREADY_SELECTED", "1").lower() in ("1", "true", "yes")
MAX_CATALOG_IMAGES = GEMINI_COVER_PICK_MAX_IMAGES  # main.py と同じ上限

MAX_BUTTONS_PER_ROW = 5  # Slack の actions block は1行5要素まで


# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────

def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_existing_selections() -> Dict[str, int]:
    if SELECTIONS_PATH.exists():
        try:
            return json.loads(SELECTIONS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_selections(selections: Dict[str, int]) -> None:
    SELECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTIONS_PATH.write_text(
        json.dumps(selections, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def catalog_to_bytes(catalog_img) -> bytes:
    buf = BytesIO()
    catalog_img.save(buf, format="PNG")
    return buf.getvalue()


def safe_block_id(property_id: str, suffix: str = "") -> str:
    """Slack の block_id / action_id に使える安全な文字列（最大255文字）を返す。"""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", property_id)
    key = f"prop_{safe}{suffix}"
    return key[:255]


# ─────────────────────────────────────────────
# 候補画像収集
# ─────────────────────────────────────────────

def get_candidates_for_property(
    record: dict,
    max_images: int = 10,
) -> Tuple[List[Path], List[int]]:
    """
    物件レコードから候補画像のパスと0ベースインデックスを返す。
    間取り図は除外し、縦型・4:5に近い画像を先頭に並べる。
    """
    images = list(record.get("images") or [])
    if not images:
        return [], []

    property_id = str(record.get("id") or "")

    # 間取り図を URL 名から除外
    blocked = {i for i, ref in enumerate(images) if is_likely_floorplan_ref(str(ref))}

    # 間取り図を画像内容から除外
    for i, ref in enumerate(images):
        if i in blocked:
            continue
        local = find_local_cached_image(property_id, i + 1, image_ref=str(ref))
        if local is None:
            continue
        try:
            if is_likely_floorplan_image_file(local):
                blocked.add(i)
        except Exception:
            pass

    ordered = ordered_cover_candidates(images, record.get("image_metrics") or [], blocked)
    ordered = reorder_by_portrait_4x5(ordered, images, property_id)

    candidate_map: List[int] = []
    candidate_paths: List[Path] = []
    for idx in ordered:
        local = find_local_cached_image(property_id, idx + 1, image_ref=str(images[idx]))
        if local is None:
            continue
        candidate_paths.append(local)
        candidate_map.append(idx)
        if len(candidate_map) >= max_images:
            break

    return candidate_paths, candidate_map


# ─────────────────────────────────────────────
# Slack Block Kit 構築
# ─────────────────────────────────────────────

def build_selection_blocks(property_id: str, n_images: int, name: str) -> list:
    """????????????????????"""
    header_block = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*{name}*\n?????????????????? {n_images} ??",
        },
    }

    buttons = []
    for i in range(1, n_images + 1):
        buttons.append({
            "type": "button",
            "text": {"type": "plain_text", "text": str(i)},
            "value": f"{property_id}:{i}",
            "action_id": f"pick_{i}",
        })

    action_blocks = []
    for chunk_start in range(0, len(buttons), MAX_BUTTONS_PER_ROW):
        chunk = buttons[chunk_start : chunk_start + MAX_BUTTONS_PER_ROW]
        block_id = safe_block_id(property_id, f"_r{chunk_start}")
        action_blocks.append({
            "type": "actions",
            "block_id": block_id,
            "elements": chunk,
        })

    action_blocks.append({
        "type": "actions",
        "block_id": safe_block_id(property_id, "_reject"),
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "ボツ"},
                "style": "danger",
                "value": f"{property_id}:REJECT",
                "action_id": "pick_reject",
                "confirm": {
                    "title": {"type": "plain_text", "text": "この物件をボツにしますか？"},
                    "text": {"type": "mrkdwn", "text": "ボツにすると投稿画像の生成対象から除外されます。"},
                    "confirm": {"type": "plain_text", "text": "ボツにする"},
                    "deny": {"type": "plain_text", "text": "キャンセル"},
                },
            }
        ],
    })

    return [header_block] + action_blocks


def build_done_blocks(name: str, num: Optional[int] = None, is_bots: bool = False) -> list:
    msg = (f"✅ *{name}* は *ボツ* に設定しました" if is_bots else f"✅ *{name}* は *{num}番* を選択しました")
    return [{
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": msg,
        },
    }]


def main() -> None:
    setup_logger()

    # ─── 環境変数チェック ───
    errors = []
    if not SLACK_BOT_TOKEN:
        errors.append("SLACK_BOT_TOKEN が未設定です")
    if not SLACK_APP_TOKEN:
        errors.append("SLACK_APP_TOKEN が未設定です（Socket Mode用の xapp- トークン）")
    if not SLACK_CHANNEL:
        errors.append("SLACK_CHANNEL が未設定です（チャンネルID: C01234ABCDE 形式）")
    if errors:
        for e in errors:
            logging.error(e)
        logging.error(".env にSlackトークンを設定してから再実行してください")
        sys.exit(1)

    # ─── データ読み込み ───
    try:
        records = load_records(DATA_PATH)
    except Exception as e:
        logging.error("データ読み込みに失敗: %s", e)
        sys.exit(1)

    if not records:
        logging.info("物件データが0件。処理をスキップします。")
        sys.exit(0)

    logging.info("物件データ: %d件", len(records))

    # ─── 既存選定を読み込み ───
    existing: Dict[str, int] = load_existing_selections() if SKIP_ALREADY_SELECTED else {}

    # ─── 候補画像を収集 ───
    # pending_props: list of (record, candidate_paths, candidate_map)
    pending_props = []
    auto_selected: Dict[str, int] = {}  # 1枚しかない → 自動選定

    for rec in records:
        pid = str(rec.get("id") or "")

        if pid in existing and SKIP_ALREADY_SELECTED:
            logging.info("[%s] 既に選定済みのためスキップ", pid)
            continue

        paths, cmap = get_candidates_for_property(rec, max_images=MAX_CATALOG_IMAGES)
        if not paths:
            logging.warning("[%s] 候補画像が見つかりません。スキップします。", pid)
            continue

        if len(paths) == 1:
            # 候補1枚のみ → 自動選定（Slackに送らない）
            auto_selected[pid] = cmap[0]
            logging.info("[%s] 候補1枚のみのため自動選定: index=%s", pid, cmap[0])
            continue

        pending_props.append((rec, paths, cmap))

    # 自動選定を既存選定にマージ
    existing.update(auto_selected)

    if not pending_props:
        logging.info("Slack選定が必要な物件がありません。")
        save_selections(existing)
        logging.info("選定結果を保存: %s", SELECTIONS_PATH)
        sys.exit(0)

    logging.info("%d件の物件をSlackに送信します（タイムアウト: %ds）", len(pending_props), SELECTION_TIMEOUT)

    # ─── Slack App 初期化 ───
    selections: Dict[str, int] = dict(existing)
    selection_events: Dict[str, threading.Event] = {
        str(rec.get("id") or ""): threading.Event()
        for rec, _, _ in pending_props
    }
    # property_id → (candidate_map, message_channel, message_ts, name) for update
    prop_meta: Dict[str, dict] = {
        str(rec.get("id") or ""): {
            "cmap": cmap,
            "name": str(rec.get("name") or rec.get("id") or ""),
        }
        for rec, _, cmap in pending_props
    }

    app = App(token=SLACK_BOT_TOKEN)

    @app.action(re.compile(r"^pick_(\d+|reject)$"))
    def handle_pick(ack, body, client):
        ack()
        try:
            action = body["actions"][0]
            value = str(action.get("value", ""))
            # value format: "{property_id}:{image_number_1based}" or "{property_id}:REJECT"
            colon_idx = value.rfind(":")
            if colon_idx < 0:
                return
            pid = value[:colon_idx]
            selection_token = value[colon_idx + 1:]

            meta = prop_meta.get(pid)
            if meta is None:
                logging.warning("??????ID??????: %s", pid)
                return

            cmap = meta["cmap"]
            name = meta["name"]

            selected_num: Optional[int] = None
            is_bots = selection_token.upper() == "REJECT"
            if is_bots:
                selections[pid] = -1
                logging.info("[%s] Slack selection: rejected", pid)
            else:
                num = int(selection_token)  # 1-based
                if not (1 <= num <= len(cmap)):
                    logging.warning("[%s] Invalid selection number: %s", pid, num)
                    return
                idx = cmap[num - 1]  # 0-based index
                selections[pid] = idx
                selected_num = num
                logging.info("[%s] Slack selection: number=%s / image_index=%s", pid, num, idx)

            try:
                channel_id = body["channel"]["id"]
                message_ts = body["message"]["ts"]
                client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=(f"{name} はボツに設定されました" if is_bots else f"{name} は{selected_num}番を選択しました"),
                    blocks=build_done_blocks(name, selected_num, is_bots=is_bots),
                )
            except Exception as e:
                logging.warning("[%s] Failed to update selection message: %s", pid, e)

            ev = selection_events.get(pid)
            if ev:
                ev.set()

        except Exception as e:
            logging.warning("Error while handling Slack action: %s", e)

    # ─── Socket Mode 接続（ノンブロッキング）───
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    try:
        handler.connect()
        logging.info("Slack Socket Mode に接続しました")
        time.sleep(1.0)  # 接続安定待ち
    except Exception as e:
        logging.error("Socket Mode の接続に失敗しました: %s", e)
        sys.exit(1)

    # ─── カタログ画像を送信 ───
    web_client = app.client
    sent_ids: List[str] = []

    for rec, paths, cmap in pending_props:
        pid = str(rec.get("id") or "")
        name = prop_meta[pid]["name"]

        catalog_img = create_candidate_catalog(paths)
        if catalog_img is None:
            logging.warning("[%s] カタログ画像の生成に失敗。スキップします。", pid)
            selection_events[pid].set()  # ブロックしないようにイベントを解放
            continue

        img_bytes = catalog_to_bytes(catalog_img)

        # 画像アップロード
        try:
            web_client.files_upload_v2(
                channel=SLACK_CHANNEL,
                content=img_bytes,
                filename=f"{sanitize_filename(pid)}_catalog.png",
                initial_comment=f"📷 *{name}*",
            )
        except Exception as e:
            logging.error("[%s] 画像アップロードに失敗: %s", pid, e)
            selection_events[pid].set()
            continue

        # ボタンメッセージ送信
        blocks = build_selection_blocks(pid, len(paths), name)
        try:
            web_client.chat_postMessage(
                channel=SLACK_CHANNEL,
                blocks=blocks,
                text=f"{name}: カバー画像番号を選んでください",
            )
            sent_ids.append(pid)
            logging.info("[%s] Slackに送信しました (%d枚の候補)", pid, len(paths))
        except Exception as e:
            logging.error("[%s] ボタンメッセージの送信に失敗: %s", pid, e)
            selection_events[pid].set()

        time.sleep(0.3)  # Slack API レート制限対策

    # ─── 全選定を待機 ───
    if sent_ids:
        logging.info("Slackからの選択を待っています... (%d件未選定)", len(sent_ids))
        deadline = time.time() + SELECTION_TIMEOUT
        for pid in sent_ids:
            ev = selection_events.get(pid)
            if ev is None:
                continue
            remaining = max(0.0, deadline - time.time())
            if not ev.wait(timeout=remaining):
                meta = prop_meta.get(pid, {})
                logging.warning("[%s] タイムアウト: 選択されませんでした（%s）", pid, meta.get("name", ""))

    # ─── 結果保存 ───
    save_selections(selections)

    total = len(pending_props) + len(auto_selected)
    selected_count = sum(1 for pid in [str(r.get("id") or "") for r, _, _ in pending_props] if pid in selections)
    logging.info(
        "選定完了: %d/%d件 (自動選定: %d件) → %s",
        selected_count + len(auto_selected),
        total,
        len(auto_selected),
        SELECTIONS_PATH,
    )

    # Socket Mode を終了
    try:
        handler.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
