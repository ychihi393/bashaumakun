#!/usr/bin/env python3
"""
Slack 画像選定スクリプト（1物件1カタログ版）

スクレイピング後・main.py実行前に実行する。
物件ごとに候補画像カタログ（最大15枚）をSlackへ送信し、
文字入れに使う1枚をボタンで選択する。

ボツ = その物件は文字入れに値する画像がない → 物件ごと投稿から除外

使い方:
    python slack_selector.py

必要な環境変数 (.env に設定):
    SLACK_BOT_TOKEN   - xoxb- から始まるボットトークン
    SLACK_APP_TOKEN   - xapp- から始まるアプリトークン（Socket Mode用）
    SLACK_CHANNEL     - 投稿先チャンネルID

任意環境変数:
    SLACK_SELECTION_TIMEOUT      - 1物件あたりの選定待ちタイムアウト秒数（デフォルト: 600）
    SLACK_SKIP_ALREADY_SELECTED  - 既選定済み物件をスキップ (1=有効, デフォルト: 1)
"""

from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).parent))

try:
    from main import (
        DATA_PATH,
        create_property_catalog,
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
SLACK_CHANNEL   = os.environ.get("SLACK_CHANNEL", "").strip()

SELECTION_TIMEOUT     = int(os.getenv("SLACK_SELECTION_TIMEOUT", "600"))
SKIP_ALREADY_SELECTED = os.getenv("SLACK_SKIP_ALREADY_SELECTED", "1").lower() in ("1", "true", "yes")
MAX_CATALOG_IMAGES    = 15  # 1物件あたりの最大候補数


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
    catalog_img.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


# ─────────────────────────────────────────────
# 候補画像収集
# ─────────────────────────────────────────────

def get_candidates_for_property(
    record: dict,
    max_images: int = MAX_CATALOG_IMAGES,
) -> Tuple[List[Path], List[int]]:
    """
    物件レコードから候補画像のパスと0-basedインデックスを返す。
    - 間取り図は除外
    - 縦型4:5に近い画像を先頭に
    戻り値: (candidate_paths, candidate_map)
      candidate_map[i] = 元のimages配列での0-basedインデックス
    """
    images = list(record.get("images") or [])
    if not images:
        return [], []

    property_id = str(record.get("id") or "")

    blocked = {i for i, ref in enumerate(images) if is_likely_floorplan_ref(str(ref))}

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

    candidate_paths: List[Path] = []
    candidate_map: List[int] = []
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

def build_property_blocks(
    pid: str,
    cmap: List[int],
    name: str,
    prop_idx: int,
    selection: Optional[int] = None,
) -> List[dict]:
    """
    1物件の選択ボタンブロックを構築する。
    ボタンは1行5個まで（Slack制限）。最大15個の候補 + ボツボタン1個。
    番号はカタログ画像の左上バッジと完全一致（1〜len(cmap)）。
    """
    blocks: List[dict] = []
    n = len(cmap)

    sel_label = "未選択"
    if selection is not None:
        if selection == -1:
            sel_label = "ボツ"
        elif selection in cmap:
            sel_label = f"{cmap.index(selection) + 1}番を選択済み"
        else:
            sel_label = "選択済み"

    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                f"*{name}*  ({n}枚)  —  {sel_label}\n"
                f"文字入れに使う画像番号を選んでください"
            ),
        },
    })

    # 候補ボタンを5個ずつの actions ブロックに分割
    elements_rows: List[List[dict]] = []
    row: List[dict] = []
    for cand_i, img_idx in enumerate(cmap):
        cand_num = cand_i + 1
        is_sel = (selection == img_idx)
        btn_text = f"[{cand_num}]" if is_sel else str(cand_num)
        btn: dict = {
            "type": "button",
            "text": {"type": "plain_text", "text": btn_text},
            "value": f"pick:{pid}:{img_idx}",
            "action_id": f"pick_{prop_idx}_{cand_i}",
        }
        if not is_sel:
            btn["style"] = "primary"
        row.append(btn)
        if len(row) == 5:
            elements_rows.append(row)
            row = []
    if row:
        elements_rows.append(row)

    # ボツボタン
    is_rejected = (selection == -1)
    bots_text = "[ボツ]" if is_rejected else "ボツ"
    bots_btn: dict = {
        "type": "button",
        "text": {"type": "plain_text", "text": bots_text},
        "value": f"reject:{pid}",
        "action_id": f"reject_{prop_idx}",
        "style": "danger",
    }
    if not is_rejected:
        bots_btn["confirm"] = {
            "title": {"type": "plain_text", "text": "ボツにしますか？"},
            "text": {
                "type": "mrkdwn",
                "text": f"*{name}*\n文字入れに値する画像がない場合、物件ごと投稿から除外されます。",
            },
            "confirm": {"type": "plain_text", "text": "ボツにする"},
            "deny": {"type": "plain_text", "text": "キャンセル"},
        }

    # ボツボタンを最後の行に追加（空きがあれば同行、なければ新行）
    if elements_rows and len(elements_rows[-1]) < 5:
        elements_rows[-1].append(bots_btn)
    else:
        elements_rows.append([bots_btn])

    for row_i, row_elements in enumerate(elements_rows):
        blocks.append({
            "type": "actions",
            "block_id": f"prop_{prop_idx}_row_{row_i}",
            "elements": row_elements,
        })

    return blocks


def build_done_block_for_property(
    name: str,
    cmap: List[int],
    selection: Optional[int],
) -> List[dict]:
    if selection == -1:
        status = "ボツ ✗"
    elif selection is not None and selection in cmap:
        status = f"{cmap.index(selection) + 1}番を選択 ✓"
    else:
        status = "自動採用 (1番) ✓"
    return [{
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"~{name}~  →  *{status}*",
        },
    }]


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main() -> None:
    setup_logger()

    # ─── 環境変数チェック ─────────────────────────────────────
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
        sys.exit(1)

    # ─── データ読み込み ───────────────────────────────────────
    try:
        records = load_records(DATA_PATH)
    except Exception as e:
        logging.error("データ読み込みに失敗: %s", e)
        sys.exit(1)

    if not records:
        logging.info("物件データが0件。処理をスキップします。")
        sys.exit(0)

    logging.info("物件データ: %d件", len(records))

    # ─── 既存選定を読み込み ───────────────────────────────────
    existing: Dict[str, int] = load_existing_selections() if SKIP_ALREADY_SELECTED else {}
    selections: Dict[str, int] = dict(existing)

    # ─── 各物件の候補画像を収集 ───────────────────────────────
    # prop_list: [(pid, cmap, display_name, catalog_paths)]
    prop_list: List[Tuple[str, List[int], str, List[Path]]] = []

    for rec in records:
        pid  = str(rec.get("id") or "")
        name = str(rec.get("name") or pid)[:20]

        if pid in existing and SKIP_ALREADY_SELECTED:
            logging.info("[%s] 選定済み → スキップ", pid)
            continue

        paths, cmap = get_candidates_for_property(rec, max_images=MAX_CATALOG_IMAGES)

        if not paths:
            logging.warning("[%s] 候補画像なし → 自動採用(idx=0)", pid)
            selections[pid] = 0
            continue

        if len(paths) == 1:
            selections[pid] = cmap[0]
            logging.info("[%s] 候補1枚のみ → 自動選定: index=%d", pid, cmap[0])
            continue

        prop_list.append((pid, cmap, name, paths))

    if not prop_list:
        logging.info("Slack選定が必要な物件がありません。")
        save_selections(selections)
        logging.info("選定結果を保存: %s", SELECTIONS_PATH)
        sys.exit(0)

    logging.info(
        "%d件の物件をSlackで順番に選定します（タイムアウト: %ds/件）",
        len(prop_list), SELECTION_TIMEOUT,
    )

    # ─── Slack App 初期化 ─────────────────────────────────────
    handler_state: Dict = {
        "current_pid": None,
        "done": threading.Event(),
    }

    app = App(token=SLACK_BOT_TOKEN)

    # ── 候補番号ボタン（pick_{prop_idx}_{cand_i}）────────────
    @app.action(re.compile(r"^pick_\d+_\d+$"))
    def handle_pick(ack, body, client):
        ack()
        try:
            value = str(body["actions"][0].get("value", ""))
            if not value.startswith("pick:"):
                return
            rest  = value[5:]
            colon = rest.rfind(":")
            if colon < 0:
                return
            pid     = rest[:colon]
            img_idx = int(rest[colon + 1:])
            selections[pid] = img_idx

            # ログ: cmap から1-based番号を表示
            for p, cmap, _, _ in prop_list:
                if p == pid and img_idx in cmap:
                    logging.info("[%s] 選択: %d番 (img_idx=%d)", pid, cmap.index(img_idx) + 1, img_idx)
                    break
            else:
                logging.info("[%s] 選択: img_idx=%d", pid, img_idx)

            if pid == handler_state["current_pid"]:
                handler_state["done"].set()
        except Exception as e:
            logging.warning("pick ハンドラエラー: %s", e)

    # ── ボツボタン（reject_{prop_idx}）──────────────────────
    @app.action(re.compile(r"^reject_\d+$"))
    def handle_reject(ack, body, client):
        ack()
        try:
            value = str(body["actions"][0].get("value", ""))
            if not value.startswith("reject:"):
                return
            pid = value[7:]
            selections[pid] = -1
            logging.info("[%s] ボツ（物件を投稿から除外）", pid)
            if pid == handler_state["current_pid"]:
                handler_state["done"].set()
        except Exception as e:
            logging.warning("reject ハンドラエラー: %s", e)

    # ─── Socket Mode 接続 ────────────────────────────────────
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    try:
        handler.connect()
        logging.info("Slack Socket Mode に接続しました")
        time.sleep(1.0)
    except Exception as e:
        logging.error("Socket Mode の接続に失敗しました: %s", e)
        sys.exit(1)

    web_client = app.client

    # ─── 物件ごとに順番に選定 ────────────────────────────────
    for prop_idx, (pid, cmap, name, catalog_paths) in enumerate(prop_list):
        n_paths = len(catalog_paths[:MAX_CATALOG_IMAGES])
        logging.info(
            "物件 %d/%d: %s (%d枚)",
            prop_idx + 1, len(prop_list), name, n_paths,
        )

        # カタログ画像を生成
        catalog_img = create_property_catalog(catalog_paths[:MAX_CATALOG_IMAGES], name=name)
        if catalog_img is None:
            logging.warning("[%s] カタログ画像生成失敗 → 自動採用(1番)", pid)
            selections[pid] = cmap[0]
            save_selections(selections)
            continue

        img_bytes = catalog_to_bytes(catalog_img)
        logging.info(
            "[%s] カタログ: %dx%d px / %.1f KB",
            pid, catalog_img.width, catalog_img.height, len(img_bytes) / 1024,
        )

        # カタログ画像をアップロード
        try:
            web_client.files_upload_v2(
                channel=SLACK_CHANNEL,
                content=img_bytes,
                filename=f"catalog_{sanitize_filename(pid)}.jpg",
                initial_comment=(
                    f"*物件 {prop_idx + 1}/{len(prop_list)}: {name}*  "
                    f"({n_paths}枚)\n"
                    f"番号 = カタログ内の画像番号（左上バッジと一致）"
                ),
            )
        except Exception as e:
            logging.error("[%s] カタログ画像のアップロードに失敗: %s", pid, e)
            selections[pid] = cmap[0]
            save_selections(selections)
            continue

        # ボタンメッセージを送信
        handler_state["current_pid"] = pid
        handler_state["done"].clear()

        blocks = build_property_blocks(
            pid, cmap[:MAX_CATALOG_IMAGES], name, prop_idx,
            selections.get(pid),
        )
        msg_ts      = None
        msg_channel = SLACK_CHANNEL
        try:
            resp = web_client.chat_postMessage(
                channel=SLACK_CHANNEL,
                blocks=blocks,
                text=f"物件 {name}: 文字入れ画像を選択してください",
            )
            msg_ts      = resp["ts"]
            msg_channel = resp["channel"]
            logging.info("[%s] ボタンメッセージ送信 (ts=%s)", pid, msg_ts)
        except Exception as e:
            logging.error("[%s] ボタンメッセージ送信失敗: %s", pid, e)
            selections[pid] = cmap[0]
            save_selections(selections)
            continue

        # 選択を待機
        completed = handler_state["done"].wait(timeout=SELECTION_TIMEOUT)
        if not completed and pid not in selections:
            selections[pid] = cmap[0]
            logging.warning("[%s] タイムアウト → 1番を自動採用", pid)

        # 完了メッセージに更新
        if msg_ts:
            try:
                web_client.chat_update(
                    channel=msg_channel,
                    ts=msg_ts,
                    blocks=build_done_block_for_property(
                        name, cmap[:MAX_CATALOG_IMAGES], selections.get(pid),
                    ),
                    text=f"物件 {name}: 選択完了",
                )
            except Exception as e:
                logging.warning("[%s] 完了メッセージ更新失敗: %s", pid, e)

        # 毎物件後に保存（途中終了でも結果が残る）
        save_selections(selections)

    # ─── 終了処理 ────────────────────────────────────────────
    try:
        handler.close()
    except Exception:
        pass

    adopted  = sum(1 for v in selections.values() if v != -1)
    rejected = sum(1 for v in selections.values() if v == -1)
    logging.info(
        "選定完了: 採用=%d件 / ボツ=%d件 → %s",
        adopted, rejected, SELECTIONS_PATH,
    )


if __name__ == "__main__":
    main()
