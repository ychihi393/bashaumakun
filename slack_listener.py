#!/usr/bin/env python3
"""
Slack 投稿完了ボタン リスナー

main.py が送った「✅ 投稿完了」ボタンを常時監視し、
ボタンが押されたら:
  1. ボタンメッセージを「完了済み」に更新
  2. assets/line_properties.json に posted_at を記録

使い方:
    python slack_listener.py  (または run_slack_listener.bat)

※ main.py の実行とは別プロセスで常時起動しておく
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler
except ImportError:
    print("ERROR: slack_bolt がインストールされていません")
    print("実行: pip install slack_bolt slack_sdk")
    sys.exit(1)

SLACK_BOT_TOKEN      = os.getenv("SLACK_BOT_TOKEN", "").strip()
SLACK_APP_TOKEN      = os.getenv("SLACK_APP_TOKEN", "").strip()
LINE_PROPERTIES_PATH = Path("assets/line_properties.json")


def load_properties() -> dict:
    if LINE_PROPERTIES_PATH.exists():
        try:
            return json.loads(LINE_PROPERTIES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_properties(props: dict) -> None:
    LINE_PROPERTIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    LINE_PROPERTIES_PATH.write_text(
        json.dumps(props, ensure_ascii=False, indent=2), encoding="utf-8"
    )


app = App(token=SLACK_BOT_TOKEN)


@app.action(re.compile(r"^mark_posted_\w+$"))
def handle_posted(ack, body, client, action):
    ack()
    value  = str(action.get("value", ""))
    parts  = value.split(":")          # "posted:076:slug"
    prop_num = parts[1] if len(parts) > 1 else "?"
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M")

    # line_properties.json に投稿日時を記録
    props = load_properties()
    if prop_num in props:
        props[prop_num]["posted_at"] = datetime.now().isoformat()
        save_properties(props)
        print(f"[INFO] 物件{prop_num} 投稿完了を記録しました")

    # ボタンメッセージを「完了済み」表示に更新
    try:
        channel = body["container"]["channel_id"]
        msg_ts  = body["container"]["message_ts"]
        client.chat_update(
            channel=channel,
            ts=msg_ts,
            text=f"✅ 物件{prop_num} — 投稿完了済み",
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"✅ *物件{prop_num}* — 投稿完了！\n"
                            f"LINEに `{prop_num}` と送ると物件情報が届きます\n"
                            f"_完了時刻: {now_str}_"
                        ),
                    },
                }
            ],
        )
        print(f"[INFO] 物件{prop_num} ボタンメッセージを完了済みに更新しました")
    except Exception as e:
        print(f"[WARN] メッセージ更新失敗: {e}")


if __name__ == "__main__":
    if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
        print("ERROR: SLACK_BOT_TOKEN / SLACK_APP_TOKEN が .env に未設定です")
        sys.exit(1)

    print("[INFO] Slack 投稿完了リスナー起動 — ✅ ボタンを待機中...")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
