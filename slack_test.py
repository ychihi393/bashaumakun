#!/usr/bin/env python3
"""
Slack 接続テスト

スクレイピングなしで、Slack にテストメッセージを送って動作確認する。

使い方:
    python slack_test.py
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except Exception:
    pass

try:
    from slack_sdk import WebClient
except ImportError:
    print("ERROR: slack_sdk がインストールされていません。")
    print("実行: pip install slack_sdk")
    sys.exit(1)


def main():
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    channel = os.environ.get("SLACK_CHANNEL", "").strip()

    if not token:
        print("ERROR: SLACK_BOT_TOKEN が設定されていません。.env を確認してください。")
        sys.exit(1)
    if not channel:
        print("ERROR: SLACK_CHANNEL が設定されていません。.env を確認してください。")
        sys.exit(1)

    print("Slack にテストメッセージを送信中...")

    try:
        client = WebClient(token=token)
        resp = client.chat_postMessage(
            channel=channel,
            text="✅ **Slack接続テスト成功！**\n馬車馬くんの設定は正常です。投稿画像選定が利用できます。",
        )
        print(f"OK: メッセージを送信しました。ts={resp.get('ts', '')}")
        print("→ Slack を開いて、チャンネルにメッセージが届いているか確認してください。")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
