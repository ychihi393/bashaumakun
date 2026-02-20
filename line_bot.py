#!/usr/bin/env python3
"""
LINE Bot サーバー

機能:
  1. POST /webhook  — LINE Messaging API ウェブフック
       ユーザーが物件番号（例: "076" "76" "物件076"）を送ると
       Flex Message カルーセルで物件情報を返信する
  2. GET  /property/{prop_num} — 物件詳細ページ（LINE 内ブラウザ / LIFF）
       全画像 + 物件情報をモバイル向け HTML で表示
  3. GET  /images/{prop_num}/{filename} — 物件画像サーブ

必要な .env 設定:
  LINE_CHANNEL_ACCESS_TOKEN  xoxb- … チャンネルアクセストークン（長期）
  LINE_CHANNEL_SECRET        チャンネルシークレット
  LINE_SERVER_URL            このサーバーの公開 URL（例: https://xxx.ngrok.io）
  LINE_BOT_PORT              ポート番号（デフォルト: 8000）

起動方法:
  python line_bot.py
  ※ 外部公開が必要 → ngrok または Cloudflare Tunnel を併用
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
from dotenv import load_dotenv
load_dotenv()

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import FileResponse, HTMLResponse
except ImportError:
    print("ERROR: fastapi / uvicorn がインストールされていません")
    print("実行: pip install fastapi uvicorn")
    sys.exit(1)

try:
    from linebot.v3 import WebhookHandler
    from linebot.v3.exceptions import InvalidSignatureError
    from linebot.v3.messaging import (
        ApiClient,
        Configuration,
        FlexMessage,
        MessagingApi,
        PushMessageRequest,
        TextMessage,
    )
    from linebot.v3.messaging.models import FlexContainer
    from linebot.v3.webhooks import MessageEvent, TextMessageContent
    LINE_AVAILABLE = True
except ImportError:
    LINE_AVAILABLE = False
    print("[WARN] linebot がインストールされていません。pip install linebot")

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
CHANNEL_SECRET       = os.getenv("LINE_CHANNEL_SECRET", "").strip()
SERVER_URL           = os.getenv("LINE_SERVER_URL", "http://localhost:8000").rstrip("/")
PORT                 = int(os.getenv("LINE_BOT_PORT", "8000"))
LSTEP_WEBHOOK_URL    = os.getenv("LSTEP_WEBHOOK_URL", "https://rcv.linestep.net/v3/call/2008591924")

LINE_PROPERTIES_PATH = Path("assets/line_properties.json")
ADOPTED_FOLDER       = Path("output/投稿用出力/採用")

# ──────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────
app = FastAPI(title="LINE Bot Server")


def load_properties() -> Dict[str, Any]:
    if LINE_PROPERTIES_PATH.exists():
        try:
            return json.loads(LINE_PROPERTIES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def find_property(prop_num_raw: str) -> Optional[Dict[str, Any]]:
    """3桁ゼロ埋め / 生番号 両方で検索"""
    props = load_properties()
    padded = f"{int(prop_num_raw):03d}" if prop_num_raw.isdigit() else prop_num_raw
    return props.get(padded) or props.get(prop_num_raw)


# ──────────────────────────────────────────────
# LSTEP プロキシ転送
# ──────────────────────────────────────────────
async def _forward_to_lstep(body: bytes, signature: str) -> None:
    """受信した Webhook をそのまま LSTEP に転送する（fire-and-forget）"""
    if not LSTEP_WEBHOOK_URL:
        return
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(
                LSTEP_WEBHOOK_URL,
                data=body,
                headers={
                    "Content-Type":    "application/json",
                    "X-Line-Signature": signature,
                },
                timeout=aiohttp.ClientTimeout(total=5),
            )
    except Exception as e:
        print(f"[WARN] LSTEP 転送失敗: {e}")


# ──────────────────────────────────────────────
# LINE Webhook
# ──────────────────────────────────────────────
if LINE_AVAILABLE and CHANNEL_SECRET:
    _handler = WebhookHandler(CHANNEL_SECRET)
    _config  = Configuration(access_token=CHANNEL_ACCESS_TOKEN)

    @app.post("/webhook")
    async def webhook(request: Request):
        sig  = request.headers.get("X-Line-Signature", "")
        body = await request.body()

        # ① LSTEP に全メッセージを転送（LSTEP は今まで通り動く）
        asyncio.create_task(_forward_to_lstep(body, sig))

        # ② 署名検証 → 物件番号判定
        try:
            _handler.handle(body.decode("utf-8"), sig)
        except InvalidSignatureError:
            raise HTTPException(status_code=400, detail="Invalid signature")
        return Response(content="OK")

    @_handler.add(MessageEvent, message=TextMessageContent)
    def handle_message(event: MessageEvent):
        text = (event.message.text or "").strip()
        m = re.match(r"^(?:物件)?(\d{2,4})$", text)
        if not m:
            return  # 物件番号以外は LSTEP に任せる
        prop_num_raw = m.group(1)
        prop_num     = f"{int(prop_num_raw):03d}"
        prop         = find_property(prop_num_raw)
        if not prop:
            return  # 該当物件なければ何もしない（LSTEP のデフォルト応答に任せる）

        # Reply API は使わず Push API で送る（LSTEP が Reply Token を使えるように）
        user_id = event.source.user_id
        flex    = _build_flex_message(prop_num, prop)
        with ApiClient(_config) as api_client:
            line_api = MessagingApi(api_client)
            line_api.push_message(PushMessageRequest(
                to=user_id,
                messages=[flex],
            ))

else:
    @app.post("/webhook")
    async def webhook_stub():
        return {"status": "LINE not configured"}


def _build_flex_message(prop_num: str, prop: Dict[str, Any]) -> FlexMessage:
    detail_url    = f"{SERVER_URL}/property/{prop_num}"
    cover_url     = f"{SERVER_URL}/images/{prop_num}/cover"
    price         = prop.get("price", "---")
    layout        = prop.get("layout", "---")
    station       = prop.get("station", "---")
    title         = prop.get("title", f"物件{prop_num}")[:40]
    features      = prop.get("features", [])
    features_text = "　".join(str(f) for f in features[:3])

    bubble: Dict[str, Any] = {
        "type": "bubble",
        "hero": {
            "type": "image",
            "url": cover_url,
            "size": "full",
            "aspectRatio": "4:3",
            "aspectMode": "cover",
            "action": {"type": "uri", "uri": detail_url},
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {"type": "text", "text": f"📍 物件{prop_num}",
                 "weight": "bold", "size": "md", "color": "#dc3c1e"},
                {"type": "text", "text": title, "weight": "bold",
                 "size": "lg", "wrap": True, "margin": "sm"},
                {
                    "type": "box", "layout": "vertical",
                    "margin": "md", "spacing": "xs",
                    "contents": [
                        {"type": "box", "layout": "baseline", "spacing": "sm",
                         "contents": [
                             {"type": "text", "text": "💰 家賃", "size": "sm",
                              "color": "#888888", "flex": 2},
                             {"type": "text", "text": price, "size": "sm", "flex": 3},
                         ]},
                        {"type": "box", "layout": "baseline", "spacing": "sm",
                         "contents": [
                             {"type": "text", "text": "🏠 間取", "size": "sm",
                              "color": "#888888", "flex": 2},
                             {"type": "text", "text": layout, "size": "sm", "flex": 3},
                         ]},
                        {"type": "box", "layout": "baseline", "spacing": "sm",
                         "contents": [
                             {"type": "text", "text": "🚉 駅", "size": "sm",
                              "color": "#888888", "flex": 2},
                             {"type": "text", "text": station[:20], "size": "sm",
                              "flex": 3, "wrap": True},
                         ]},
                    ],
                },
                *([{"type": "text", "text": features_text, "size": "xs",
                    "color": "#888888", "margin": "md", "wrap": True}]
                  if features_text else []),
            ],
        },
        "footer": {
            "type": "box", "layout": "vertical",
            "contents": [{
                "type": "button", "style": "primary", "color": "#dc3c1e",
                "action": {"type": "uri", "label": "📷 写真と詳細を見る", "uri": detail_url},
            }],
        },
    }

    return FlexMessage(
        alt_text=f"物件{prop_num}の詳細はこちら",
        contents=FlexContainer.from_dict({"type": "carousel", "contents": [bubble]}),
    )


# ──────────────────────────────────────────────
# 物件詳細ページ
# ──────────────────────────────────────────────
@app.get("/property/{prop_num}", response_class=HTMLResponse)
async def property_detail(prop_num: str):
    prop = find_property(prop_num)
    if not prop:
        raise HTTPException(status_code=404, detail=f"物件{prop_num}が見つかりません")
    padded    = f"{int(prop_num):03d}" if prop_num.isdigit() else prop_num
    slug      = prop.get("slug", "")
    adopted   = ADOPTED_FOLDER / slug
    image_urls = []
    if adopted.is_dir():
        for img in sorted(p for p in adopted.iterdir()
                          if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}):
            image_urls.append(f"{SERVER_URL}/images/{padded}/{img.name}")
    return HTMLResponse(content=_render_html(padded, prop, image_urls))


@app.get("/images/{prop_num}/{filename}")
async def serve_image(prop_num: str, filename: str):
    prop = find_property(prop_num)
    if not prop:
        raise HTTPException(status_code=404)
    slug    = prop.get("slug", "")
    adopted = ADOPTED_FOLDER / slug
    if filename == "cover":
        for name in ["04_文字入れ完成.png",
                     *sorted(p.name for p in adopted.glob("saved_*") if p.is_file())]:
            p = adopted / name
            if p.exists():
                return FileResponse(str(p))
        raise HTTPException(status_code=404)
    p = adopted / filename
    if not p.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(p))


def _render_html(prop_num: str, prop: Dict[str, Any], image_urls: list) -> str:
    title       = prop.get("title", f"物件{prop_num}")
    price       = prop.get("price", "---")
    layout      = prop.get("layout", "---")
    station     = prop.get("station", "---")
    features    = prop.get("features", [])
    caption     = prop.get("caption", "")
    detail_url  = prop.get("detail_url", "")

    caption_html  = caption.replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br>")
    features_html = "".join(
        f'<span style="background:#f0f0f0;border-radius:12px;padding:4px 10px;'
        f'font-size:13px;margin:3px;display:inline-block">{f}</span>'
        for f in features
    )
    images_html = "\n".join(
        f'<img src="{u}" loading="lazy" '
        f'style="width:100%;border-radius:8px;margin-bottom:10px;display:block">'
        for u in image_urls
    )
    inquiry_btn = (
        f'<a href="{detail_url}" style="display:block;text-align:center;padding:14px;'
        f'background:#06c755;color:white;border-radius:10px;font-weight:bold;'
        f'text-decoration:none;margin-top:16px;font-size:16px">🏠 詳細ページを開く</a>'
    ) if detail_url else ""

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0">
  <title>物件{prop_num}</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,'Hiragino Kaku Gothic ProN',sans-serif;
          background:#f5f5f5;color:#333;padding-bottom:40px}}
    .header{{background:#dc3c1e;color:white;padding:16px;
              text-align:center;font-weight:bold;font-size:18px}}
    .card{{background:white;border-radius:12px;margin:12px;
           padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.08)}}
    .label{{font-size:12px;color:#888;margin-bottom:2px}}
    .value{{font-size:16px;font-weight:bold;margin-bottom:12px}}
    .grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}}
    .images{{margin:12px}}
    .caption{{white-space:pre-wrap;line-height:1.8;font-size:14px;color:#444}}
  </style>
</head>
<body>
  <div class="header">📍 物件{prop_num}</div>
  <div class="card">
    <div class="label">タイトル</div>
    <div style="font-size:14px;font-weight:bold;margin-bottom:14px">{title}</div>
    <div class="grid">
      <div><div class="label">💰 家賃</div><div class="value">{price}</div></div>
      <div><div class="label">🏠 間取り</div><div class="value">{layout}</div></div>
    </div>
    <div class="label">🚉 最寄り駅</div>
    <div class="value" style="font-size:14px">{station}</div>
    <div style="margin-top:6px">{features_html}</div>
    {inquiry_btn}
  </div>
  <div class="images">{images_html}</div>
  <div class="card">
    <div class="label" style="margin-bottom:10px">📝 キャプション</div>
    <div class="caption">{caption_html}</div>
  </div>
</body>
</html>"""


if __name__ == "__main__":
    print(f"[INFO] LINE Bot サーバー起動: http://0.0.0.0:{PORT}")
    print(f"[INFO] Webhook URL: {SERVER_URL}/webhook")
    print(f"[INFO] LINE_CHANNEL_SECRET: {'設定済み' if CHANNEL_SECRET else '未設定'}")
    print(f"[INFO] LINE_CHANNEL_ACCESS_TOKEN: {'設定済み' if CHANNEL_ACCESS_TOKEN else '未設定'}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
