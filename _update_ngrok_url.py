"""ngrok の公開URL を取得して .env の LINE_SERVER_URL を更新するスクリプト"""
import json
import sys
import time
import urllib.request
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"

for attempt in range(10):
    try:
        data = json.loads(urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=3).read())
        tunnels = data.get("tunnels", [])
        https_tunnels = [t for t in tunnels if t["public_url"].startswith("https://")]
        if https_tunnels:
            url = https_tunnels[0]["public_url"].rstrip("/")
            break
    except Exception:
        pass
    time.sleep(1)
else:
    print("[ERROR] ngrok URL を取得できませんでした")
    sys.exit(1)

# .env の LINE_SERVER_URL を書き換え
lines = ENV_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
new_lines = []
replaced = False
for line in lines:
    if line.startswith("LINE_SERVER_URL="):
        new_lines.append(f"LINE_SERVER_URL={url}\n")
        replaced = True
    else:
        new_lines.append(line)

if not replaced:
    new_lines.append(f"LINE_SERVER_URL={url}\n")

ENV_PATH.write_text("".join(new_lines), encoding="utf-8")
print(f"[2] LINE_SERVER_URL を更新しました: {url}")
print(f"    Webhook URL: {url}/webhook")
