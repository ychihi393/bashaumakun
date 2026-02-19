import json
from pathlib import Path


def main() -> int:
    path = Path("assets/data.json")
    if not path.exists():
        print("[check] assets/data.json が見つかりません")
        return 2

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[check] assets/data.json の読み込みに失敗しました: {e}")
        return 1

    if not isinstance(data, list):
        print("[check] assets/data.json は配列形式(JSON list)ではありません")
        return 1

    count = len(data)
    print(f"adopted_count= {count}")
    return 0 if count > 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
