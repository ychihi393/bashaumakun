"""
パイプライン出力をリセットする。

削除対象:
  - assets/data.json          （スクレイピング結果）
  - output/投稿用出力/         （生成画像・テキスト）

保持するもの（テスト繰り返しに必要なため削除しない）:
  - output/itanji_video/saved_images/  （ダウンロード済み画像）
  - assets/property_numbers.json       （物件番号の連番）
  - assets/slack_selections.json       （Slack選定結果）
"""
from pathlib import Path
import shutil


def remove_file(path: Path) -> None:
    if path.exists():
        path.unlink()
        print(f"削除: {path}")
    else:
        print(f"対象なし: {path}")


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        print(f"削除: {path}/")
    else:
        print(f"対象なし: {path}/")


def main() -> None:
    # スクレイピング結果（次のスクレイピングで上書きされる）
    remove_file(Path("assets/data.json"))

    # 生成出力のみ削除（ダウンロード済み画像は保持）
    remove_dir(Path("output/投稿用出力"))

    print("clean_done")


if __name__ == "__main__":
    main()
