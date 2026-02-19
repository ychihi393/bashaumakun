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
    remove_file(Path("assets/data.json"))
    remove_dir(Path("output"))
    print("clean_done")


if __name__ == "__main__":
    main()
