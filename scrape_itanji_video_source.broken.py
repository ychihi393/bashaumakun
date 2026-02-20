#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _find_cached_entrypoint() -> Path:
    cache_dir = Path(__file__).with_name("__pycache__")
    candidates = sorted(cache_dir.glob("scrape_itanji_video.cpython-*.pyc"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Cached bytecode not found under: {cache_dir}")
    return candidates[0]


def main() -> None:
    try:
        pyc_path = _find_cached_entrypoint()
    except Exception as e:
        print(f"[ERROR] Could not locate cached scraper module: {e}")
        sys.exit(1)

    runpy.run_path(str(pyc_path), run_name="__main__")


if __name__ == "__main__":
    main()
