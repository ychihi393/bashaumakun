#!/usr/bin/env python3
"""Run post-scrape test flow from existing assets/data.json.

Flow:
1) Slack selection for all current records (no skip of already selected records)
2) main.py post generation / consolidate / drive upload / slack notify

This script intentionally does not run scraping.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "assets" / "data.json"
SELECTIONS_PATH = ROOT / "assets" / "slack_selections.json"


def _print(msg: str) -> None:
    print(msg, flush=True)


def ensure_data_ready() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"[ERROR] Missing data file: {DATA_PATH}")
    try:
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"[ERROR] Failed to parse {DATA_PATH}: {e}") from e
    if not isinstance(data, list) or not data:
        raise SystemExit(f"[ERROR] {DATA_PATH} is empty or invalid list JSON.")


def ensure_slack_env() -> None:
    required = ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_CHANNEL")
    missing = [k for k in required if not os.getenv(k, "").strip()]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"[ERROR] Missing required env vars: {joined}")


def backup_and_clear_selections() -> None:
    clear = os.getenv("POST_TEST_CLEAR_SELECTIONS", "1").lower() in ("1", "true", "yes")
    if not clear or not SELECTIONS_PATH.exists():
        return
    backup = os.getenv("POST_TEST_BACKUP_SELECTIONS", "1").lower() in ("1", "true", "yes")
    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = SELECTIONS_PATH.with_name(f"slack_selections.backup_{stamp}.json")
        shutil.copy2(SELECTIONS_PATH, dst)
        _print(f"[INFO] Backed up existing selections -> {dst}")
    SELECTIONS_PATH.unlink(missing_ok=True)
    _print("[INFO] Cleared assets/slack_selections.json for full re-selection test.")


def run_step(name: str, script_name: str, env: dict[str, str]) -> None:
    _print(f"[RUN] {name}: {script_name}")
    cmd = [sys.executable, script_name]
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if result.returncode != 0:
        raise SystemExit(f"[ERROR] {script_name} failed with exit code {result.returncode}")


def main() -> None:
    os.chdir(ROOT)
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")

    ensure_data_ready()
    ensure_slack_env()
    backup_and_clear_selections()

    env = os.environ.copy()
    # Always run Slack selection for current records in this test.
    env["SLACK_SKIP_ALREADY_SELECTED"] = "0"
    # Require manual Slack selection before generation.
    env["POSTGEN_REQUIRE_SLACK_SELECTION"] = "1"

    _print("[INFO] Starting post-scrape test flow from existing assets/data.json")
    _print("[INFO] Step 1/2: slack_selector.py")
    run_step("Slack selection", "slack_selector.py", env)
    _print("[INFO] Step 2/2: main.py")
    run_step("Post generation", "main.py", env)
    _print("[OK] Test flow completed.")


if __name__ == "__main__":
    main()

