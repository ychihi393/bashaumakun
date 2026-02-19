#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from google import genai
except Exception:
    genai = None


DATA_PATH = Path("assets/data.json")
SLACK_SELECTIONS_PATH = Path("assets/slack_selections.json")
OUTPUT_ROOT = Path("output/投稿用出力")
WORK_ROOT = OUTPUT_ROOT / "_work"
ADOPTED_FOLDER = OUTPUT_ROOT / "採用"
BOTS_FOLDER = OUTPUT_ROOT / "ボツ"
IMAGE_ROOT = Path("output/itanji_video/saved_images")
IMAGE_ROOT_ADOPTED = IMAGE_ROOT / "adopted"
IMAGE_ROOT_BOTS = IMAGE_ROOT / "bots"
POSTS_JSON_PATH = OUTPUT_ROOT / "投稿一覧.json"
COPY_TXT_PATH = OUTPUT_ROOT / "コピペ用_投稿文.txt"
COPY_MD_PATH = OUTPUT_ROOT / "コピペ用_投稿文.md"
CLEAN_COPY_TXT_PATH = OUTPUT_ROOT / "コピペ専用_タイトルキャプション.txt"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_COVER_PICK_MAX_IMAGES = int(os.getenv("POSTGEN_GEMINI_COVER_PICK_MAX_IMAGES", "10"))


def setup_logger() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def load_env() -> None:
    if load_dotenv is not None:
        load_dotenv()


def ensure_output_root() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    ADOPTED_FOLDER.mkdir(parents=True, exist_ok=True)
    BOTS_FOLDER.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    s = str(name or "").strip()
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or "unknown"


def sanitize_public_caption(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"\b(?:itandibb|bukkakun)\S*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("assets/data.json は配列(JSON list)である必要があります")
    return [x for x in data if isinstance(x, dict)]


def load_slack_selections() -> Dict[str, int]:
    """
    slack_selections.json を読み込む。
    - 0以上の値: 採用する画像の0-based インデックス
    - -1       : ボツ判定（main() 内で BOTS_FOLDER へ振り分け）
    """
    if not SLACK_SELECTIONS_PATH.exists():
        return {}
    try:
        raw = json.loads(SLACK_SELECTIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, int] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
    return out


def is_likely_floorplan_ref(ref: str) -> bool:
    s = str(ref or "").lower()
    return any(x in s for x in ["間取", "間取り", "madori", "floor", "layout", "plan", "図面"])


def is_likely_floorplan_image_file(path: Path) -> bool:
    if is_likely_floorplan_ref(path.name):
        return True
    try:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_ratio = float((gray > 245).mean())
        edge_ratio = float((cv2.Canny(gray, 80, 160) > 0).mean())
        return white_ratio > 0.75 and edge_ratio > 0.08
    except Exception:
        return False


def _candidate_image_paths(property_id: str, image_number_1based: int, image_ref: str = "") -> List[Path]:
    pid = sanitize_filename(property_id)
    n2 = f"{image_number_1based:02d}"
    stem = Path(str(image_ref).split("?", 1)[0]).stem if image_ref else ""
    suffix = Path(str(image_ref).split("?", 1)[0]).suffix.lower() if image_ref else ""
    exts = [".jpg", ".jpeg", ".png", ".webp"]
    roots = [IMAGE_ROOT_ADOPTED / pid, IMAGE_ROOT_BOTS / pid]
    out: List[Path] = []
    for root in roots:
        for ext in exts:
            out.append(root / f"{n2}{ext}")
            out.append(root / f"{image_number_1based}{ext}")
        if stem:
            if suffix in exts:
                out.append(root / f"{stem}{suffix}")
            for ext in exts:
                out.append(root / f"{stem}{ext}")
    uniq: List[Path] = []
    seen = set()
    for p in out:
        k = str(p)
        if k not in seen:
            uniq.append(p)
            seen.add(k)
    return uniq


def find_local_cached_image(property_id: str, image_number_1based: int, image_ref: str = "") -> Optional[Path]:
    for p in _candidate_image_paths(property_id, image_number_1based, image_ref=image_ref):
        if p.exists() and p.is_file():
            return p
    return None


def ordered_cover_candidates(images: List[Any], image_metrics: List[Dict[str, Any]], blocked: set) -> List[int]:
    valid = [i for i in range(len(images)) if i not in blocked]
    scored: List[Tuple[float, int]] = []
    used = set()
    for m in image_metrics or []:
        try:
            idx = int(m.get("index"))
            score = float(m.get("score", 0.0))
        except Exception:
            continue
        if idx in blocked or idx < 0 or idx >= len(images):
            continue
        scored.append((score, idx))
        used.add(idx)
    scored.sort(key=lambda x: x[0], reverse=True)
    ordered = [idx for _, idx in scored]
    ordered.extend([i for i in valid if i not in used])
    return ordered


def reorder_by_portrait_4x5(ordered: List[int], images: List[Any], property_id: str) -> List[int]:
    front: List[Tuple[float, int]] = []
    back: List[int] = []
    target = 4.0 / 5.0
    for idx in ordered:
        local = find_local_cached_image(property_id, idx + 1, image_ref=str(images[idx]))
        if local is None:
            back.append(idx)
            continue
        img = cv2.imdecode(np.fromfile(str(local), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            back.append(idx)
            continue
        h, w = img.shape[:2]
        if h > w:
            front.append((abs((w / float(h)) - target), idx))
        else:
            back.append(idx)
    front.sort(key=lambda x: x[0])
    return [idx for _, idx in front] + back


def _contain(im: Image.Image, w: int, h: int) -> Image.Image:
    src = im.copy()
    src.thumbnail((w, h), Image.Resampling.LANCZOS)
    bg = Image.new("RGB", (w, h), (236, 240, 247))
    bg.paste(src, ((w - src.width) // 2, (h - src.height) // 2))
    return bg


def create_candidate_catalog(paths: List[Path]) -> Optional[Image.Image]:
    if not paths:
        return None
    n = len(paths)
    cols = 3 if n > 4 else 2
    rows = int(np.ceil(n / cols))
    tile = 360
    gap = 20
    top = 40
    cw = cols * tile + (cols + 1) * gap
    ch = rows * tile + (rows + 1) * gap + top
    canvas = Image.new("RGB", (cw, ch), (245, 248, 252))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                thumb = _contain(im.convert("RGB"), tile, tile)
        except Exception:
            continue
        r, c = divmod(i, cols)
        x = gap + c * (tile + gap)
        y = top + gap + r * (tile + gap)
        canvas.paste(thumb, (x, y))
        draw.rectangle([x, y, x + 56, y + 34], fill=(33, 118, 255))
        draw.text((x + 14, y + 8), str(i + 1), fill=(255, 255, 255), font=font)
    draw.text((gap, 10), "カバー画像を番号で選択", fill=(40, 60, 80), font=font)
    return canvas


def _fit_4x5(im: Image.Image) -> Image.Image:
    target_w, target_h = 1080, 1350
    tr = target_w / float(target_h)
    w, h = im.size
    sr = w / float(h)
    if h > w:
        if sr > tr:
            nw = int(h * tr)
            x0 = max(0, (w - nw) // 2)
            crop = im.crop((x0, 0, x0 + nw, h))
        else:
            nh = int(w / tr)
            y0 = max(0, (h - nh) // 2)
            crop = im.crop((0, y0, w, y0 + nh))
        return crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
    bg = im.resize((target_w, target_h), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(radius=18))
    fg = _contain(im, target_w, target_h)
    bg.paste(fg, (0, 0))
    return bg


def _upscale(im: Image.Image) -> Image.Image:
    w, h = im.size
    return im.resize((int(w * 1.5), int(h * 1.5)), Image.Resampling.LANCZOS)


def _draw_title(im: Image.Image, title: str) -> Image.Image:
    out = im.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    draw.rectangle([0, 0, out.width, 130], fill=(0, 0, 0, 110))
    draw.text((24, 42), title[:60], font=ImageFont.load_default(), fill=(255, 255, 255, 255))
    return out


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    s = str(text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _fallback_title(record: Dict[str, Any]) -> str:
    return f"{str(record.get('layout') or '間取り要確認')}｜{str(record.get('price') or '家賃要確認')}"


def _fallback_caption(record: Dict[str, Any], property_id: str) -> str:
    station = str(record.get("station") or "").strip()
    feats = [str(x).strip() for x in (record.get("features") or []) if str(x).strip()]
    lines = "\n".join([f"- {x}" for x in feats[:4]]) if feats else "- 設備情報はお問い合わせください"
    txt = (
        "この条件、ちゃんと比べるとかなりアリです。\n"
        f"{station}\n\n設備・条件:\n{lines}\n\n"
        "図面と詳細を確認したい方は\n"
        "プロフのリンクから\n"
        f"「{property_id}」\n"
        "とだけLINEを送ってください。\n"
        "すぐに詳細をお送りします。\n\n"
        "#賃貸 #お部屋探し #一人暮らし #同棲 #物件紹介"
    )
    return sanitize_public_caption(txt)


def _gemini_copy(record: Dict[str, Any], property_id: str) -> Dict[str, str]:
    fallback = {"title": _fallback_title(record), "caption": _fallback_caption(record, property_id)}
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return fallback
    payload = {
        "id": property_id,
        "price": str(record.get("price") or ""),
        "layout": str(record.get("layout") or ""),
        "station": str(record.get("station") or ""),
        "features": [str(x) for x in (record.get("features") or [])],
        "is_new_building": bool(record.get("is_new_building")),
    }
    prompt = (
        "あなたは「SNSでバズる不動産アカウント」の専属コピーライターです。\n"
        "目的は、LINE問い合わせ（CV）につなげること。\n"
        "JSONのみ。キーは title, caption。\n"
        "- title: 漢字/数字中心、縦棒｜で区切る\n"
        "- caption: フック→説得→設備3-4点→締め→CTA→ハッシュタグ5個\n"
        "CTAは必ず:\n図面と詳細を確認したい方は\nプロフのリンクから\n"
        f"「{property_id}」\nとだけLINEを送ってください。\nすぐに詳細をお送りします。\n"
        "禁止: URL、itandibb/bukkakun、業者情報、堅苦しい口調\n"
        f"入力資料(JSON): {json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        parsed = _extract_json_block(getattr(resp, "text", "") or "")
        if not parsed:
            return fallback
        title = str(parsed.get("title") or fallback["title"]).strip()[:60]
        caption = sanitize_public_caption(str(parsed.get("caption") or fallback["caption"]).strip())
        if len(caption) < 20:
            caption = fallback["caption"]
        return {"title": title, "caption": caption}
    except Exception as e:
        logging.warning("[%s] Gemini copy fallback: %s", property_id, e.__class__.__name__)
        return fallback


def _source_folder_for_property(property_id: str) -> Optional[Path]:
    pid = sanitize_filename(property_id)
    for root in (IMAGE_ROOT_ADOPTED, IMAGE_ROOT_BOTS):
        d = root / pid
        if d.is_dir():
            return d
    return None


def _pick_source_image(record: Dict[str, Any], slack_idx: Optional[int]) -> Tuple[Optional[Path], int]:
    pid = str(record.get("id") or "")
    images = list(record.get("images") or [])
    chosen = 0
    if isinstance(slack_idx, int) and slack_idx >= 0:
        chosen = slack_idx
    if images:
        chosen = min(max(chosen, 0), len(images) - 1)
        p = find_local_cached_image(pid, chosen + 1, image_ref=str(images[chosen]))
        if p is not None:
            return p, chosen
    src_dir = _source_folder_for_property(pid)
    if src_dir:
        imgs = sorted([x for x in src_dir.iterdir() if x.is_file() and x.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        if imgs:
            return imgs[0], 0
    return None, 0


def _save_property_outputs(record: Dict[str, Any], src_image: Path, copy_payload: Dict[str, str]) -> Dict[str, Any]:
    pid = str(record.get("id") or "unknown")
    slug = sanitize_filename(pid)
    work_dir = WORK_ROOT / slug
    adopted_dir = ADOPTED_FOLDER / slug
    work_dir.mkdir(parents=True, exist_ok=True)
    adopted_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(src_image) as im:
        original = im.convert("RGB")
    upscaled = _upscale(original)
    resized = _fit_4x5(upscaled)
    titled = _draw_title(resized, copy_payload["title"])

    original.save(adopted_dir / "01_元画像.jpg", quality=95)
    upscaled.save(adopted_dir / "02_アップスケール済み.png")
    resized.save(adopted_dir / "03_4対5リサイズ済み.jpg", quality=95)
    titled.save(adopted_dir / "04_文字入れ完成.png")

    original.save(work_dir / "01_元画像.jpg", quality=95)
    upscaled.save(work_dir / "02_アップスケール済み.png")
    resized.save(work_dir / "03_4対5リサイズ済み.jpg", quality=95)
    titled.save(work_dir / "04_文字入れ完成.png")

    src_dir = _source_folder_for_property(pid)
    saved_count = 0
    if src_dir:
        imgs = sorted([x for x in src_dir.iterdir() if x.is_file() and x.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        for i, f in enumerate(imgs, start=1):
            shutil.copy2(f, adopted_dir / f"saved_{i:03d}{f.suffix.lower()}")
            saved_count += 1

    (adopted_dir / "投稿文.txt").write_text(
        f"【タイトル】\n{copy_payload['title']}\n\n【キャプション】\n{copy_payload['caption']}\n",
        encoding="utf-8-sig",
    )

    return {
        "id": pid,
        "slug": slug,
        "title": copy_payload["title"],
        "caption": copy_payload["caption"],
        "saved_images_count": saved_count,
    }


def write_copy_outputs(rows: List[Dict[str, Any]]) -> None:
    POSTS_JSON_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    txt, md, clean = [], [], []
    for i, row in enumerate(rows, start=1):
        title = str(row.get("title") or "")
        caption = str(row.get("caption") or "")
        txt += [f"{'='*60}", f"物件{i}: {row.get('id','')}", f"{'='*60}", f"タイトル: {title}", "キャプション:", caption, ""]
        md += [f"## 物件{i}: `{row.get('id','')}`", f"- タイトル: {title}", "", caption, ""]
        clean += ["━━━━━━━━━━━━━━━━━━━━━━━━", f"物件{i}: {title}", "━━━━━━━━━━━━━━━━━━━━━━━━", "【タイトル】", title, "", "【キャプション】", caption, ""]
    COPY_TXT_PATH.write_text("\n".join(txt).strip() + "\n", encoding="utf-8-sig")
    COPY_MD_PATH.write_text("\n".join(md).strip() + "\n", encoding="utf-8")
    CLEAN_COPY_TXT_PATH.write_text("\n".join(clean).strip() + "\n", encoding="utf-8-sig")


def _upload_and_send_to_slack(done_rows: List[Dict[str, Any]]) -> None:
    try:
        from drive_uploader import create_run_folder, is_configured, upload_folder_and_get_link
    except Exception:
        create_run_folder = None
        is_configured = None
        upload_folder_and_get_link = None

    drive_enabled = bool(is_configured and is_configured())
    drive_parent = create_run_folder(f"post_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}") if drive_enabled and create_run_folder else None

    slack_client = None
    slack_token = os.getenv("SLACK_BOT_TOKEN", "").strip()
    slack_channel = os.getenv("SLACK_CHANNEL", "").strip()
    if slack_token and slack_channel:
        try:
            from slack_sdk import WebClient
            slack_client = WebClient(token=slack_token)
        except Exception as e:
            logging.warning("Slack client init failed: %s", e)

    for row in done_rows:
        slug = str(row.get("slug") or "").strip()
        if not slug:
            continue
        folder = ADOPTED_FOLDER / slug
        if not folder.is_dir():
            continue

        drive_link = None
        if drive_enabled and upload_folder_and_get_link is not None:
            drive_link = upload_folder_and_get_link(folder, slug, drive_parent)

        if slack_client is not None:
            parts = []
            title = str(row.get("title") or "").strip()
            caption = sanitize_public_caption(str(row.get("caption") or "").strip())
            if title:
                parts.append(f"【タイトル】\n{title}")
            if drive_link:
                parts.append(f"【Driveリンク】\n{drive_link}")
            if caption:
                parts.append(f"【キャプション】\n{caption}")
            if parts:
                try:
                    slack_client.chat_postMessage(channel=slack_channel, text="\n\n".join(parts))
                    logging.info("[%s] Slack notification sent", slug)
                except Exception as e:
                    logging.warning("[%s] Slack notification failed: %s", slug, e)


def main() -> None:
    setup_logger()
    load_env()
    ensure_output_root()
    records = load_records(DATA_PATH)
    slack_selections = load_slack_selections()

    require_manual_slack = os.getenv("POSTGEN_REQUIRE_SLACK_SELECTION", "1").lower() in ("1", "true", "yes")
    slack_manual_mode = all(os.getenv(k, "").strip() for k in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_CHANNEL"))
    if require_manual_slack and slack_manual_mode:
        missing = []
        for rec in records:
            rid = str(rec.get("id") or "")
            if rid and list(rec.get("images") or []) and rid not in slack_selections:
                missing.append(rid)
        if missing:
            logging.error("Slack手動選定が未完了です。先に slack_selector.py を実行してください。未選定=%d件", len(missing))
            raise SystemExit(1)

    done_rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []
    rejected: List[str] = []

    for rec in records:
        rid = str(rec.get("id") or "unknown")
        slack_idx = slack_selections.get(rid)
        if slack_idx == -1:
            rejected.append(rid)
            d = BOTS_FOLDER / sanitize_filename(rid)
            d.mkdir(parents=True, exist_ok=True)
            (d / "bots_reason.txt").write_text(
                f"property_id: {rid}\nreason: selected as rejected(ボツ) in Slack\n",
                encoding="utf-8-sig",
            )
            logging.info("[%s] skipped because rejected(ボツ) was selected in Slack", rid)
            continue
        try:
            src, chosen = _pick_source_image(rec, slack_idx)
            if src is None:
                raise FileNotFoundError("ローカル画像が見つかりません")
            copy_payload = _gemini_copy(rec, rid)
            row = _save_property_outputs(rec, src, copy_payload)
            row["selected_index"] = chosen
            done_rows.append(row)
            logging.info("[%s] processed", rid)
        except Exception as e:
            failed.append({"id": rid, "error": str(e)})
            logging.exception("[%s] 処理失敗: %s", rid, e)

    if done_rows:
        write_copy_outputs(done_rows)
        _upload_and_send_to_slack(done_rows)

    for f in failed:
        d = BOTS_FOLDER / sanitize_filename(str(f.get("id") or "unknown"))
        d.mkdir(parents=True, exist_ok=True)
        (d / "失敗理由.txt").write_text(f"物件ID: {f.get('id','')}\nエラー: {f.get('error','')}\n", encoding="utf-8-sig")

    (OUTPUT_ROOT / "failed_records.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("全体完了: 成功=%s件, 失敗=%s件, ボツ=%s件", len(done_rows), len(failed), len(rejected))


if __name__ == "__main__":
    main()

