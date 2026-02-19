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
PROPERTY_NUMBERS_PATH = Path("assets/property_numbers.json")
PROPERTY_NUMBER_START = 73  # 最初の物件番号 → "073"
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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_COVER_PICK_MAX_IMAGES = int(os.getenv("POSTGEN_GEMINI_COVER_PICK_MAX_IMAGES", "10"))

# 日本語フォント候補（太字優先）
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\MPLUS1p-Bold.ttf",
    r"C:\Windows\Fonts\BIZUDGothic-Bold.ttc",
    r"C:\Windows\Fonts\BIZ-UDGothicB.ttc",
    r"C:\Windows\Fonts\meiryob.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\YuGothB.ttc",
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
]


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


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """日本語が使えるフォントをロードする。見つからなければデフォルト。"""
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_overlay(im: Image.Image, lines: List[str]) -> Image.Image:
    """
    画像下部に複数行テキストを重ねる。
    - 半透明黒帯を敷いてから白文字で描画（影付き）
    - lines: 表示したいテキストのリスト（最大3行）
    """
    out = im.copy().convert("RGBA")
    w, h = out.size

    lines = [str(l).strip() for l in lines if str(l).strip()][:3]
    if not lines:
        return out.convert("RGB")

    sizes = [62, 54, 48]
    line_gap = 14
    pad_x, pad_bottom = 36, 44

    # 各行の高さを計算して帯の高さを決める
    fonts = [_load_font(sizes[i] if i < len(sizes) else 44) for i in range(len(lines))]
    line_heights = []
    for font, text in zip(fonts, lines):
        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        bb = dummy.textbbox((0, 0), text, font=font)
        line_heights.append(bb[3] - bb[1])

    total_text_h = sum(line_heights) + line_gap * (len(lines) - 1)
    band_h = total_text_h + pad_bottom * 2

    # 半透明帯
    overlay = Image.new("RGBA", (w, band_h), (0, 0, 0, 0))
    band_draw = ImageDraw.Draw(overlay)
    band_draw.rectangle([0, 0, w, band_h], fill=(10, 10, 10, 175))
    out.paste(overlay, (0, h - band_h), overlay)

    draw = ImageDraw.Draw(out)
    y = h - band_h + pad_bottom
    for font, text, lh in zip(fonts, lines, line_heights):
        # 影
        draw.text((pad_x + 2, y + 2), text, font=font, fill=(0, 0, 0, 200))
        # 本文
        draw.text((pad_x, y), text, font=font, fill=(255, 255, 255, 255))
        y += lh + line_gap

    return out.convert("RGB")


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


# ── 物件番号管理 ─────────────────────────────────────────────────────────────

def load_property_numbers() -> Dict[str, str]:
    """assets/property_numbers.json から property_id → "073" 形式のマッピングを読み込む。"""
    if not PROPERTY_NUMBERS_PATH.exists():
        return {}
    try:
        return json.loads(PROPERTY_NUMBERS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_property_numbers(mapping: Dict[str, str]) -> None:
    PROPERTY_NUMBERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROPERTY_NUMBERS_PATH.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def assign_property_number(property_id: str, mapping: Dict[str, str]) -> str:
    """
    property_id に物件番号を割り当てる。
    - 既に番号があればそれを返す（再実行しても変わらない）
    - 新規なら既存の最大番号+1 を割り当て、mapping を更新する
    - 番号は3桁ゼロ埋め文字列（例: "073"）
    """
    if property_id in mapping:
        return mapping[property_id]
    if mapping:
        max_num = max((int(v) for v in mapping.values() if v.isdigit()), default=PROPERTY_NUMBER_START - 1)
        next_num = max_num + 1
    else:
        next_num = PROPERTY_NUMBER_START
    num_str = f"{next_num:03d}"
    mapping[property_id] = num_str
    return num_str


# ── キャプション生成 ──────────────────────────────────────────────────────────

def _fallback_title(record: Dict[str, Any]) -> str:
    layout = str(record.get("layout") or "")
    price = str(record.get("price") or "")
    parts = [p for p in [layout, price] if p]
    return "｜".join(parts) if parts else "物件情報"


def _fallback_caption(record: Dict[str, Any], prop_num: str) -> str:
    feats = [str(x).strip() for x in (record.get("features") or []) if str(x).strip()]
    feat_lines = "\n".join([f"- {x}" for x in feats[:4]]) if feats else "- 設備情報はお問い合わせください"
    txt = (
        "この条件、ちゃんと比べるとかなりアリです。\n\n"
        f"設備・条件:\n{feat_lines}\n\n"
        "詳細が気になった方は\n"
        "プロフのリンクから\n"
        f"「{prop_num}」\n"
        "とだけLINEを送ってください。\n"
        "すぐに詳細をお送りします。\n\n"
        "#賃貸 #お部屋探し #一人暮らし #同棲 #物件紹介"
    )
    return sanitize_public_caption(txt)


def _gemini_copy(record: Dict[str, Any], property_id: str, prop_num: str) -> Dict[str, str]:
    fallback = {
        "title": _fallback_title(record),
        "caption": _fallback_caption(record, prop_num),
        "overlay_line1": _fallback_title(record),
        "overlay_line2": "",
    }
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return fallback

    payload = {
        "prop_num": prop_num,
        "price": str(record.get("price") or ""),
        "layout": str(record.get("layout") or ""),
        "area_hint": str(record.get("station") or ""),  # エリア参考（出力には駅名を書かせない）
        "features": [str(x) for x in (record.get("features") or [])],
        "is_new_building": bool(record.get("is_new_building")),
    }
    prompt = (
        "あなたは「SNSでバズる不動産アカウント」の専属コピーライターです。\n"
        "目的: 読者に「詳細が気になる」と感じさせ、LINE問い合わせ（CV）につなげる。\n\n"
        "JSONのみ出力。キーは title, caption, overlay_line1, overlay_line2。\n\n"
        "【title】\n"
        "- 漢字・数字中心、縦棒｜で区切る（例: 2LDK｜築浅｜南向き）\n"
        "- 物件名・号室・駅名は書かない。エリア（区・市など）はOK\n"
        "- 60文字以内\n\n"
        "【caption】\n"
        "- 構成: フック→物件の魅力（設備・条件3〜4点）→含みを持たせた締め→CTA→ハッシュタグ5個\n"
        "- 物件名・号室・最寄り駅名は書かない。区・エリア・間取り・価格帯はOK\n"
        "- 読者に「どこだろう？詳細が知りたい」と思わせる含みのある表現にする\n"
        "- CTAは以下の文言で固定:\n"
        "  詳細が気になった方は\n"
        "  プロフのリンクから\n"
        f"  「{prop_num}」\n"
        "  とだけLINEを送ってください。\n"
        "  すぐに詳細をお送りします。\n\n"
        "【overlay_line1】画像に重ねる1行目テキスト（例: 2LDK｜◯万円台）20文字以内\n"
        "【overlay_line2】画像に重ねる2行目テキスト（例: 駅名なし・エリアや設備の特徴）20文字以内\n\n"
        "【禁止】URL、itandibb、bukkakun、業者情報、堅苦しい口調、物件名、号室、駅名\n\n"
        f"入力資料(JSON):\n{json.dumps(payload, ensure_ascii=False)}"
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
        overlay1 = str(parsed.get("overlay_line1") or fallback["overlay_line1"]).strip()[:24]
        overlay2 = str(parsed.get("overlay_line2") or "").strip()[:24]
        return {"title": title, "caption": caption, "overlay_line1": overlay1, "overlay_line2": overlay2}
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

    # 文字入れ: overlay_line1 / overlay_line2 を使う（なければ title を使う）
    overlay_lines = [
        copy_payload.get("overlay_line1") or copy_payload.get("title") or "",
        copy_payload.get("overlay_line2") or "",
    ]
    titled = _draw_overlay(resized, overlay_lines)

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
    # ── Google Drive 初期化 ──────────────────────────────────────────────────
    try:
        from drive_uploader import create_run_folder, is_configured, upload_folder_and_get_link
        drive_enabled = bool(is_configured and is_configured())
    except Exception as e:
        logging.warning("drive_uploader のインポートに失敗: %s", e)
        drive_enabled = False
        create_run_folder = None
        upload_folder_and_get_link = None

    if drive_enabled:
        logging.info("[Drive] 有効: Google Driveアップロードを開始します")
        drive_parent = create_run_folder(f"post_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}") if create_run_folder else None
        if drive_parent:
            logging.info("[Drive] 実行フォルダ作成: folder_id=%s", drive_parent)
        else:
            logging.warning("[Drive] 実行フォルダの作成に失敗しました。物件ごとに直接アップロードします")
    else:
        drive_parent = None
        logging.warning("[Drive] 無効: GOOGLE_DRIVE_CREDENTIALS_JSON が未設定または認証ファイルが見つかりません")

    # ── Slack クライアント初期化 ─────────────────────────────────────────────
    slack_client = None
    slack_token = os.getenv("SLACK_BOT_TOKEN", "").strip()
    slack_channel = os.getenv("SLACK_CHANNEL", "").strip()
    if slack_token and slack_channel:
        try:
            from slack_sdk import WebClient
            slack_client = WebClient(token=slack_token)
            logging.info("[Slack] 通知クライアント初期化OK: channel=%s", slack_channel)
        except Exception as e:
            logging.warning("[Slack] クライアント初期化に失敗: %s", e)
    else:
        logging.warning("[Slack] 通知スキップ: SLACK_BOT_TOKEN または SLACK_CHANNEL が未設定")

    # ── 物件ごとにアップロード & 通知 ────────────────────────────────────────
    for row in done_rows:
        slug = str(row.get("slug") or "").strip()
        prop_num = str(row.get("property_number") or "")
        if not slug:
            continue
        folder = ADOPTED_FOLDER / slug
        if not folder.is_dir():
            logging.warning("[%s] 採用フォルダが見つかりません: %s", slug, folder)
            continue

        # Drive アップロード
        drive_link = None
        if drive_enabled and upload_folder_and_get_link is not None:
            logging.info("[%s] Driveアップロード中...", slug)
            drive_link = upload_folder_and_get_link(folder, f"{prop_num}_{slug}" if prop_num else slug, drive_parent)
            if drive_link:
                logging.info("[%s] Driveアップロード完了: %s", slug, drive_link)
            else:
                logging.warning("[%s] Driveアップロード失敗（リンクなし）", slug)

        # Slack 通知
        if slack_client is not None:
            title = str(row.get("title") or "").strip()
            caption = sanitize_public_caption(str(row.get("caption") or "").strip())
            num_label = f"物件番号: {prop_num}" if prop_num else ""
            parts = []
            if num_label:
                parts.append(num_label)
            if title:
                parts.append(f"【タイトル】\n{title}")
            if drive_link:
                parts.append(f"【Driveリンク】\n{drive_link}")
            if caption:
                parts.append(f"【キャプション】\n{caption}")
            if parts:
                try:
                    slack_client.chat_postMessage(channel=slack_channel, text="\n\n".join(parts))
                    logging.info("[%s] Slack通知送信完了", slug)
                except Exception as e:
                    logging.warning("[%s] Slack通知送信失敗: %s", slug, e)


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

    # 物件番号マッピングを読み込む（複数回実行しても番号が変わらない）
    prop_numbers = load_property_numbers()

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
            logging.info("[%s] ボツ判定 → ボツフォルダへ", rid)
            continue

        # ボツ以外のみ物件番号を割り当てる
        prop_num = assign_property_number(rid, prop_numbers)
        logging.info("[%s] 物件番号: %s", rid, prop_num)

        try:
            src, chosen = _pick_source_image(rec, slack_idx)
            if src is None:
                raise FileNotFoundError("ローカル画像が見つかりません")
            copy_payload = _gemini_copy(rec, rid, prop_num)
            row = _save_property_outputs(rec, src, copy_payload)
            row["selected_index"] = chosen
            row["property_number"] = prop_num
            done_rows.append(row)
            logging.info("[%s] 処理完了 (物件番号=%s)", rid, prop_num)
        except Exception as e:
            failed.append({"id": rid, "error": str(e)})
            logging.exception("[%s] 処理失敗: %s", rid, e)

    # 物件番号を保存（次回実行時に引き継がれる）
    save_property_numbers(prop_numbers)
    logging.info("物件番号マッピングを保存: %s", PROPERTY_NUMBERS_PATH)

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

