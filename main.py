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

import math

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

# 日本語フォント候補（手書き風優先 → 太字ゴシックにフォールバック）
# ★おすすめ無料フォント: 「851チカラづよく」をダウンロードしてインストールすると
#   Instagram映えするキャッチーな手書き風になります。
#   https://pm85.com/ で「851チカラづよく」を検索してダウンロード後、
#   C:\Windows\Fonts\ にインストール（ファイルを右クリック→インストール）
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\851CHIKARA-DZUYOKU_kanaA_004.ttf",  # 851チカラづよく（要インストール・最推奨）
    r"C:\Windows\Fonts\HGRGE.TTC",           # HGS行書E（筆書き風・最もキャッチー）
    r"C:\Windows\Fonts\HGRSMP.TTF",          # HGP正楷書体（手書き・楷書）
    r"C:\Windows\Fonts\HGRSKP.TTF",          # HGP教科書体（手書き風）
    r"C:\Windows\Fonts\UDDigiKyokashoN-B.ttc",  # UD教科書体Bold
    r"C:\Windows\Fonts\HGRPRE.TTC",
    r"C:\Windows\Fonts\BIZ-UDGothicB.ttc",
    r"C:\Windows\Fonts\meiryob.ttc",
    r"C:\Windows\Fonts\YuGothB.ttc",
    r"C:\Windows\Fonts\NotoSansJP-VF.ttf",
    r"C:\Windows\Fonts\msgothic.ttc",
]
_cached_font_path: Optional[str] = None


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


def create_property_catalog(paths: List[Path], name: str = "") -> Optional[Image.Image]:
    """
    1物件の候補画像を横長カタログ画像に並べる（最大15枚、5列）。
    番号バッジ・番号ラベルはSlackボタンの番号と完全に一致する。
    """
    if not paths:
        return None

    COLS    = 5
    THUMB_W = 190
    THUMB_H = 238
    NUM_H   = 32
    GAP     = 10
    TOP_PAD = 48
    BADGE_COLOR = (33, 118, 255)

    n    = min(len(paths), 15)
    rows = math.ceil(n / COLS)
    canvas_w = COLS * THUMB_W + (COLS + 1) * GAP
    canvas_h = TOP_PAD + rows * (THUMB_H + NUM_H + GAP) + GAP

    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 248, 252))
    draw   = ImageDraw.Draw(canvas)
    fhdr   = _load_font(20)
    fbadge = _load_font(20)
    fnum   = _load_font(18)

    label = "文字入れ画像を選んでください" + (f"  ({name})" if name else "")
    draw.text((GAP, 12), label, fill=(40, 60, 80), font=fhdr)

    for i, path in enumerate(paths[:15]):
        row, col = divmod(i, COLS)
        x = GAP + col * (THUMB_W + GAP)
        y = TOP_PAD + row * (THUMB_H + NUM_H + GAP)

        try:
            with Image.open(path) as im:
                thumb = _contain(im.convert("RGB"), THUMB_W, THUMB_H)
        except Exception:
            thumb = Image.new("RGB", (THUMB_W, THUMB_H), (200, 200, 210))
        canvas.paste(thumb, (x, y))

        # 番号バッジ（左上）
        nstr = str(i + 1)
        bw = 32 if len(nstr) == 1 else 46
        bh = 26
        draw.rectangle([x, y, x + bw, y + bh], fill=BADGE_COLOR)
        draw.text((x + bw // 2, y + bh // 2), nstr, fill=(255, 255, 255), font=fbadge, anchor="mm")
        # 番号（サムネイル下）
        draw.text((x + THUMB_W // 2, y + THUMB_H + 4), nstr, fill=BADGE_COLOR, font=fnum, anchor="mt")

    return canvas


# 後方互換エイリアス（旧コードからの参照用）
create_candidate_catalog = create_property_catalog


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
    # 横長画像: SD アウトペインティングを試みる（USE_SD_OUTPAINTING=1 の場合）
    try:
        from sd_outpainter import outpaint_to_4x5
        sd_result = outpaint_to_4x5(im)
        if sd_result is not None:
            return sd_result
    except Exception as _sd_err:
        logging.info("[SD] スキップ: %s", _sd_err)

    # フォールバック: ぼかし背景で上下を埋める
    bg = im.resize((target_w, target_h), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(radius=18))
    fg = _contain(im, target_w, target_h)
    bg.paste(fg, (0, 0))
    return bg


def _upscale(im: Image.Image) -> Image.Image:
    w, h = im.size
    return im.resize((int(w * 1.5), int(h * 1.5)), Image.Resampling.LANCZOS)


def _load_font(size: int) -> ImageFont.ImageFont:
    """日本語が使えるフォントをロードする。キャッシュして高速化。"""
    global _cached_font_path
    # キャッシュ済みフォントを使う
    if _cached_font_path:
        try:
            return ImageFont.truetype(_cached_font_path, size)
        except Exception:
            _cached_font_path = None  # キャッシュが無効なら再探索

    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                _cached_font_path = path
                logging.debug("フォント: %s", path)
                return font
            except Exception:
                continue

    # フォールバック: Windows Fonts ディレクトリをスキャン
    fonts_dir = Path(r"C:\Windows\Fonts")
    if fonts_dir.exists():
        for name in ["BIZ-UDGothicB.ttc", "meiryob.ttc", "YuGothB.ttc", "msgothic.ttc"]:
            p = fonts_dir / name
            if p.exists():
                try:
                    font = ImageFont.truetype(str(p), size)
                    _cached_font_path = str(p)
                    return font
                except Exception:
                    continue
        # 任意の .ttc を試す
        for p in list(fonts_dir.glob("*.ttc"))[:20]:
            try:
                font = ImageFont.truetype(str(p), size)
                _cached_font_path = str(p)
                return font
            except Exception:
                continue

    logging.warning("日本語フォントが見つかりません。デフォルトフォントを使用します")
    return ImageFont.load_default()


def _draw_overlay(im: Image.Image, overlay_data: Dict[str, str]) -> Image.Image:
    """
    画像中央にテキストオーバーレイを描画する。

    overlay_data:
      tag    - 行1 (朱色)       例: "新着"
      main   - 行2 (淡い黄緑)   例: "池袋まで30分"
      attr   - 行3前半 (淡い黄) 例: "新築 1LDK"
      detail - 行3後半 (白)     例: "8.9万"

    エフェクト: ぼかし黒グロー（太い黒縁をガウスぼかし）+ 細い白外フチ
    テキスト折り返し: 1行最大9文字、超えたら改行
    """
    out = im.copy().convert("RGBA")
    w, h = out.size
    draw = ImageDraw.Draw(out)
    cx = w // 2

    # フォントサイズ（縦幅基準・やや控えめ）
    size_tag  = max(12, int(h * 0.045))
    size_main = max(14, int(h * 0.058))
    size_attr = max(14, int(h * 0.058))
    font_tag  = _load_font(size_tag)
    font_main = _load_font(size_main)
    font_attr = _load_font(size_attr)

    scale  = max(0.5, h / 1350.0)
    sw_b   = max(5, int(9 * scale))    # グロー用ストローク幅（太め）
    sw_w   = max(1, int(2 * scale))    # 白外フチ幅
    blur_r = max(4, int(sw_b * 1.3))   # ぼかし半径

    tag_text    = str(overlay_data.get("tag")    or "").strip()
    main_text   = str(overlay_data.get("main")   or "").strip()
    attr_text   = str(overlay_data.get("attr")   or "").strip()
    detail_text = str(overlay_data.get("detail") or "").strip()

    if not any([tag_text, main_text, attr_text, detail_text]):
        logging.warning("_draw_overlay: テキストが空のためオーバーレイをスキップ")
        return out.convert("RGB")

    logging.info("文字入れ: tag=%r / main=%r / attr=%r / detail=%r",
                 tag_text, main_text, attr_text, detail_text)

    _dm = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    def _lh(font, text: str = "Ag") -> int:
        """1行分のピクセル高さを返す"""
        try:
            bb = _dm.textbbox((0, 0), text, font=font, anchor="lt")
            return max(1, bb[3] - bb[1])
        except Exception:
            return font.size

    section_gap = max(6, int(scale * 10))  # セクション間ギャップ

    # rows: list of (text, font, color, gap_before, is_split, attr_str, det_str)
    # 折り返しなし: 各フィールドを1行で表示
    rows: List[tuple] = []

    if tag_text:
        rows.append((tag_text, font_tag, (220, 60, 30), 0, False, "", ""))

    if main_text:
        gap = section_gap if rows else 0
        rows.append((main_text, font_main, (155, 210, 70), gap, False, "", ""))

    if attr_text and detail_text:
        gap = section_gap if rows else 0
        combined = attr_text + "  " + detail_text
        rows.append((combined, font_attr, (235, 215, 60), gap, True, attr_text, detail_text))
    elif attr_text:
        gap = section_gap if rows else 0
        rows.append((attr_text, font_attr, (235, 215, 60), gap, False, "", ""))
    elif detail_text:
        gap = section_gap if rows else 0
        rows.append((detail_text, font_attr, (255, 255, 255), gap, False, "", ""))

    if not rows:
        return out.convert("RGB")

    # 総高さを計算してブロック全体を縦中央に配置
    total_h = sum(_lh(row[1], row[0]) + sw_b + row[3] for row in rows)
    block_top = int(h * 0.46) - total_h // 2
    cur_y = block_top

    def _glow(x: int, y: int, text: str, font, fill: tuple, anchor: str = "mm") -> None:
        """ぼかし黒グロー + 白外フチ + 本体テキストを描画"""
        if not text:
            return
        # 1. ぼかし黒グロー（太いストロークをガウスぼかし）
        glow_img = Image.new("RGBA", out.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow_img)
        gd.text((x, y), text, font=font, fill=(0, 0, 0, 210), anchor=anchor,
                stroke_width=sw_b, stroke_fill=(0, 0, 0, 210))
        out.alpha_composite(glow_img.filter(ImageFilter.GaussianBlur(radius=blur_r)))
        # 2. 白外フチ + 本体テキスト
        draw.text((x, y), text, font=font, fill=fill, anchor=anchor,
                  stroke_width=sw_w, stroke_fill=(255, 255, 255))

    for (text, font, color, gap, is_split, attr, det) in rows:
        lh = _lh(font, text)
        cy = cur_y + gap + (lh + sw_b) // 2
        cur_y += gap + lh + sw_b

        if is_split:
            # attr（淡い黄）と detail（白）を同行に並べて2色で描画
            try:
                aw  = _dm.textlength(attr, font=font)
                spw = _dm.textlength("  ", font=font)
                dw  = _dm.textlength(det,  font=font)
                x0  = cx - int((aw + spw + dw) / 2)
            except Exception:
                x0  = cx - len(text) * size_attr // 2
                aw  = float(size_attr * len(attr))
                spw = float(size_attr)
            # glow はテキスト全体で1回
            glow_img = Image.new("RGBA", out.size, (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow_img)
            gd.text((cx, cy), text, font=font, fill=(0, 0, 0, 210), anchor="mm",
                    stroke_width=sw_b, stroke_fill=(0, 0, 0, 210))
            out.alpha_composite(glow_img.filter(ImageFilter.GaussianBlur(radius=blur_r)))
            draw.text((x0,                 cy), attr, font=font, fill=(235, 215, 60),
                      anchor="lm", stroke_width=sw_w, stroke_fill=(255, 255, 255))
            draw.text((x0 + int(aw + spw), cy), det,  font=font, fill=(255, 255, 255),
                      anchor="lm", stroke_width=sw_w, stroke_fill=(255, 255, 255))
        else:
            _glow(cx, cy, text, font, color)

    return out.convert("RGB")


def create_all_properties_catalog(
    prop_data: List[Tuple[str, List[Optional[Path]]]],
    max_candidates: int = 4,
) -> Optional[Image.Image]:
    """
    全物件の候補画像を1枚の横長カタログ画像に生成する。

    レイアウト:
      各行 = 1物件（左端に物件番号、右に候補画像を横並び）
      候補画像の番号はSlackボタンの番号と完全一致（左→右が1,2,3,4）

    prop_data: [(物件名, [候補パス1, 候補パス2, ...]), ...]
    max_candidates: 1物件あたりの最大候補数（Slackボタン制限に合わせ最大4）
    """
    if not prop_data:
        return None

    LABEL_W = 68    # 左の物件番号エリア幅
    THUMB_W = 210   # 各候補サムネイル幅
    THUMB_H = 265   # 各候補サムネイル高さ（縦型）
    NUM_H   = 34    # サムネイル下の番号表示エリア
    GAP     = 10    # 各要素の間隔
    TOP_PAD = 48    # ヘッダー用上部余白
    BG_COLOR    = (245, 248, 252)
    BADGE_COLOR = (33, 118, 255)
    LABEL_COLOR = (60, 80, 100)

    n_props = len(prop_data)
    # 実際の最大候補数を算出（max_candidates以下に制限）
    actual_max = min(max((len(ps) for _, ps in prop_data), default=1), max_candidates)

    canvas_w = LABEL_W + actual_max * (THUMB_W + GAP) + GAP
    row_h    = THUMB_H + NUM_H + GAP
    canvas_h = TOP_PAD + n_props * (row_h + GAP) + GAP

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    font_header    = _load_font(20)
    font_prop_num  = _load_font(30)   # 左端の物件番号
    font_badge     = _load_font(22)   # 候補番号バッジ
    font_num_below = _load_font(20)   # 候補番号（サムネイル下）

    draw.text(
        (GAP, 12),
        "文字入れする画像の番号を各物件ごとに押してください",
        fill=LABEL_COLOR,
        font=font_header,
    )

    for prop_i, (name, paths) in enumerate(prop_data):
        row_y = TOP_PAD + prop_i * (row_h + GAP)

        # 左端: 物件番号
        prop_num_str = str(prop_i + 1)
        draw.text(
            (LABEL_W // 2, row_y + THUMB_H // 2),
            prop_num_str,
            fill=LABEL_COLOR,
            font=font_prop_num,
            anchor="mm",
        )
        # 仕切り線（物件ごとに）
        if prop_i > 0:
            draw.line(
                [(0, row_y - GAP // 2), (canvas_w, row_y - GAP // 2)],
                fill=(210, 215, 225),
                width=1,
            )

        for cand_i, path in enumerate(paths[:max_candidates]):
            x = LABEL_W + cand_i * (THUMB_W + GAP)
            y = row_y

            # サムネイル
            if path and path.exists():
                try:
                    with Image.open(path) as im:
                        thumb = _contain(im.convert("RGB"), THUMB_W, THUMB_H)
                except Exception:
                    thumb = Image.new("RGB", (THUMB_W, THUMB_H), (200, 200, 210))
            else:
                thumb = Image.new("RGB", (THUMB_W, THUMB_H), (200, 200, 210))

            canvas.paste(thumb, (x, y))

            # 候補番号バッジ（左上）
            cand_num_str = str(cand_i + 1)
            badge_w = 36 if len(cand_num_str) == 1 else 50
            badge_h = 28
            draw.rectangle([x, y, x + badge_w, y + badge_h], fill=BADGE_COLOR)
            draw.text(
                (x + badge_w // 2, y + badge_h // 2),
                cand_num_str,
                fill=(255, 255, 255),
                font=font_badge,
                anchor="mm",
            )

            # 候補番号（サムネイル下中央）
            draw.text(
                (x + THUMB_W // 2, y + THUMB_H + 4),
                cand_num_str,
                fill=BADGE_COLOR,
                font=font_num_below,
                anchor="mt",
            )

    return canvas


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
    fallback_title = _fallback_title(record)
    fallback = {
        "title":   fallback_title,
        "caption": _fallback_caption(record, prop_num),
        "tag":     "",
        "main":    fallback_title,
        "attr":    "",
        "detail":  "",
    }
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return fallback

    # 対象主要駅リスト（.env の TARGET_STATIONS で上書き可）
    target_stations_raw = os.getenv("TARGET_STATIONS", "渋谷,新宿,池袋,品川,東京,銀座,上野,秋葉原").strip()
    target_stations_str = "・".join(s.strip() for s in target_stations_raw.split(",") if s.strip())

    nearest_station = str(record.get("station") or "")

    payload = {
        "prop_num": prop_num,
        "price": str(record.get("price") or ""),
        "layout": str(record.get("layout") or ""),
        "nearest_station_info": nearest_station,
        "features": [str(x) for x in (record.get("features") or [])],
        "is_new_building": bool(record.get("is_new_building")),
    }
    prompt = (
        "あなたは「SNSでバズる不動産アカウント」の専属コピーライターです。\n"
        "目的: 読者に「詳細が気になる」と感じさせ、LINE問い合わせ（CV）につなげる。\n\n"
        "JSONのみ出力。キーは title, caption, tag, main, attr, detail。\n\n"
        "【title】\n"
        "- 漢字・数字中心、縦棒｜で区切る（例: 2LDK｜築浅｜南向き）\n"
        "- 物件名・号室・駅名は書かない。エリア（区・市など）はOK\n"
        "- 60文字以内\n\n"
        "【caption】\n"
        "- 構成: フック→物件の魅力（設備・条件3〜4点）→含みを持たせた締め→CTA→ハッシュタグ5個\n"
        "- 物件名・号室・最寄り駅名は書かない。区・エリア・間取り・価格帯はOK\n"
        "- 読者に「どこだろう？詳細が知りたい」と思わせる含みのある表現にする\n"
        "- スマートフォン表示を前提に読みやすく書く:\n"
        "  ・1〜2文ごとに改行する（\\n を使う）\n"
        "  ・段落間は空行（\\n\\n）で区切る\n"
        "  ・行頭や見出し代わりに絵文字を使う（✨🏠💰🚉📍など）\n"
        "  ・マークダウン記法（**や##）は使わない — プレーンテキストのみ\n"
        "  ・箇条書きは「・」を使う\n"
        "- CTAは以下の文言で固定（改行を維持）:\n"
        "  詳細が気になった方は\n"
        "  プロフのリンクから\n"
        f"  「{prop_num}」\n"
        "  とだけLINEを送ってください。\n"
        "  すぐに詳細をお送りします。\n\n"
        "【tag】画像オーバーレイ 行1: 物件の状態・特徴を短く（例: \"新着\", \"限定1室\", \"値下げ\"）10文字以内\n"
        "【main】画像オーバーレイ 行2: 主要駅へのアクセス時間\n"
        f"  最寄り駅情報: {nearest_station}\n"
        f"  対象主要駅: {target_stations_str}\n"
        "  最寄り路線から上記対象駅のうち最もアクセスしやすい1駅を選び、\n"
        "  実際の所要時間と直通かどうかを調べて記載。\n"
        "  形式: \"〇〇まで〇分\" または \"〇〇まで〇分(直通)\"\n"
        "  駅名を書いてよい。15文字以内\n"
        "【attr】画像オーバーレイ 行3前半: 間取り・建物タイプ（例: \"新築 1LDK\"）10文字以内\n"
        "【detail】画像オーバーレイ 行3後半: 価格（例: \"8.9万円台\"）10文字以内\n\n"
        "【禁止】URL、itandibb、bukkakun、業者情報、堅苦しい口調、物件名、号室\n\n"
        f"入力資料(JSON):\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        client = genai.Client(api_key=api_key)
        # Google検索グラウンディングで最新の路線情報を参照
        try:
            from google.genai import types as _genai_types
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=_genai_types.GenerateContentConfig(
                    tools=[_genai_types.Tool(google_search=_genai_types.GoogleSearch())]
                ),
            )
        except Exception:
            # 検索グラウンディング非対応の場合はフォールバック
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        parsed = _extract_json_block(getattr(resp, "text", "") or "")
        if not parsed:
            return fallback
        title   = str(parsed.get("title")   or fallback["title"]).strip()[:60]
        caption = sanitize_public_caption(str(parsed.get("caption") or fallback["caption"]).strip())
        if len(caption) < 20:
            caption = fallback["caption"]
        tag     = str(parsed.get("tag")    or "").strip()[:12]
        main    = str(parsed.get("main")   or fallback["main"]).strip()[:24]
        attr    = str(parsed.get("attr")   or "").strip()[:12]
        detail  = str(parsed.get("detail") or "").strip()[:12]
        return {"title": title, "caption": caption, "tag": tag, "main": main, "attr": attr, "detail": detail}
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

    # 文字入れ: tag/main/attr/detail を使う（なければ title を main にフォールバック）
    overlay_data = {
        "tag":    copy_payload.get("tag")    or "",
        "main":   copy_payload.get("main")   or copy_payload.get("title") or "",
        "attr":   copy_payload.get("attr")   or "",
        "detail": copy_payload.get("detail") or "",
    }
    titled = _draw_overlay(resized, overlay_data)

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


LINE_PROPERTIES_PATH = Path("assets/line_properties.json")


def _save_line_properties(done_rows: List[Dict[str, Any]]) -> None:
    """LINE Bot が参照する物件データを assets/line_properties.json に保存"""
    existing: Dict[str, Any] = {}
    if LINE_PROPERTIES_PATH.exists():
        try:
            existing = json.loads(LINE_PROPERTIES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    for row in done_rows:
        prop_num = str(row.get("property_number") or "")
        if not prop_num:
            continue
        existing[prop_num] = {
            "property_number": prop_num,
            "slug":       str(row.get("slug")       or ""),
            "title":      str(row.get("title")      or ""),
            "caption":    str(row.get("caption")    or ""),
            "price":      str(row.get("price")      or ""),
            "layout":     str(row.get("layout")     or ""),
            "station":    str(row.get("station")    or ""),
            "features":   list(row.get("features")  or []),
            "detail_url": str(row.get("detail_url") or ""),
            "posted_at":  existing.get(prop_num, {}).get("posted_at"),
        }
    LINE_PROPERTIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    LINE_PROPERTIES_PATH.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("[LINE] 物件データ保存: %s (%d件)", LINE_PROPERTIES_PATH, len(done_rows))

    # Google Apps Script に物件データを送信（GAS が参照するため）
    gas_url = os.getenv("GAS_WEBHOOK_URL", "").strip()
    gas_secret = os.getenv("GAS_UPDATE_SECRET", "").strip()
    if gas_url and gas_secret:
        try:
            import urllib.request as _urllib_req
            payload = json.dumps({
                "type": "update_properties",
                "secret": gas_secret,
                "data": existing,
            }, ensure_ascii=False).encode("utf-8")
            req = _urllib_req.Request(
                gas_url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with _urllib_req.urlopen(req, timeout=10) as resp:
                logging.info("[LINE] GAS へ物件データ送信完了: %s", resp.read().decode())
        except Exception as e:
            logging.warning("[LINE] GAS 送信失敗（スキップ）: %s", e)
    else:
        logging.info("[LINE] GAS_WEBHOOK_URL 未設定のため GAS 送信をスキップ")


def _upload_and_send_to_slack(done_rows: List[Dict[str, Any]]) -> None:
    slack_token   = os.getenv("SLACK_BOT_TOKEN", "").strip()
    slack_channel = os.getenv("SLACK_CHANNEL",   "").strip()
    if not slack_token or not slack_channel:
        logging.warning("[Slack] 通知スキップ: SLACK_BOT_TOKEN または SLACK_CHANNEL が未設定")
        return

    try:
        from slack_sdk import WebClient
        client = WebClient(token=slack_token)
        logging.info("[Slack] クライアント初期化OK: channel=%s", slack_channel)
    except Exception as e:
        logging.warning("[Slack] クライアント初期化失敗: %s", e)
        return

    # ── 物件ごとに送信 ─────────────────────────────────────────────────────
    for i, row in enumerate(done_rows, 1):
        slug     = str(row.get("slug")            or "")
        prop_num = str(row.get("property_number") or str(i))
        title    = str(row.get("title")           or "")
        caption  = sanitize_public_caption(str(row.get("caption") or ""))

        adopted_dir = ADOPTED_FOLDER / slug
        if not adopted_dir.is_dir():
            logging.warning("[%s] 採用フォルダが見つかりません", slug)
            continue

        # 画像収集（文字入れ完成 + 全ソース画像）
        file_uploads = []
        overlay = adopted_dir / "04_文字入れ完成.png"
        if overlay.exists():
            file_uploads.append({
                "file": str(overlay), "filename": f"00_cover_{prop_num}.png",
                "title": f"【文字入れ】物件{prop_num}",
            })
        for img in sorted(p for p in adopted_dir.glob("saved_*")
                          if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}):
            file_uploads.append({"file": str(img), "filename": img.name, "title": img.stem})

        if not file_uploads:
            logging.warning("[%s] 送信する画像がありません", slug)
            continue

        # ── メッセージ①: 物件番号 + タイトル（コピペ用）────────────────
        prop_id_msg = f"物件{prop_num}　{title}" if title else f"物件{prop_num}"
        try:
            client.chat_postMessage(channel=slack_channel, text=prop_id_msg, mrkdwn=False)
        except Exception as e:
            logging.warning("[%s] 物件番号メッセージ送信失敗: %s", slug, e)

        # ── メッセージ②: タイトルのみ（コピペ用）────────────────────────
        if title:
            try:
                client.chat_postMessage(channel=slack_channel, text=title, mrkdwn=False)
            except Exception as e:
                logging.warning("[%s] タイトル送信失敗: %s", slug, e)

        # ── メッセージ③: キャプションのみ（コピペ用）────────────────────
        if caption:
            try:
                client.chat_postMessage(channel=slack_channel, text=caption, mrkdwn=False)
            except Exception as e:
                logging.warning("[%s] キャプション送信失敗: %s", slug, e)

        # ── メッセージ④: 投稿完了ボタン（スレッドのアンカー）────────────
        thread_ts = None
        try:
            resp = client.chat_postMessage(
                channel=slack_channel,
                text=f"物件{prop_num} — 画像{len(file_uploads)}枚をスレッドに送信済み",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f"📷 *物件{prop_num}* — 全{len(file_uploads)}枚\n"
                                f"画像はこのメッセージのスレッドにあります\n"
                                f"Instagramに投稿したら下のボタンを押してください"
                            ),
                        },
                    },
                    {
                        "type": "actions",
                        "block_id": f"posted_{prop_num}",
                        "elements": [{
                            "type": "button",
                            "text": {"type": "plain_text", "text": "✅ 投稿完了"},
                            "style": "primary",
                            "value": f"posted:{prop_num}:{slug}",
                            "action_id": f"mark_posted_{prop_num}",
                        }],
                    },
                ],
            )
            thread_ts = resp.get("ts")
            logging.info("[%s] ボタンメッセージ送信完了 (ts=%s)", slug, thread_ts)
        except Exception as e:
            logging.warning("[%s] ボタンメッセージ送信失敗: %s", slug, e)

        # ── スレッドに全画像を送信（10枚ずつ）────────────────────────────
        BATCH = 10
        batches = [file_uploads[j:j + BATCH] for j in range(0, len(file_uploads), BATCH)]
        for b_i, batch in enumerate(batches):
            try:
                kwargs: Dict[str, Any] = {"channel": slack_channel, "file_uploads": batch}
                if thread_ts:
                    kwargs["thread_ts"] = thread_ts
                client.files_upload_v2(**kwargs)
                logging.info("[%s] 画像バッチ %d/%d (%d枚) アップロード完了",
                             slug, b_i + 1, len(batches), len(batch))
            except Exception as e:
                logging.warning("[%s] 画像バッチ %d アップロード失敗: %s", slug, b_i + 1, e)


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
            # LINE Bot 用に物件データを追加
            row["price"]      = str(rec.get("price")    or "")
            row["layout"]     = str(rec.get("layout")   or "")
            row["station"]    = str(rec.get("station")  or "")
            row["features"]   = list(rec.get("features") or [])
            row["detail_url"] = str(rec.get("detail_url") or "")
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
        _save_line_properties(done_rows)
        _upload_and_send_to_slack(done_rows)

    for f in failed:
        d = BOTS_FOLDER / sanitize_filename(str(f.get("id") or "unknown"))
        d.mkdir(parents=True, exist_ok=True)
        (d / "失敗理由.txt").write_text(f"物件ID: {f.get('id','')}\nエラー: {f.get('error','')}\n", encoding="utf-8-sig")

    (OUTPUT_ROOT / "failed_records.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("全体完了: 成功=%s件, 失敗=%s件, ボツ=%s件", len(done_rows), len(failed), len(rejected))


if __name__ == "__main__":
    main()

