import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import Page, sync_playwright

import scrape_itanji as base

try:
    from google import genai
except Exception:
    genai = None

TOP_URL = "https://bukkakun.com/"
LIST_SEARCH_URL = "https://itandibb.com/rent_rooms/list"
OUTPUT_DIR = Path("output/itanji_video")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_PATH = Path("assets/data.json")
SEEN_URLS_PATH = OUTPUT_DIR / "seen_detail_urls.json"
SAVED_IMAGES_DIR = OUTPUT_DIR / "saved_images"

TARGET_WARDS = ["北区", "板橋区", "練馬区", "足立区", "葛飾区", "江戸川区"]
TARGET_CITIES_RAW = [
    "立川市", "三鷹市", "武蔵野市", "府中市", "町田市", "調布市", "小金井市", "小平市",
    "東村山市", "国分寺市", "国立市", "福生市", "東大和市", "日野市", "東久留米市", "西東京市",
]
AREA_FALLBACK_MAP = {}

# 埼玉県の対象市
SAITAMA_CITIES = [
    "戸田市", "川口市", "さいたま市", "蕨市", "所沢市", "ふじみ野市", "和光市", "草加市",
]

# 千葉県の対象市
CHIBA_CITIES = [
    "船橋市", "習志野市", "千葉市", "市川市", "浦安市", "流山市", "松戸市",
]
TARGET_FEATURES = ["バストイレ別", "温水洗浄便座", "独立洗面台"]

MAIN_STATIONS = ["新宿駅", "渋谷駅", "池袋駅", "東京駅", "品川駅", "大手町駅", "飯田橋駅"]

MAX_PAGES = int(os.getenv("ITANJI_VIDEO_MAX_PAGES", "80"))
MAX_PROPERTIES = int(os.getenv("ITANJI_VIDEO_MAX_PROPERTIES", "120"))
HEADLESS = os.getenv("ITANJI_VIDEO_HEADLESS", "0").lower() in ("1", "true", "yes", "on")
HTTP_TIMEOUT = int(os.getenv("ITANJI_VIDEO_HTTP_TIMEOUT", "12"))
MIN_IMAGES = int(os.getenv("ITANJI_VIDEO_MIN_IMAGES", "7"))
SAVE_REVIEW_IMAGES = os.getenv("ITANJI_VIDEO_SAVE_IMAGES", "1").lower() in ("1", "true", "yes", "on")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"


@dataclass
class RouteCheck:
    destination: str
    minutes: Optional[int]
    transfers: Optional[int]
    ok: bool
    source: str


@dataclass
class ImageQuality:
    index: int
    score: float
    brightness: float
    white_ratio: float
    floor_open_ratio: float
    room_likeness: float = 0.5  # 上部エッジ多め＝部屋らしさ
    close_up_penalty: float = 0.0  # 接写・単体物っぽいときのペナルティ


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def normalize_area_names() -> List[str]:
    names: List[str] = []
    for city in TARGET_CITIES_RAW:
        names.append(city)
        for alt in AREA_FALLBACK_MAP.get(city, []):
            if alt not in names:
                names.append(alt)
    for ward in TARGET_WARDS:
        if ward not in names:
            names.append(ward)
    return names


def safe_click(page: Page, selectors: List[str], step_name: str) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                loc.first.click(timeout=3000)
                return True
        except Exception:
            continue
    log(f"[WARN] click failed: {step_name}")
    return False


def check_by_text(page: Page, text: str) -> bool:
    selectors = [
        f'label:has-text("{text}")',
        f'button:has-text("{text}")',
        f'div:has-text("{text}")',
        f'span:has-text("{text}")',
        f'p:has-text("{text}")',
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() == 0:
                continue
            loc.first.click(timeout=2500)
            page.wait_for_timeout(250)
            return True
        except Exception:
            continue
    return False


def close_area_modal_if_open(page: Page) -> None:
    # Some UI flows keep an area drawer/modal open after "決定".
    # Close it so subsequent controls are clickable.
    close_selectors = [
        'button:has(.itandi-bb-ui__CloseButton__Icon)',
        'button:has(svg.itandi-bb-ui__CloseButton__Icon)',
        'button[aria-label="閉じる"]',
        'button[aria-label="close"]',
    ]
    for sel in close_selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                loc.first.click(timeout=2000)
                page.wait_for_timeout(400)
                log("[OK] 所在地モーダルを閉じました")
                return
        except Exception:
            continue
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
    except Exception:
        pass


def click_search_submit(page: Page) -> bool:
    selectors = [
        'button.ListSearchButton[type="submit"]',
        'button:has-text("検索")[type="submit"]',
        'button:has-text("検索")',
        'input[type="submit"][value*="検索"]',
    ]
    if safe_click(page, selectors, "検索実行"):
        return True

    # JS fallback for cases where overlay blocks normal click handling.
    try:
        ok = page.evaluate(
            """
            () => {
              const nodes = Array.from(document.querySelectorAll('button, input[type="submit"]'));
              const target = nodes.find((el) => {
                const txt = (el.innerText || el.textContent || el.value || "").trim();
                return txt.includes("検索");
              });
              if (!target) return false;
              target.click();
              return true;
            }
            """
        )
        if ok:
            log("[OK] 検索実行 (js fallback)")
            return True
    except Exception:
        pass
    log("[WARN] click failed: 検索実行 (all fallbacks)")
    return False


def apply_search_filters(page: Page) -> bool:
    log("検索条件を適用します")
    page.goto(LIST_SEARCH_URL, wait_until="load", timeout=60000)
    page.wait_for_timeout(1200)

    if not safe_click(page, ['button:has-text("所在地で絞り込み")', 'button:has(div:has-text("所在地で絞り込み"))'], "所在地で絞り込み"):
        return False
    page.wait_for_timeout(700)

    def _select_prefecture_areas(pref: str, areas: List[str]) -> None:
        """都道府県を選択してから市区町村にチェックを入れる"""
        check_by_text(page, pref)
        page.wait_for_timeout(500)
        ok = ng = 0
        for area in areas:
            if check_by_text(page, area):
                ok += 1
            else:
                ng += 1
                log(f"[WARN] area not found in UI ({pref}): {area}")
        log(f"[INFO] {pref} area select: ok={ok}, not_found={ng}")
        page.wait_for_timeout(300)

    # 東京都: 区部 + 市部
    _select_prefecture_areas("東京都", normalize_area_names())

    # 埼玉県
    _select_prefecture_areas("埼玉県", SAITAMA_CITIES)

    # 千葉県
    _select_prefecture_areas("千葉県", CHIBA_CITIES)

    if not safe_click(page, ['button:has-text("決定")', 'button:has(div:has-text("決定"))'], "所在地の決定"):
        check_by_text(page, "決定")
    page.wait_for_timeout(900)
    close_area_modal_if_open(page)

    if not check_by_text(page, "可能"):
        log("[WARN] 広告掲載可能(可能)の選択に失敗")

    try:
        page.select_option('select[name="rent:lteq"]', value="10")
        log("[OK] rent:lteq=10")
    except Exception:
        try:
            page.locator('input[name="rent:lteq"]').first.fill("10")
            log("[OK] rent:lteq=10 (input fill)")
        except Exception:
            log("[WARN] rent:lteq=10 の設定に失敗")

    try:
        page.select_option('select[name="offer_conditions_updated_at:gteq"]', value="3")
        log("[OK] offer_conditions_updated_at:gteq=3")
    except Exception:
        log("[WARN] offer_conditions_updated_at:gteq の設定に失敗")

    try:
        page.select_option('select[name="building_age:lteq"]', value="15")
        log("[OK] building_age:lteq=15")
    except Exception:
        try:
            page.locator('input[name="building_age:lteq"]').first.fill("15")
            log("[OK] building_age:lteq=15 (input fill)")
        except Exception:
            log("[WARN] building_age:lteq=15 の設定に失敗")

    if not click_search_submit(page):
        return False

    page.wait_for_timeout(2200)
    log("[OK] 検索条件の適用が完了")
    return True


def extract_cards_from_page(page: Page) -> List[Dict[str, Any]]:
    payload = page.evaluate(
        """
        () => {
          const cardSelectors = [
            "div.itandi-bb-ui__Box",
            "div[class*='itandi-bb-ui__Box']",
            "[data-testid='property-card']",
            ".property-card",
            "a[href*='/rent_rooms/']"
          ];
          let cards = [];
          for (const sel of cardSelectors) {
            const found = Array.from(document.querySelectorAll(sel));
            if (found.length > 0) { cards = found; break; }
          }
          const textOf = (el) => ((el?.innerText || el?.textContent || "").trim());
          return cards.map((card) => {
            let href = "";
            try {
              if (card.matches && card.matches("a[href*='/rent_rooms/']")) href = card.getAttribute("href") || "";
            } catch {}
            if (!href) {
              const link = card.querySelector("a[href*='/rent_rooms/']");
              href = link ? (link.getAttribute("href") || "") : "";
            }
            if (!href) {
              const linkAny = card.querySelector("a[href]");
              href = linkAny ? (linkAny.getAttribute("href") || "") : "";
            }
            if (!href) {
              href = card.getAttribute("data-href") || card.getAttribute("href") || "";
            }
            const cardText = textOf(card);
            const statusTexts = Array.from(
              card.querySelectorAll("div.itandi-bb-ui__Flex, p, span")
            ).map((el) => textOf(el)).filter(Boolean);
            let imageCount = null;
            const direct = cardText.match(/(\\d+)\\s*枚/);
            if (direct) imageCount = Number.parseInt(direct[1], 10);
            if (imageCount === null) {
              for (const t of statusTexts) {
                const m = t.match(/(\\d+)\\s*枚/);
                if (m) { imageCount = Number.parseInt(m[1], 10); break; }
              }
            }
            return { href, cardText, statusTexts, imageCount };
          });
        }
        """
    )
    return payload or []


def to_absolute_url(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    return href if href.startswith("http") else f"https://itandibb.com{href}"


def load_seen_urls() -> Set[str]:
    if not SEEN_URLS_PATH.exists():
        return set()
    try:
        with SEEN_URLS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {str(x).strip() for x in data if str(x).strip()}
    except Exception:
        pass
    return set()


def save_seen_urls(urls: Set[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SEEN_URLS_PATH.open("w", encoding="utf-8") as f:
        json.dump(sorted(urls), f, ensure_ascii=False, indent=2)


def image_ext_from_url(url: str) -> str:
    clean = (url or "").split("?", 1)[0].lower()
    ext = Path(clean).suffix
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        return ext
    return ".jpg"


def save_record_images(records: List[Dict[str, Any]], label: str) -> int:
    if not SAVE_REVIEW_IMAGES:
        return 0
    saved = 0
    label_dir = SAVED_IMAGES_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        rid = str(rec.get("id") or "").strip()
        if not rid:
            continue
        rec_dir = label_dir / rid
        rec_dir.mkdir(parents=True, exist_ok=True)
        for idx, img_url in enumerate(rec.get("images") or [], start=1):
            if not str(img_url).startswith("http"):
                continue
            ext = image_ext_from_url(str(img_url))
            dst = rec_dir / f"{idx:02d}{ext}"
            if dst.exists() and dst.stat().st_size > 0:
                continue
            raw = fetch_image_bytes(str(img_url))
            if not raw:
                continue
            dst.write_bytes(raw)
            saved += 1
    return saved


def collect_filtered_urls(page: Page) -> List[str]:
    seen = set()
    filtered_urls: List[str] = []
    page_no = 1
    stats = {
        "cards_total": 0,
        "no_url": 0,
        "dup": 0,
        "status_reject": 0,
        "image_reject": 0,
        "kept": 0,
    }
    ng_status_re = re.compile(r"(申し込みあり|申込あり|先行申込|申込受付終了|申込済|成約済)")
    positive_status_re = re.compile(r"(募集中|空室|募集)")

    while page_no <= MAX_PAGES:
        cards = extract_cards_from_page(page)
        log(f"ページ{page_no}: cards={len(cards)}")
        stats["cards_total"] += len(cards)

        for card in cards:
            url = to_absolute_url(card.get("href") or "")
            if not url:
                stats["no_url"] += 1
                continue
            if "/rent_rooms/" not in url:
                stats["no_url"] += 1
                continue
            if url in seen:
                stats["dup"] += 1
                continue
            seen.add(url)

            all_text = "\n".join([(card.get("cardText") or "")] + (card.get("statusTexts") or []))
            # Reject only explicit negative statuses; do not reject generic "申込" wording.
            if ng_status_re.search(all_text):
                stats["status_reject"] += 1
                continue
            if not positive_status_re.search(all_text) and "募集終了" in all_text:
                stats["status_reject"] += 1
                continue

            image_count = card.get("imageCount")
            if image_count is not None:
                try:
                    if int(image_count) < MIN_IMAGES:
                        stats["image_reject"] += 1
                        continue
                except Exception:
                    pass

            filtered_urls.append(url)
            stats["kept"] += 1
            if 0 < MAX_PROPERTIES <= len(filtered_urls):
                log(
                    f"[INFO] filter stats: total={stats['cards_total']}, no_url={stats['no_url']}, dup={stats['dup']}, "
                    f"status_reject={stats['status_reject']}, image_reject={stats['image_reject']}, kept={stats['kept']}"
                )
                return filtered_urls

        try:
            next_btn = page.locator('button:has-text("次へ"), a:has-text("次へ"), [data-testid="next-page"]').first
            if next_btn.count() == 0 or next_btn.is_disabled():
                break
            next_btn.click(timeout=4000)
            page.wait_for_timeout(1200)
            page_no += 1
        except Exception:
            break

    log(
        f"[INFO] filter stats: total={stats['cards_total']}, no_url={stats['no_url']}, dup={stats['dup']}, "
        f"status_reject={stats['status_reject']}, image_reject={stats['image_reject']}, kept={stats['kept']}, min_images={MIN_IMAGES}"
    )
    return filtered_urls


def format_price(rent_value: Any) -> str:
    text = str(rent_value or "").strip()
    if not text:
        return ""

    m_man = re.search(r"([0-9]+(?:\\.[0-9]+)?)\\s*万", text)
    if m_man:
        num = float(m_man.group(1))
        return f"{int(num)}万円" if abs(num - int(num)) < 1e-9 else f"{num:.1f}万円"

    m_yen = re.search(r"([0-9][0-9,]*)\\s*円", text)
    if not m_yen:
        m_yen = re.search(r"([0-9][0-9,]*)", text)
    if not m_yen:
        return text

    yen = int(m_yen.group(1).replace(",", ""))
    man = yen / 10000.0
    return f"{int(man)}万円" if yen % 10000 == 0 else f"{man:.1f}万円"


def is_new_building(detail: Dict[str, Any]) -> bool:
    built = str(detail.get("built_date") or "")
    if "新築" in built:
        return True
    if "築0年" in built:
        return True
    year_match = re.search(r"(20\\d{2}|19\\d{2})年", built)
    if year_match:
        y = int(year_match.group(1))
        return y >= datetime.now().year
    return False


def fetch_image_bytes(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": UA})
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def evaluate_image_quality(img_bgr: np.ndarray, index: int) -> ImageQuality:
    """
    洋室・リビングなど床面積が広く見える画像を高スコアに。
    エアコン・シューズボックス・収納のみの接写は低スコア（room_likeness / close_up_penalty で調整）。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = hsv.shape[:2]
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    brightness = float(np.mean(v) / 255.0)
    white_ratio = float(np.mean((s < 45) & (v > 170)))

    bottom = hsv[int(h * 0.45) :, :, :]
    floor_open_ratio = float(np.mean((bottom[:, :, 1] < 65) & (bottom[:, :, 2] > 95))) if bottom.size else 0.0

    # 部屋らしさ: 上部（壁・天井）にエッジが多く、下部（床）は少ない＝広い部屋の遠景
    edges = cv2.Canny(gray, 50, 150)
    top_half = edges[: int(h * 0.5), :]
    bottom_half = edges[int(h * 0.5) :, :]
    edge_top = float(np.sum(top_half > 0)) / max(1, top_half.size)
    edge_bottom = float(np.sum(bottom_half > 0)) / max(1, bottom_half.size)
    if edge_bottom > 1e-6:
        room_likeness = min(1.0, edge_top / edge_bottom)
    else:
        room_likeness = 0.7 if edge_top > 0.01 else 0.5

    # 接写・単体物ペナルティ: 中央がほぼ白一色で全体の明度分散が小さい（エアコン・シューズボックス等）
    center_y1, center_y2 = int(h * 0.33), int(h * 0.66)
    center_x1, center_x2 = int(w * 0.33), int(w * 0.66)
    center_region = v[center_y1:center_y2, center_x1:center_x2]
    center_white = float(np.mean((s[center_y1:center_y2, center_x1:center_x2] < 50) & (center_region > 180)))
    v_std = float(np.std(v))

    # エアコン接写追加チェック:
    #   - 上部40%が白く均一 (エアコン本体が上に据付)
    #   - 全体のエッジ密度が低い (複雑な構造物がない)
    #   - 床が見えない (floor_open_ratio が低い)
    top_region_v = v[:int(h * 0.40), :]
    top_region_s = s[:int(h * 0.40), :]
    top_white_uniform = float(np.mean((top_region_s < 45) & (top_region_v > 175)))
    overall_edge_density = float(np.mean(edges > 0))
    ac_close_up = (top_white_uniform > 0.50 and overall_edge_density < 0.10 and floor_open_ratio < 0.15)

    if ac_close_up:
        close_up_penalty = 0.45
    elif center_white > 0.75 and v_std < 35:
        close_up_penalty = 0.35
    elif center_white > 0.65 and v_std < 50:
        close_up_penalty = 0.15
    else:
        close_up_penalty = 0.0

    # スコア: 床面積が広く見える（floor_open_ratio + room_likeness）を重視し、白一色接写を下げる
    score = (
        brightness * 0.15
        + white_ratio * 0.25
        + floor_open_ratio * 0.35
        + room_likeness * 0.25
        - close_up_penalty
    )
    score = max(0.0, min(1.0, score))
    return ImageQuality(
        index=index,
        score=score,
        brightness=brightness,
        white_ratio=white_ratio,
        floor_open_ratio=floor_open_ratio,
        room_likeness=room_likeness,
        close_up_penalty=close_up_penalty,
    )


def pick_cover_index(image_urls: List[str]) -> Tuple[int, List[Dict[str, Any]]]:
    metrics: List[ImageQuality] = []
    for i, url in enumerate(image_urls):
        raw = fetch_image_bytes(url)
        if not raw:
            continue
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        metrics.append(evaluate_image_quality(img, i))

    if not metrics:
        return 0, []

    best = max(metrics, key=lambda x: x.score)
    serial = [
        {
            "index": m.index,
            "score": round(m.score, 4),
            "brightness": round(m.brightness, 4),
            "white_ratio": round(m.white_ratio, 4),
            "floor_open_ratio": round(m.floor_open_ratio, 4),
            "room_likeness": round(getattr(m, "room_likeness", 0.5), 4),
            "close_up_penalty": round(getattr(m, "close_up_penalty", 0.0), 4),
        }
        for m in metrics
    ]
    return best.index, serial


def yahoo_route_check(from_station: str, to_station: str) -> RouteCheck:
    from_station = re.sub(r"\\s+", "", from_station or "")
    if not from_station:
        return RouteCheck(destination=to_station, minutes=None, transfers=None, ok=False, source="no_from_station")

    url = "https://transit.yahoo.co.jp/search/result"
    params = {"from": from_station, "to": to_station, "y": datetime.now().year, "m": datetime.now().month, "d": datetime.now().day, "hh": "09", "m1": "0", "m2": "0", "type": "1"}
    try:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT, headers={"User-Agent": UA})
        r.raise_for_status()
        html = r.text

        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text("\n", strip=True)

        time_matches = [int(x) for x in re.findall(r"(\\d{1,3})分", text)]
        transfer_matches = [int(x) for x in re.findall(r"乗換\\s*[:：]?\\s*(\\d+)回", text)]

        minutes = min(time_matches) if time_matches else None
        transfers = min(transfer_matches) if transfer_matches else None

        ok = False
        if minutes is not None and transfers is not None:
            ok = (transfers == 0) or (minutes <= 40 and transfers <= 1)
        elif minutes is not None:
            ok = minutes <= 40

        return RouteCheck(destination=to_station, minutes=minutes, transfers=transfers, ok=ok, source=r.url)
    except Exception:
        return RouteCheck(destination=to_station, minutes=None, transfers=None, ok=False, source="route_fetch_failed")


def analyze_station_access(station: str) -> Dict[str, Any]:
    checks = [yahoo_route_check(station, dest) for dest in MAIN_STATIONS]
    pass_any = any(c.ok for c in checks)
    parsed_any = any(c.minutes is not None for c in checks)

    rows = []
    for c in checks:
        rows.append({
            "destination": c.destination,
            "minutes": c.minutes,
            "transfers": c.transfers,
            "ok": c.ok,
            "source": c.source,
        })

    return {
        "station": station,
        "pass_any": pass_any,
        "parsed_any": parsed_any,
        "details": rows,
    }


def gemini_review(payload: Dict[str, Any]) -> Dict[str, Any]:
    # 「必ず解析」はここで実行。キーが無い場合でもヒューリスティック解析を返す。
    api_key = (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("GEMINI_API", "").strip()
    )
    heuristic_ok = bool(payload.get("image_quality_ok"))

    if not api_key or genai is None:
        return {
            "engine": "heuristic",
            "decision": "adopt" if heuristic_ok else "bots",
            "reason": "gemini_api_unavailable",
        }

    prompt = (
        "あなたは不動産SNS投稿の審査AIです。JSONを読み、adopt か bots を返してください。"
        "基準: 明るい白基調・床面積が広く見える画像を優先。"
        "駅アクセス条件は今回の判定から除外してください。"
        "出力はJSONのみ {\"decision\":\"adopt|bots\",\"reason\":\"...\",\"priority\":\"high|normal|low\"}"
        f"\n入力: {json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = (resp.text or "").strip()
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError("json block not found")
        parsed = json.loads(m.group(0))
        decision = str(parsed.get("decision", "")).lower()
        if decision not in ("adopt", "bots"):
            decision = "adopt" if heuristic_ok else "bots"
        return {
            "engine": "gemini-2.5-flash",
            "decision": decision,
            "reason": parsed.get("reason", ""),
            "priority": parsed.get("priority", "normal"),
        }
    except Exception as e:
        return {
            "engine": "heuristic_fallback",
            "decision": "adopt" if heuristic_ok else "bots",
            "reason": f"gemini_error:{e}",
        }


def to_video_record(detail: Dict[str, Any], access: Dict[str, Any], cover_index: int, image_metrics: List[Dict[str, Any]], review: Dict[str, Any]) -> Dict[str, Any]:
    title = detail.get("title") or "property"
    room = detail.get("room_number") or ""
    pid = base.sanitize_filename(f"{title}_{room}") or f"prop_{int(time.time() * 1000)}"

    image_urls = detail.get("image_urls") or []
    if 0 <= cover_index < len(image_urls):
        ordered_images = [image_urls[cover_index]] + [u for i, u in enumerate(image_urls) if i != cover_index]
    else:
        ordered_images = image_urls

    stations = detail.get("stations") or []
    station = stations[0] if stations else ""

    facilities = detail.get("facilities") or []
    picked_features = [f for f in TARGET_FEATURES if any(f in str(v) for v in facilities)]
    for fallback in ["オートロック", "宅配ボックス", "南向き", "角部屋"]:
        if len(picked_features) >= 3:
            break
        if fallback not in picked_features:
            picked_features.append(fallback)

    new_flag = is_new_building(detail)
    layout = str(detail.get("layout") or "")

    return {
        "id": pid,
        "images": ordered_images,
        "price": format_price(detail.get("rent")),
        "layout": layout,
        "station": str(station),
        "features": picked_features[:3],
        "is_new_building": new_flag,
        "overlay_headline": ("新築 " if new_flag else "") + layout,
        "cover_index_original": cover_index,
        "image_metrics": image_metrics,
        "access_check": access,
        "review": review,
        "selection_tag": "採用データ" if review.get("decision") == "adopt" else "BOTS",
        "detail_url": detail.get("detail_url", ""),
    }


def scrape_and_review(context, urls: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    adopted: List[Dict[str, Any]] = []
    bots: List[Dict[str, Any]] = []
    attempted_urls: List[str] = []

    for idx, url in enumerate(urls, start=1):
        attempted_urls.append(url)
        page = context.new_page()
        try:
            log(f"詳細取得 {idx}/{len(urls)}: {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
            base.fully_render(page)
            base.wait_for_detail_content(page, timeout_ms=7000)
            body = page.evaluate("document.body.innerText")

            detail = base.extract_property_details(page, body)
            detail["detail_url"] = url
            image_urls = detail.get("image_urls") or []
            if len(image_urls) < 5:
                continue

            station = (detail.get("stations") or [""])[0]
            access = {
                "station": station,
                "pass_any": True,
                "parsed_any": False,
                "details": [],
                "source": "access_check_disabled",
            }

            cover_index, metrics = pick_cover_index(image_urls)
            top_score = max([m["score"] for m in metrics], default=0.0)
            image_quality_ok = top_score >= 0.43

            review_input = {
                "title": detail.get("title", ""),
                "layout": detail.get("layout", ""),
                "rent": detail.get("rent", ""),
                "station": station,
                "is_new_building": is_new_building(detail),
                "image_quality_ok": image_quality_ok,
                "top_image_score": top_score,
                "facilities": detail.get("facilities", []),
            }
            review = gemini_review(review_input)

            record = to_video_record(detail, access, cover_index, metrics, review)
            if not record["price"] or not record["layout"]:
                continue

            if record["selection_tag"] == "採用データ":
                adopted.append(record)
            else:
                bots.append(record)

            if 0 < MAX_PROPERTIES <= (len(adopted) + len(bots)):
                break
        except Exception as e:
            log(f"[WARN] 詳細取得失敗: {e}")
        finally:
            page.close()

    return adopted, bots, attempted_urls


def save_outputs(adopted: List[Dict[str, Any]], bots: List[Dict[str, Any]], filtered_urls: List[str], attempted_urls: List[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    uniq_adopted: Dict[str, Dict[str, Any]] = {}
    for r in adopted:
        rid = r.get("id")
        if rid and rid not in uniq_adopted:
            uniq_adopted[rid] = r

    uniq_bots: Dict[str, Dict[str, Any]] = {}
    for r in bots:
        rid = r.get("id")
        if rid and rid not in uniq_bots:
            uniq_bots[rid] = r

    adopted_list = list(uniq_adopted.values())
    bots_list = list(uniq_bots.values())

    with OUTPUT_DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(adopted_list, f, ensure_ascii=False, indent=2)

    with (OUTPUT_DIR / "BOTS_list.json").open("w", encoding="utf-8") as f:
        json.dump(bots_list, f, ensure_ascii=False, indent=2)

    with (OUTPUT_DIR / "filtered_urls.json").open("w", encoding="utf-8") as f:
        json.dump(filtered_urls, f, ensure_ascii=False, indent=2)

    all_rows = adopted_list + bots_list
    pd.DataFrame(all_rows).to_csv(OUTPUT_DIR / "review_summary.csv", index=False, encoding="utf-8-sig")

    prev_seen = load_seen_urls()
    now_seen = {u for u in attempted_urls if u and "/rent_rooms/" in u}
    now_seen.update(
        str(r.get("detail_url") or "").strip()
        for r in adopted_list + bots_list
        if r.get("detail_url") and "/rent_rooms/" in str(r.get("detail_url"))
    )
    merged_seen = prev_seen | now_seen
    save_seen_urls(merged_seen)

    saved_adopt = save_record_images(adopted_list, "adopted")
    saved_bots = save_record_images(bots_list, "bots")

    log(f"保存完了: {OUTPUT_DATA_PATH} (採用 {len(adopted_list)}件)")
    log(f"保存完了: {OUTPUT_DIR / 'BOTS_list.json'} (BOTS {len(bots_list)}件)")
    log(f"保存完了: {SEEN_URLS_PATH} (累計 {len(merged_seen)}件)")
    if SAVE_REVIEW_IMAGES:
        log(f"画像保存: adopted={saved_adopt}枚, bots={saved_bots}枚 -> {SAVED_IMAGES_DIR}")


def main() -> None:
    log("専用スクレイパーを開始")
    log("ログイン処理は scrape_itanji.auto_login をそのまま利用")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context()
        base.configure_context(context)
        page = context.new_page()

        page.goto(TOP_URL, wait_until="load", timeout=60000)
        if not base.auto_login(page):
            raise SystemExit("ログインに失敗しました")

        if not apply_search_filters(page):
            raise SystemExit("検索条件の適用に失敗しました")

        filtered_urls_all = collect_filtered_urls(page)
        seen_urls = load_seen_urls()
        filtered_urls = [u for u in filtered_urls_all if u not in seen_urls]
        skipped_seen = len(filtered_urls_all) - len(filtered_urls)
        log(f"抽出URL件数(申込なし+画像{MIN_IMAGES}枚以上): {len(filtered_urls_all)}")
        log(f"再取得除外件数(seen): {skipped_seen}")
        log(f"今回の詳細取得対象件数: {len(filtered_urls)}")

        adopted, bots, attempted_urls = scrape_and_review(context, filtered_urls)
        save_outputs(adopted, bots, filtered_urls, attempted_urls)

        browser.close()


if __name__ == "__main__":
    # 初回実行時:
    #   pip install -r requirements.txt
    #   playwright install
    # Geminiを使う場合:
    #   set GEMINI_API_KEY=your_api_key
    main()
