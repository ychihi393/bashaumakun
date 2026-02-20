#!/usr/bin/env python3
"""
Stable Diffusion アウトペインティングモジュール

横長（landscape）画像を 4:5（1080×1350）サイズに上下拡張生成する。
CPU 動作対応 — LCM-LoRA（8ステップ高速）または DPM++（フォールバック）。

パイプラインはプロセス内でシングルトン管理するため、
2枚目以降はモデル読み込みコストなしで動作する。

.env 設定:
    USE_SD_OUTPAINTING  - 1 で有効化（デフォルト: 0）
    SD_STEPS            - ステップ数（デフォルト: 8）
    SD_GUIDANCE         - ガイダンススケール（デフォルト: 1.5）
    SD_CACHE_DIR        - モデルキャッシュディレクトリ（省略可）

初回起動時:
    runwayml/stable-diffusion-inpainting  (~4 GB) を自動ダウンロード
    latent-consistency/lcm-lora-sdv1-5   (~600 MB) を自動ダウンロード
"""

import logging
import os
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter

# SD の作業解像度（512 の倍数 / 4:5 比率）
_WORK_W = 512
_WORK_H = 640

# 最終出力解像度
_TARGET_W = 1080
_TARGET_H = 1350

# マスク境界のフェザリング量（ピクセル）
_FEATHER_RADIUS = 24

# 生成プロンプト
_PROMPT = (
    "interior room, seamless wall, ceiling, floor, "
    "clean bright real estate photo, natural continuation, high quality"
)
_NEG_PROMPT = (
    "blurry, distorted, watermark, text, logo, person, "
    "ugly, deformed, duplicate, cut off, border"
)


def is_enabled() -> bool:
    """USE_SD_OUTPAINTING=1 が設定されているか"""
    return os.getenv("USE_SD_OUTPAINTING", "0").lower() in ("1", "true", "yes")


# ─── パイプライン シングルトン ──────────────────────────────────────────────────
_pipe = None
_pipe_loaded = False


def _load_pipe():
    """パイプラインを遅延読み込み（2回目以降はキャッシュを返す）"""
    global _pipe, _pipe_loaded
    if _pipe_loaded:
        return _pipe
    _pipe_loaded = True

    if not is_enabled():
        return None

    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        cache = os.getenv("SD_CACHE_DIR", "").strip() or None

        logging.info(
            "[SD] モデルを読み込んでいます... "
            "（初回は runwayml/stable-diffusion-inpainting ≈4GB をダウンロードします）"
        )

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=cache,
        )

        # ── LCM-LoRA で 8 ステップ高速化を試みる ─────────────────────────
        _lcm_ok = False
        try:
            from diffusers import LCMScheduler
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights(
                "latent-consistency/lcm-lora-sdv1-5",
                cache_dir=cache,
            )
            pipe.fuse_lora()
            _lcm_ok = True
            logging.info("[SD] LCM-LoRA 適用完了 → 8ステップ高速モード")
        except Exception as lcm_err:
            # LCM が使えなければ DPM++ 15 ステップにフォールバック
            logging.warning(
                "[SD] LCM-LoRA 適用失敗: %s → DPM++ 15ステップで動作します", lcm_err
            )
            try:
                from diffusers import DPMSolverMultistepScheduler
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                )
            except Exception:
                pass  # デフォルトスケジューラーのまま

        pipe.to("cpu")
        pipe.enable_attention_slicing()

        # Intel CPU スレッドをフル活用
        n_threads = os.cpu_count() or 4
        torch.set_num_threads(n_threads)
        logging.info(
            "[SD] 準備完了（%s / CPU %dスレッド）",
            "LCM 8step" if _lcm_ok else "DPM++ 15step",
            n_threads,
        )
        _pipe = pipe

    except Exception as e:
        logging.warning("[SD] パイプラインの読み込みに失敗しました: %s", e)
        _pipe = None

    return _pipe


# ─── アウトペインティング本体 ──────────────────────────────────────────────────

def outpaint_to_4x5(im: Image.Image) -> Optional[Image.Image]:
    """
    横長画像を 4:5（1080×1350）に SD アウトペインティングで拡張する。

    処理フロー:
      1. 元画像を 512×640 の作業キャンバスに中央配置
      2. 上下の空きエリアをぼかし引き伸ばしでプレフィル（SD へのヒント）
      3. マスク（上下=白/生成、元画像=黒/保持）を作成、境界をフェザリング
      4. SD インペインティングで上下を補完生成
      5. 1080×1350 にアップスケール後、元画像を高解像度で中央へ再貼付

    戻り値:
      PIL.Image (1080×1350)  — 成功時
      None                   — 縦型画像・無効・失敗時（呼び出し側でフォールバック）
    """
    w, h = im.size
    if h >= w:
        return None  # 縦型はスキップ

    pipe = _load_pipe()
    if pipe is None:
        return None

    steps    = int(os.getenv("SD_STEPS",    "8"))
    guidance = float(os.getenv("SD_GUIDANCE", "1.5"))

    try:
        import torch

        # ── 元画像を作業解像度にフィット（幅 = _WORK_W）──────────────────
        scale       = _WORK_W / w
        work_h_orig = min(int(h * scale), _WORK_H - 4)
        orig_work   = im.resize((_WORK_W, work_h_orig), Image.Resampling.LANCZOS)
        y_off       = (_WORK_H - work_h_orig) // 2   # 上下の空き高さ（等分）

        # ── キャンバス: ぼかし引き伸ばし + 元画像を中央に貼る ────────────
        # ぼかしベースを背景にすることで SD への色・トーンヒントを与える
        bg_hint = im.resize((_WORK_W, _WORK_H), Image.Resampling.LANCZOS)
        bg_hint = bg_hint.filter(ImageFilter.GaussianBlur(radius=30))
        bg_hint.paste(orig_work, (0, y_off))
        canvas = bg_hint  # RGB

        # ── マスク: 白=生成エリア、黒=元画像保持エリア ───────────────────
        mask = Image.new("L", (_WORK_W, _WORK_H), 255)   # 全体を「生成」で初期化
        draw = ImageDraw.Draw(mask)
        draw.rectangle([0, y_off, _WORK_W - 1, y_off + work_h_orig - 1], fill=0)
        # 境界をフェザリングして自然なブレンドに
        if _FEATHER_RADIUS > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=_FEATHER_RADIUS))

        # ── SD インペインティング実行 ──────────────────────────────────────
        top_fill    = y_off
        bottom_fill = _WORK_H - y_off - work_h_orig
        logging.info(
            "[SD] アウトペインティング開始 — 上 %dpx / 下 %dpx 補完 (%dステップ / guidance=%.1f)",
            top_fill, bottom_fill, steps, guidance,
        )

        with torch.inference_mode():
            result_work = pipe(
                prompt=_PROMPT,
                negative_prompt=_NEG_PROMPT,
                image=canvas,
                mask_image=mask,
                height=_WORK_H,
                width=_WORK_W,
                num_inference_steps=steps,
                guidance_scale=guidance,
            ).images[0]

        # ── 1080×1350 にアップスケール ──────────────────────────────────
        result_full = result_work.resize((_TARGET_W, _TARGET_H), Image.Resampling.LANCZOS)

        # ── 元画像を高解像度で中央に再貼付（品質保持） ───────────────────
        # SD の作業解像度から生じるボケを元画像エリアだけ解消する
        full_orig_h = int(h * _TARGET_W / w)
        orig_full   = im.resize((_TARGET_W, full_orig_h), Image.Resampling.LANCZOS)
        y_full      = (_TARGET_H - full_orig_h) // 2
        result_full.paste(orig_full, (0, y_full))

        logging.info("[SD] アウトペインティング完了: %dx%d", _TARGET_W, _TARGET_H)
        return result_full

    except Exception as e:
        logging.warning("[SD] アウトペインティングに失敗しました: %s", e)
        return None
