import shutil
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import requests

ModelName = Literal["edsr", "fsrcnn"]

MODEL_DIR = Path("assets/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SOURCES = {
    "edsr": {
        "filename": "EDSR_x2.pb",
        "scale": 2,
        "url": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
    },
    "fsrcnn": {
        "filename": "FSRCNN_x2.pb",
        "scale": 2,
        "url": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
    },
}


def _download_file(url: str, dst: Path, timeout: int = 45) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
    tmp.replace(dst)


def ensure_model_file(model_name: ModelName = "edsr") -> Path:
    if model_name not in MODEL_SOURCES:
        raise ValueError(f"Unsupported model: {model_name}")

    meta = MODEL_SOURCES[model_name]
    path = MODEL_DIR / meta["filename"]
    if path.exists() and path.stat().st_size > 0:
        return path

    _download_file(meta["url"], path)
    return path


def _create_superres(model_name: ModelName = "edsr"):
    model_path = ensure_model_file(model_name)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel(model_name, MODEL_SOURCES[model_name]["scale"])
    return sr


def _imread_any_path(image_path: Path):
    # OpenCV on Windows may fail for non-ASCII paths; use fromfile+imdecode.
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_any_path(output_path: Path, img: np.ndarray) -> bool:
    suffix = output_path.suffix.lower() or ".png"
    ext = ".jpg" if suffix == ".jpeg" else suffix
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(output_path))
    return True


def upscale_image_cpu(
    image_path: str | Path,
    output_path: str | Path,
    target_width: int = 1080,
    preferred_model: ModelName = "edsr",
) -> Path:
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = _imread_any_path(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    h, w = img.shape[:2]
    if w >= target_width:
        if not _imwrite_any_path(output_path, img):
            raise RuntimeError(f"Failed to write image: {output_path}")
        return output_path

    upscaled = None
    try:
        sr = _create_superres(preferred_model)
        upscaled = sr.upsample(img)
    except Exception:
        fallback = "fsrcnn" if preferred_model == "edsr" else "edsr"
        try:
            sr = _create_superres(fallback)
            upscaled = sr.upsample(img)
        except Exception:
            upscaled = None

    if upscaled is None:
        scale = target_width / float(w)
        new_h = max(1, int(h * scale))
        upscaled = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_CUBIC)

    uh, uw = upscaled.shape[:2]
    if uw < target_width:
        scale = target_width / float(uw)
        new_h = max(1, int(uh * scale))
        upscaled = cv2.resize(upscaled, (target_width, new_h), interpolation=cv2.INTER_CUBIC)

    if not _imwrite_any_path(output_path, upscaled):
        raise RuntimeError(f"Failed to write image: {output_path}")
    return output_path


def warmup_models(primary: ModelName = "edsr") -> None:
    try:
        ensure_model_file(primary)
    except Exception:
        ensure_model_file("fsrcnn")


def clear_model_cache() -> None:
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
