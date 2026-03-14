from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

# Confirmed from all 15 reference PNGs in tableturf_vision/参照基础.
PLAYABLE_BANNER_REF_SIZE = (1920, 1080)
PLAYABLE_BANNER_ROI_ABS = (20, 111, 262, 159)
PLAYABLE_BANNER_ROI_NORM = (
    PLAYABLE_BANNER_ROI_ABS[0] / PLAYABLE_BANNER_REF_SIZE[0],
    PLAYABLE_BANNER_ROI_ABS[1] / PLAYABLE_BANNER_REF_SIZE[1],
    PLAYABLE_BANNER_ROI_ABS[2] / PLAYABLE_BANNER_REF_SIZE[0],
    PLAYABLE_BANNER_ROI_ABS[3] / PLAYABLE_BANNER_REF_SIZE[1],
)


def _scale_roi(img_shape: Tuple[int, int, int], roi_norm: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    h, w = img_shape[:2]
    x = int(round(w * roi_norm[0]))
    y = int(round(h * roi_norm[1]))
    rw = int(round(w * roi_norm[2]))
    rh = int(round(h * roi_norm[3]))
    x = max(0, min(x, w - 2))
    y = max(0, min(y, h - 2))
    rw = max(2, min(rw, w - x))
    rh = max(2, min(rh, h - y))
    return x, y, rw, rh


def get_playable_banner_roi_abs(img_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    return _scale_roi(img_shape, PLAYABLE_BANNER_ROI_NORM)


def detect_playable_banner(frame_bgr: np.ndarray) -> Dict:
    x, y, w, h = get_playable_banner_roi_abs(frame_bgr.shape)
    roi = frame_bgr[y : y + h, x : x + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    gray_mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 45, 255]))
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        area = float(bw * bh)
        if area < float(w * h) * 0.35:
            continue
        rect_ratio = area / max(1.0, float(w * h))
        if best is None or rect_ratio > best["rect_ratio"]:
            best = {
                "bbox_local": [int(bx), int(by), int(bw), int(bh)],
                "bbox_abs": [int(x + bx), int(y + by), int(bw), int(bh)],
                "rect_ratio": float(rect_ratio),
            }

    gray_ratio = float((gray_mask > 0).mean())
    inner = roi[max(0, int(h * 0.18)) : max(1, int(h * 0.82)), max(0, int(w * 0.08)) : max(1, int(w * 0.92))]
    inner_hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    dark_text_ratio = float((inner_hsv[:, :, 2] <= 115).mean()) if inner.size else 0.0
    mean_bgr = [float(v) for v in roi.reshape(-1, 3).mean(axis=0)]

    is_present = bool(
        best is not None
        and gray_ratio >= 0.28
        and best["rect_ratio"] >= 0.45
        and dark_text_ratio >= 0.015
    )

    return {
        "playable": is_present,
        "roi_abs": [int(x), int(y), int(w), int(h)],
        "gray_ratio": gray_ratio,
        "dark_text_ratio": dark_text_ratio,
        "mean_bgr": mean_bgr,
        "banner_bbox_abs": (best["bbox_abs"] if best is not None else []),
        "banner_rect_ratio": (best["rect_ratio"] if best is not None else 0.0),
    }


def is_playable_frame(frame_bgr: np.ndarray) -> bool:
    return bool(detect_playable_banner(frame_bgr)["playable"])


def is_playable_image_path(image_path: Path) -> bool:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")
    return is_playable_frame(frame)


def _resolve_image(image: str, input_dir: str) -> Path:
    if image:
        p = Path(image)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p
    d = Path(input_dir)
    if not d.is_absolute():
        d = REPO_ROOT / d
    cands = sorted(glob.glob(str(d / "capture_*.*")))
    if not cands:
        raise FileNotFoundError(f"no capture_* image in {d}")
    return Path(cands[-1])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect whether the top-left playable banner is present.")
    p.add_argument("--image", default="", help="image path; empty means latest capture image")
    p.add_argument("--input-dir", default="vision_capture/debug")
    p.add_argument("--json", action="store_true", help="print full JSON result")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    image = _resolve_image(args.image, args.input_dir)
    frame = cv2.imread(str(image))
    if frame is None:
        raise ValueError(f"cannot read image: {image}")
    result = detect_playable_banner(frame)
    result["image"] = str(image)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("true" if result["playable"] else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
