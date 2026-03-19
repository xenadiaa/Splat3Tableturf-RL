from __future__ import annotations

import argparse
import glob
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tableturf_vision.map_state_detector import (
    MAP_NAMES,
    _classify_cell,
    _extract_pure_green_points,
    _map_nonzero_columns,
    _sample_patch_mean_bgr,
    render_map_state_grid,
)


SETTLEMENT_COORD_DIR = REPO_ROOT / "tableturf_vision" / "参照基础_坐标点确定"


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


def _coord_image_path(map_name: str) -> Path:
    return SETTLEMENT_COORD_DIR / f"{map_name}_结算.png"


@lru_cache(maxsize=None)
def load_settlement_map_reference_points(map_name: str) -> List[Dict]:
    if map_name not in MAP_NAMES:
        raise ValueError(f"unsupported map: {map_name}")
    raw_points = _extract_pure_green_points(_coord_image_path(map_name))
    row_columns = _map_nonzero_columns(map_name)
    expected_count = sum(len(cols) for cols in row_columns)
    if len(raw_points) != expected_count:
        raise ValueError(
            f"settlement reference point count mismatch for {map_name}: "
            f"green_points={len(raw_points)} json_nonzero={expected_count}"
        )

    mapped_points: List[Dict] = []
    offset = 0
    for row_idx, columns in enumerate(row_columns):
        row_count = len(columns)
        row_points = raw_points[offset : offset + row_count]
        if len(row_points) != row_count:
            raise ValueError(
                f"settlement row point count mismatch for {map_name} row {row_idx}: "
                f"got={len(row_points)} expected={row_count}"
            )
        row_points = sorted(row_points, key=lambda p: p["center"][0])
        for col_idx, point in zip(columns, row_points):
            mapped = dict(point)
            mapped["json_row"] = row_idx
            mapped["json_col"] = col_idx
            mapped_points.append(mapped)
        offset += row_count
    return mapped_points


def load_all_settlement_map_reference_points() -> Dict[str, List[Dict]]:
    return {name: load_settlement_map_reference_points(name) for name in MAP_NAMES}


def _scaled_reference_points(map_name: str, img_shape: tuple[int, int, int]) -> List[Dict]:
    h, w = img_shape[:2]
    out: List[Dict] = []
    for p in load_settlement_map_reference_points(map_name):
        cx = float(p["center_norm"][0]) * float(w)
        cy = float(p["center_norm"][1]) * float(h)
        radius = max(4.0, float(p["radius_norm"]) * float(max(w, h)))
        out.append(
            {
                "center": [cx, cy],
                "radius": radius,
                "json_row": int(p["json_row"]),
                "json_col": int(p["json_col"]),
            }
        )
    return out


def analyze_settlement_map_state(frame_bgr: np.ndarray, map_name: str) -> Dict:
    if map_name not in MAP_NAMES:
        raise ValueError(f"unsupported map: {map_name}")

    ref_points = _scaled_reference_points(map_name, frame_bgr.shape)
    cells: List[Dict] = []
    counts = {
        "p1_fill": 0,
        "p2_fill": 0,
        "p1_special": 0,
        "p2_special": 0,
        "conflict": 0,
        "transparent": 0,
    }

    for idx, p in enumerate(ref_points, start=1):
        mean_bgr = _sample_patch_mean_bgr(frame_bgr, p["center"][0], p["center"][1], p["radius"])
        label, scores = _classify_cell(mean_bgr)
        counts[label] += 1
        cells.append(
            {
                "index": idx,
                "center": [round(float(p["center"][0]), 2), round(float(p["center"][1]), 2)],
                "radius": round(float(p["radius"]), 2),
                "json_row": int(p["json_row"]),
                "json_col": int(p["json_col"]),
                "mean_bgr": [round(float(v), 2) for v in mean_bgr.tolist()],
                "label": label,
                "scores": {k: round(float(v), 4) for k, v in scores.items()},
            }
        )

    return {
        "map_name": map_name,
        "reference_point_count": len(cells),
        "counts": counts,
        "cells": cells,
    }


def analyze_settlement_map_state_image_path(image_path: Path, map_name: str) -> Dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")
    result = analyze_settlement_map_state(frame, map_name)
    result["image"] = str(image_path)
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect settlement-time cell-state labels for one of the 15 maps.")
    p.add_argument("--map-name", default="", help="Chinese map name")
    p.add_argument("--image", default="", help="image path; empty means latest capture image")
    p.add_argument("--input-dir", default="vision_capture/debug")
    p.add_argument("--json", action="store_true", help="print full JSON result")
    p.add_argument("--print-grid", action="store_true", help="print terminal map-structure preview")
    p.add_argument("--no-color", action="store_true", help="disable ANSI color in --print-grid output")
    p.add_argument("--show-reference-points", action="store_true", help="print green-circle reference points for --map-name")
    p.add_argument("--show-all-reference-points", action="store_true", help="print reference points for all 15 settlement maps")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.show_all_reference_points:
        payload = {}
        for map_name in MAP_NAMES:
            payload[map_name] = {
                "count": len(load_settlement_map_reference_points(map_name)),
                "points": [
                    {
                        "index": i + 1,
                        "center": [round(float(p["center"][0]), 2), round(float(p["center"][1]), 2)],
                        "radius": round(float(p["radius"]), 2),
                    }
                    for i, p in enumerate(load_settlement_map_reference_points(map_name))
                ],
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not args.map_name:
        raise ValueError("--map-name is required unless --show-all-reference-points is used")

    if args.show_reference_points:
        payload = {
            "map_name": args.map_name,
            "count": len(load_settlement_map_reference_points(args.map_name)),
            "points": [
                {
                    "index": i + 1,
                    "center": [round(float(p["center"][0]), 2), round(float(p["center"][1]), 2)],
                    "radius": round(float(p["radius"]), 2),
                }
                for i, p in enumerate(load_settlement_map_reference_points(args.map_name))
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    image = _resolve_image(args.image, args.input_dir)
    result = analyze_settlement_map_state_image_path(image, args.map_name)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.print_grid:
        print(f"Image: {result['image']}")
        print(f"Map: {args.map_name}")
        print(f"Counts: {json.dumps(result['counts'], ensure_ascii=False)}")
        print("[Grid]")
        for line in render_map_state_grid(args.map_name, result, colorize=not args.no_color):
            print(line)
    else:
        print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
