from __future__ import annotations

import argparse
import glob
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAP_REFERENCE_DIR = REPO_ROOT / "tableturf_vision" / "参照基础"
MAP_COORD_DIR = REPO_ROOT / "tableturf_vision" / "参照基础_坐标点确定"
MAP_NAMES = [
    "加速高速公路",
    "双子岛",
    "小巧竞技场",
    "扭转河",
    "正方广场",
    "正直大道",
    "清脆饼干",
    "漆黑深林",
    "罚分花园",
    "轻飘湖",
    "酷似大道",
    "钢骨建筑",
    "间隔墙",
    "雷霆车站",
    "面具屋",
]
MAP_INFO_PATH = REPO_ROOT / "tableturf_sim" / "data" / "maps" / "MiniGameMapInfo.json"

CLR_RESET = "\033[0m"
RGB = {
    "transparent": (0, 0, 0),
    "p1_fill": (255, 255, 0),
    "p1_special": (255, 192, 0),
    "p2_fill": (0, 112, 192),
    "p2_special": (0, 176, 240),
    "conflict": (200, 200, 200),
    "invalid": (80, 80, 80),
}
TOKEN_MAP = {
    "invalid": "  ",
    "transparent": "[]",
    "p1_fill": "[]",
    "p1_special": "[]",
    "p2_fill": "[]",
    "p2_special": "[]",
    "conflict": "[]",
}

CellPos = Tuple[int, int]


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
    return MAP_COORD_DIR / f"{map_name}.png"


def _reference_image_path(map_name: str) -> Path:
    return MAP_REFERENCE_DIR / f"{map_name}.png"


@lru_cache(maxsize=1)
def _load_map_info() -> Dict[str, Dict]:
    rows = json.loads(MAP_INFO_PATH.read_text(encoding="utf-8"))
    return {str(row["name"]): row for row in rows}


def _extract_pure_green_points(image_path: Path) -> List[Dict]:
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read coordinate image: {image_path}")
    bgr = img[:, :, :3] if img.shape[2] == 4 else img
    mask = (
        (bgr[:, :, 0] == 0)
        & (bgr[:, :, 1] >= 242)
        & (bgr[:, :, 2] <= 20)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = bgr.shape[:2]
    points: List[Dict] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < 5 or radius > 18:
            continue
        points.append(
            {
                "center": [float(cx), float(cy)],
                "center_norm": [float(cx) / float(w), float(cy) / float(h)],
                "radius": float(radius),
                "radius_norm": float(radius / max(w, h)),
            }
        )
    points.sort(key=lambda p: (p["center"][1], p["center"][0]))
    return points


def _map_nonzero_columns(map_name: str) -> List[List[int]]:
    map_info = _load_map_info().get(map_name)
    if map_info is None:
        raise ValueError(f"map info not found: {map_name}")
    rows: List[List[int]] = []
    for row in map_info["point_type"]:
        rows.append([col_idx for col_idx, value in enumerate(row) if int(value) != 0])
    return rows


@lru_cache(maxsize=None)
def load_map_reference_points(map_name: str) -> List[Dict]:
    if map_name not in MAP_NAMES:
        raise ValueError(f"unsupported map: {map_name}")
    raw_points = _extract_pure_green_points(_coord_image_path(map_name))
    row_columns = _map_nonzero_columns(map_name)
    expected_count = sum(len(cols) for cols in row_columns)
    if len(raw_points) != expected_count:
        raise ValueError(
            f"reference point count mismatch for {map_name}: "
            f"green_points={len(raw_points)} json_nonzero={expected_count}"
        )

    mapped_points: List[Dict] = []
    offset = 0
    for row_idx, columns in enumerate(row_columns):
        row_count = len(columns)
        row_points = raw_points[offset : offset + row_count]
        if len(row_points) != row_count:
            raise ValueError(
                f"row point count mismatch for {map_name} row {row_idx}: "
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


def load_all_map_reference_points() -> Dict[str, List[Dict]]:
    return {name: load_map_reference_points(name) for name in MAP_NAMES}


def _scaled_reference_points(map_name: str, img_shape: tuple[int, int, int]) -> List[Dict]:
    h, w = img_shape[:2]
    out: List[Dict] = []
    for p in load_map_reference_points(map_name):
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


def _sample_patch_mean_bgr(frame_bgr: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    r = max(3, int(round(radius * 0.6)))
    ix, iy = int(round(cx)), int(round(cy))
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ph, pw = patch.shape[:2]
    # Use a vertical sliding gradient: strongest at the top edge, linearly
    # down to 0.3 at the midline, then continue decaying toward the bottom.
    # This keeps the tile body dominant while strongly suppressing rising
    # effects from the lower edge.
    row_idx = np.arange(ph, dtype=np.float32)
    if ph == 1:
        weights_1d = np.array([1.0], dtype=np.float32)
    else:
        mid = max(1.0, (ph - 1) / 2.0)
        top_to_mid = np.clip(row_idx / mid, 0.0, 1.0)
        bot_span = max(1.0, (ph - 1) - mid)
        mid_to_bot = np.clip((row_idx - mid) / bot_span, 0.0, 1.0)
        weights_1d = np.where(
            row_idx <= mid,
            1.0 - 0.7 * top_to_mid,
            0.3 - 0.22 * mid_to_bot,
        ).astype(np.float32)
    row_weights = weights_1d.reshape(ph, 1, 1)
    weighted_patch = patch.astype(np.float32) * row_weights
    mean_bgr = weighted_patch.sum(axis=(0, 1)) / (row_weights.sum() * float(pw))
    return mean_bgr.astype(np.float32)


def _classify_cell(mean_bgr: np.ndarray) -> tuple[str, Dict[str, float], bool]:
    hsv = cv2.cvtColor(np.uint8([[mean_bgr.astype(np.uint8)]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    b, g, r = [float(x) for x in mean_bgr.tolist()]

    scores = {
        "p1_fill": 0.0,
        "p2_fill": 0.0,
        "p1_special": 0.0,
        "p2_special": 0.0,
        "conflict": 0.0,
        "transparent": 0.0,
    }

    # Transparent board cells are very dark patterned tiles.
    # Recognize them explicitly so that any other uncaptured color state can be
    # treated conservatively as conflict/error instead of a playable empty cell.
    if v <= 52 or (v <= 72 and max(b, g, r) <= 78):
        scores["transparent"] = 2.0 + ((72 - min(v, 72)) / 255.0)

    if 28 <= h <= 42 and s >= 150 and v >= 180:
        scores["p1_fill"] = 1.0 + (v / 255.0)
    if (
        96 <= h <= 135
        and s >= 95
        and 80 <= v <= 255
        and b >= 90
        and b > g + 30
        and g >= 35
        and g >= r - 10
        and (b - r) >= 35
    ):
        # Enemy fill is a deeper blue than enemy special.
        # Keep it bright enough to avoid transparent board cells, but allow
        # the hue/value drift seen in real captures.
        scores["p2_fill"] = 1.0 + (s / 255.0) + min(0.35, (b - g) / 255.0)
    if 12 <= h <= 27 and s >= 140 and v >= 150:
        scores["p1_special"] = 1.0 + (v / 255.0)
    if (
        29 <= h <= 30
        and s >= 235
        and v >= 245
        and b >= 10
        and g >= 245
        and r >= 250
        and abs(g - r) <= 8
    ):
        # Activated own special can shift into the yellow fill hue band.
        # Keep this rule narrow so ordinary fill cells are not broadly affected.
        scores["p1_special"] = max(
            scores["p1_special"],
            1.15 + (v / 255.0) + 0.1,
        )
    if 80 <= h <= 102 and s >= 90 and v >= 180:
        scores["p2_special"] = 1.0 + (v / 255.0)
    if (
        86 <= h <= 94
        and 8 <= s <= 45
        and v >= 245
        and b >= 245
        and g >= 245
        and 210 <= r <= 242
        and (b - r) >= 10
        and (g - r) >= 10
    ):
        # Activated enemy special can wash out into a bright cyan-white.
        scores["p2_special"] = max(
            scores["p2_special"],
            1.12 + (v / 255.0),
        )
    if s <= 45 and v >= 150:
        scores["conflict"] = 1.0 + (v / 255.0)

    is_error = False
    if max(scores.values()) <= 0.0:
        scores["conflict"] = 0.95
        is_error = True

    priority = ["p1_fill", "p2_fill", "p1_special", "p2_special", "conflict", "transparent"]
    label = max(priority, key=lambda name: (scores[name], -priority.index(name)))
    return label, scores, is_error


def detect_map_state(frame_bgr: np.ndarray, map_name: str) -> Dict:
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
    error_cells: List[Dict] = []

    for idx, p in enumerate(ref_points, start=1):
        mean_bgr = _sample_patch_mean_bgr(frame_bgr, p["center"][0], p["center"][1], p["radius"])
        label, scores, is_error = _classify_cell(mean_bgr)
        counts[label] += 1
        cell = {
            "index": idx,
            "center": [round(float(p["center"][0]), 2), round(float(p["center"][1]), 2)],
            "radius": round(float(p["radius"]), 2),
            "json_row": int(p["json_row"]),
            "json_col": int(p["json_col"]),
            "mean_bgr": [round(float(v), 2) for v in mean_bgr.tolist()],
            "label": label,
            "scores": {k: round(float(v), 4) for k, v in scores.items()},
            "is_error": bool(is_error),
        }
        cells.append(cell)
        if is_error:
            error_cells.append(
                {
                    "index": idx,
                    "json_row": int(p["json_row"]),
                    "json_col": int(p["json_col"]),
                    "center": [round(float(p["center"][0]), 2), round(float(p["center"][1]), 2)],
                    "mean_bgr": [round(float(v), 2) for v in mean_bgr.tolist()],
                }
            )

    return {
        "map_name": map_name,
        "reference_point_count": len(cells),
        "counts": counts,
        "error_count": len(error_cells),
        "error_cells": error_cells,
        "cells": cells,
    }


def _recount_labels(cells: List[Dict]) -> Dict[str, int]:
    counts = {
        "p1_fill": 0,
        "p2_fill": 0,
        "p1_special": 0,
        "p2_special": 0,
        "conflict": 0,
        "transparent": 0,
    }
    for cell in cells:
        label = str(cell["label"])
        if label in counts:
            counts[label] += 1
    return counts


def apply_p1_special_persistence(result: Dict, persisted_positions: Set[CellPos] | None = None) -> Dict:
    # Deprecated:
    # The older "persisted special positions" workaround is intentionally disabled.
    # Special/fill recognition now relies on direct vision classification plus
    # current-action correction only, so we pass the result through unchanged.
    return result


def _placed_card_special_positions(card_number: int, rotation: int, x: int, y: int) -> Set[CellPos]:
    from tableturf_sim.src.utils.common_utils import _card_cells_on_map, create_card_from_id

    card = create_card_from_id(int(card_number))
    positions: Set[CellPos] = set()
    for map_x, map_y, cell_type in _card_cells_on_map(card, int(x), int(y), int(rotation)):
        if int(cell_type) == 2:
            positions.add((int(map_y), int(map_x)))
    return positions


def apply_current_action_special_correction(
    result: Dict,
    card_number: int,
    rotation: int,
    x: int,
    y: int,
) -> Dict:
    # Deprecated:
    # Current-action special correction is intentionally disabled because the
    # latest vision pipeline can distinguish special vs fill directly.
    return result


class MapStateTracker:
    def __init__(self, map_name: str):
        if map_name not in MAP_NAMES:
            raise ValueError(f"unsupported map: {map_name}")
        self.map_name = map_name

    def reset(self) -> None:
        # Deprecated: no persisted special-position state is kept anymore.
        return None

    def update_frame(self, frame_bgr: np.ndarray) -> Dict:
        result = detect_map_state(frame_bgr, self.map_name)
        return result

    def update_image_path(self, image_path: Path) -> Dict:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"cannot read image: {image_path}")
        result = self.update_frame(frame)
        result["image"] = str(image_path)
        return result

    def update_frame_with_action(
        self,
        frame_bgr: np.ndarray,
        card_number: int,
        rotation: int,
        x: int,
        y: int,
    ) -> Dict:
        result = detect_map_state(frame_bgr, self.map_name)
        return result

    def update_image_path_with_action(
        self,
        image_path: Path,
        card_number: int,
        rotation: int,
        x: int,
        y: int,
    ) -> Dict:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"cannot read image: {image_path}")
        result = self.update_frame_with_action(frame, card_number, rotation, x, y)
        result["image"] = str(image_path)
        return result


def _build_grid_labels(map_name: str, cells: List[Dict]) -> List[List[str]]:
    map_info = _load_map_info().get(map_name)
    if map_info is None:
        raise ValueError(f"map info not found: {map_name}")
    point_type = map_info["point_type"]
    label_by_pos = {
        (int(cell["json_row"]), int(cell["json_col"])): str(cell["label"])
        for cell in cells
    }
    grid: List[List[str]] = []
    for row_idx, row in enumerate(point_type):
        out_row: List[str] = []
        for col_idx, v in enumerate(row):
            if int(v) == 0:
                out_row.append("invalid")
            else:
                out_row.append(label_by_pos.get((row_idx, col_idx), "transparent"))
        grid.append(out_row)
    return grid


def _rgb(text: str, rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m{text}{CLR_RESET}"


def render_map_state_grid(map_name: str, result: Dict, colorize: bool = True) -> List[str]:
    grid = _build_grid_labels(map_name, result["cells"])
    lines: List[str] = []
    for row in grid:
        parts: List[str] = []
        for label in row:
            token = TOKEN_MAP.get(label, "??")
            if token.strip() == "":
                parts.append(token)
            elif colorize:
                parts.append(_rgb(token, RGB.get(label, (255, 255, 255))))
            else:
                parts.append(token)
        lines.append("".join(parts))
    return lines


def detect_map_state_image_path(image_path: Path, map_name: str) -> Dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")
    result = detect_map_state(frame, map_name)
    result["image"] = str(image_path)
    return result


def _make_map_wrapper(map_name: str) -> Callable[[np.ndarray], Dict]:
    def _wrapper(frame_bgr: np.ndarray) -> Dict:
        return detect_map_state(frame_bgr, map_name)

    _wrapper.__name__ = f"detect_map_state__{map_name}"
    _wrapper.__doc__ = f"Detect map-state cells for {map_name}."
    return _wrapper


for _map_name in MAP_NAMES:
    globals()[f"detect_map_state__{_map_name}"] = _make_map_wrapper(_map_name)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect current cell-state labels for one of the 15 reference maps.")
    p.add_argument("--map-name", default="", help="Chinese map name")
    p.add_argument("--image", default="", help="image path; empty means latest capture image")
    p.add_argument("--input-dir", default="vision_capture/debug")
    p.add_argument("--json", action="store_true", help="print full JSON result")
    p.add_argument("--print-grid", action="store_true", help="print terminal map-structure preview")
    p.add_argument("--no-color", action="store_true", help="disable ANSI color in --print-grid output")
    p.add_argument("--show-reference-points", action="store_true", help="print green-circle reference points for --map-name")
    p.add_argument("--show-all-reference-points", action="store_true", help="print reference points for all 15 maps")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.show_all_reference_points:
        payload = {}
        for map_name in MAP_NAMES:
            payload[map_name] = {
                "count": len(load_map_reference_points(map_name)),
                "points": [
                    {
                        "index": i + 1,
                        "center": [round(float(p["center"][0]), 2), round(float(p["center"][1]), 2)],
                        "radius": round(float(p["radius"]), 2),
                    }
                    for i, p in enumerate(load_map_reference_points(map_name))
                ],
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not args.map_name:
        raise ValueError("--map-name is required unless --show-all-reference-points is used")

    if args.show_reference_points:
        payload = {
            "map_name": args.map_name,
            "count": len(load_map_reference_points(args.map_name)),
            "points": [
                {
                    "index": i + 1,
                    "center": [round(float(p["center"][0]), 2), round(float(p["center"][1]), 2)],
                    "radius": round(float(p["radius"]), 2),
                }
                for i, p in enumerate(load_map_reference_points(args.map_name))
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    image = _resolve_image(args.image, args.input_dir)
    result = detect_map_state_image_path(image, args.map_name)
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
