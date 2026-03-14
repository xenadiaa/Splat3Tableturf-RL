from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from tableturf_vision.tableturf_mapper import _parse_board


REPO_ROOT = Path(__file__).resolve().parent.parent
MAP_INFO_PATH = REPO_ROOT / "tableturf_sim" / "data" / "maps" / "MiniGameMapInfo.json"
REFERENCE_TEMPLATE_DIR = REPO_ROOT / "tableturf_vision" / "参照基础"


def load_map_info() -> List[Dict]:
    return json.loads(MAP_INFO_PATH.read_text(encoding="utf-8"))


def map_name_cn_to_id() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in load_map_info():
        out[str(row.get("name", ""))] = str(row.get("id", ""))
    return out


def list_reference_template_images() -> List[Path]:
    if not REFERENCE_TEMPLATE_DIR.exists():
        return []
    return sorted(
        p
        for p in REFERENCE_TEMPLATE_DIR.glob("*.png")
        if p.is_file()
    )


def valid_mask_from_labels(labels: List[List[str]]) -> np.ndarray:
    return np.array([[0 if v == "invalid" else 1 for v in row] for row in labels], dtype=np.uint8)


def compare_masks(obs: np.ndarray, tmpl: np.ndarray) -> Dict:
    oh, ow = obs.shape[:2]
    th, tw = tmpl.shape[:2]
    if (th, tw) != (oh, ow):
        tmpl2 = cv2.resize(tmpl, (ow, oh), interpolation=cv2.INTER_NEAREST)
    else:
        tmpl2 = tmpl
    same = int((obs == tmpl2).sum())
    total = int(obs.size)
    exact_ratio = same / max(1, total)
    return {"exact_ratio": float(exact_ratio), "template_mask_resized": tmpl2}


def match_map_by_reference_board_labels(board_labels: List[List[str]], layout: Dict) -> Dict:
    # Map matching must use only the parsed center board grid.
    # Left-hand card area and right-hand play UI are excluded upstream by
    # tableturf_mapper._parse_board via layout["board_roi_norm"].
    obs_mask = valid_mask_from_labels(board_labels)
    best = {
        "enum_index": -1,
        "map_id": "",
        "map_name_zh": "",
        "score": -1.0,
        "obs_shape": [int(obs_mask.shape[0]), int(obs_mask.shape[1])] if obs_mask.size else [0, 0],
        "template_image": "",
        "board_from_template": None,
        "match_mode": "reference_png_mask_exact_ratio",
    }
    if obs_mask.size == 0:
        return best

    name2id = map_name_cn_to_id()
    maps_order = load_map_info()
    id_to_enum = {str(m.get("id", "")): i + 1 for i, m in enumerate(maps_order)}

    for template_path in list_reference_template_images():
        name_cn = template_path.stem
        map_id = name2id.get(name_cn, "")
        if not map_id:
            continue
        tmpl_frame = cv2.imread(str(template_path))
        if tmpl_frame is None:
            continue
        tmpl_board = _parse_board(tmpl_frame, layout)
        tmpl_mask = valid_mask_from_labels(tmpl_board.labels)
        comp = compare_masks(obs_mask, tmpl_mask)
        score = float(comp["exact_ratio"])
        if score > best["score"]:
            best = {
                "enum_index": int(id_to_enum.get(map_id, -1)),
                "map_id": map_id,
                "map_name_zh": name_cn,
                "score": score,
                "obs_shape": [int(obs_mask.shape[0]), int(obs_mask.shape[1])],
                "template_image": str(template_path),
                "board_from_template": tmpl_board,
                "match_mode": "reference_png_mask_exact_ratio",
            }
    return best
