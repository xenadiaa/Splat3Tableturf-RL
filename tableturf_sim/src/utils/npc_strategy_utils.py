"""NPC strategy table loader and NN override resolver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NPC_ASSET_DIR = PROJECT_ROOT / "src" / "assets" / "NPC"
NPC_STRATEGY_JSON = NPC_ASSET_DIR / "npc_strategies.json"


def load_npc_strategy_table() -> List[dict]:
    if not NPC_STRATEGY_JSON.exists():
        raise FileNotFoundError(f"NPC strategy table missing: {NPC_STRATEGY_JSON}")
    data = json.loads(NPC_STRATEGY_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("npc_strategies.json root must be list")
    return data


def get_npc_entry_by_name(npc_name: str) -> Optional[dict]:
    q = str(npc_name).strip().lower()
    if not q:
        return None
    for row in load_npc_strategy_table():
        if str(row.get("npc_name", "")).lower() == q:
            return row
    return None


def get_npc_entry_by_order(order: int) -> Optional[dict]:
    for row in load_npc_strategy_table():
        if int(row.get("order", -1)) == int(order):
            return row
    return None


def get_level_strategy(npc_name: str, level: int) -> Optional[dict]:
    row = get_npc_entry_by_name(npc_name)
    if not row:
        return None
    strategies = row.get("strategies", [])
    for s in strategies:
        if int(s.get("ai_level", -1)) == int(level):
            return s
    return None


def _nn_spec_path_candidates(npc_name: str) -> List[Path]:
    # Preferred naming requested by user: NPCName_nn
    return [
        NPC_ASSET_DIR / f"{npc_name}_nn.json",
        NPC_ASSET_DIR / f"{npc_name}_nn.py",
    ]


def resolve_nn_spec(npc_name: str) -> Optional[Dict[str, object]]:
    candidates = _nn_spec_path_candidates(npc_name)
    json_path = candidates[0]
    py_path = candidates[1]
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(data, dict):
            # keep explicit type if provided; default to file_action_json
            out = dict(data)
            out.setdefault("type", "file_action_json")
            return out
        return None
    if py_path.exists():
        return {
            "type": "python_module",
            "module_file": str(py_path),
            "function": "choose_action",
        }
    return None
