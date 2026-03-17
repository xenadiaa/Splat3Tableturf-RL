#!/usr/bin/env python3
"""Infer both players' actions from before/after map snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.action_infer_utils import infer_actions_from_map_transition


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Infer possible action pairs from before/after map snapshots")
    p.add_argument("input_json", help="JSON file with map_id/before_grid/after_grid/p1_hand/p2_hand[/p1_sp/p2_sp/turn]")
    p.add_argument("--max-results", type=int, default=128)
    return p


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    result = infer_actions_from_map_transition(
        map_id=str(payload["map_id"]),
        before_grid=payload["before_grid"],
        after_grid=payload["after_grid"],
        p1_deck=[int(x) for x in payload.get("p1_deck", [])] or None,
        p2_deck=[int(x) for x in payload.get("p2_deck", [])] or None,
        p1_hand=[int(x) for x in payload.get("p1_hand", [])] or None,
        p2_hand=[int(x) for x in payload.get("p2_hand", [])] or None,
        p1_sp=(int(payload["p1_sp"]) if "p1_sp" in payload and payload["p1_sp"] is not None else None),
        p2_sp=(int(payload["p2_sp"]) if "p2_sp" in payload and payload["p2_sp"] is not None else None),
        turn=int(payload.get("turn", 1)),
        max_results=int(args.max_results),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
