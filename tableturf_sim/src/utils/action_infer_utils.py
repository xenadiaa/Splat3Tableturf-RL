"""Infer player actions from before/after map snapshots."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..assets.tableturf_types import GameMap
from ..engine.env_core import Action, BotConfig, GameState, PlayerState, step
from ..engine.loaders import MAP_PADDING, load_map
from .common_utils import create_card_from_id, validate_place_card_action


def _all_card_ids() -> List[int]:
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "data" / "cards" / "MiniGameCardInfo.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(int(item["Number"]) for item in data)


def _dedupe_keep_order(values: Sequence[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for v in values:
        i = int(v)
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def _build_state_from_snapshot(
    map_id: str,
    before_grid: List[List[int]],
    p1_cards: List[int],
    p2_cards: List[int],
    p1_sp: int = 0,
    p2_sp: int = 0,
    turn: int = 1,
) -> GameState:
    base_map = load_map(map_id)
    expected_h = base_map.height - MAP_PADDING * 2
    expected_w = base_map.width - MAP_PADDING * 2
    if len(before_grid) != expected_h or any(len(r) != expected_w for r in before_grid):
        raise ValueError(
            f"before_grid shape mismatch for map {map_id}: expected {expected_h}x{expected_w}, "
            f"got {len(before_grid)}x{(len(before_grid[0]) if before_grid else 0)}"
        )
    grid = [row[:] for row in base_map.grid]
    for y, row in enumerate(before_grid):
        for x, val in enumerate(row):
            grid[y + MAP_PADDING][x + MAP_PADDING] = int(val)

    game_map = GameMap(
        map_id=base_map.map_id,
        name=base_map.name,
        ename=base_map.ename,
        width=base_map.width,
        height=base_map.height,
        grid=grid,
    )
    return GameState(
        map=game_map,
        players={
            "P1": PlayerState(
                deck_ids=list(p1_cards),
                draw_pile=[],
                hand=[create_card_from_id(cid) for cid in p1_cards],
                sp=int(p1_sp),
            ),
            "P2": PlayerState(
                deck_ids=list(p2_cards),
                draw_pile=[],
                hand=[create_card_from_id(cid) for cid in p2_cards],
                sp=int(p2_sp),
            ),
        },
        turn=int(turn),
        max_turns=12,
        mode="2P",
        pending_actions={},
        done=False,
        seed=None,
        bot_config=BotConfig(style="aggressive", level="high"),
        bot_nn_spec=None,
        log_path=None,
    )


def _extract_unpadded_grid(game_map: GameMap) -> List[List[int]]:
    return [
        [int(game_map.get_point(x + MAP_PADDING, y + MAP_PADDING)) for x in range(game_map.width - MAP_PADDING * 2)]
        for y in range(game_map.height - MAP_PADDING * 2)
    ]


def _stage_specs(
    hand: Optional[Sequence[int]],
    deck: Optional[Sequence[int]],
    sp: Optional[int],
) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    full_pool = _all_card_ids()

    hand_ids = _dedupe_keep_order(hand or [])
    deck_ids = _dedupe_keep_order(deck or [])
    deck_extra = [cid for cid in deck_ids if cid not in hand_ids]

    def add_stage(label: str, card_ids: Sequence[int], allow_normal: bool, allow_sp: bool, sp_value: int, source: str) -> None:
        ids = _dedupe_keep_order(card_ids)
        if not ids:
            return
        specs.append(
            {
                "label": label,
                "card_ids": ids,
                "allow_normal": allow_normal,
                "allow_sp": allow_sp,
                "sp_value": int(sp_value),
                "source": source,
            }
        )

    inferred_sp = max((create_card_from_id(cid).SpecialCost for cid in (hand_ids or deck_ids or full_pool)), default=10)
    sp_for_search = int(sp) if sp is not None else inferred_sp

    if hand_ids:
        add_stage("hand_normal", hand_ids, True, False, int(sp or 0), "hand")
        add_stage("hand_sp", hand_ids, False, True, sp_for_search, "hand")
    if deck_extra:
        add_stage("deck_normal", deck_extra, True, False, int(sp or 0), "deck")
        add_stage("deck_sp", deck_extra, False, True, sp_for_search, "deck")
    add_stage("all_cards_normal", full_pool, True, False, int(sp or 0), "all_cards")
    add_stage("all_cards_sp", full_pool, False, True, sp_for_search, "all_cards")
    return specs


def _enumerate_actions_for_stage(state: GameState, player: str, stage: Dict[str, object]) -> List[Action]:
    ps = state.players[player]
    is_p1 = player == "P1"
    ps.sp = int(stage["sp_value"])
    actions: List[Action] = []
    for cid in stage["card_ids"]:
        card = next((c for c in ps.hand if c.Number == int(cid)), None)
        if card is None:
            continue
        actions.append(Action(player=player, card_number=card.Number, pass_turn=True))
        for rot in (0, 1, 2, 3):
            for y in range(state.map.height):
                for x in range(state.map.width):
                    if stage["allow_normal"]:
                        ok_n, _, _ = validate_place_card_action(
                            card=card,
                            game_map=state.map,
                            anchor_x=x,
                            anchor_y=y,
                            rotation=rot,
                            is_p1=is_p1,
                            use_sp_attack=False,
                        )
                        if ok_n:
                            actions.append(
                                Action(
                                    player=player,
                                    card_number=card.Number,
                                    pass_turn=False,
                                    use_sp_attack=False,
                                    rotation=rot,
                                    x=x,
                                    y=y,
                                )
                            )
                    if stage["allow_sp"] and ps.sp >= card.SpecialCost:
                        ok_s, _, _ = validate_place_card_action(
                            card=card,
                            game_map=state.map,
                            anchor_x=x,
                            anchor_y=y,
                            rotation=rot,
                            is_p1=is_p1,
                            use_sp_attack=True,
                        )
                        if ok_s:
                            actions.append(
                                Action(
                                    player=player,
                                    card_number=card.Number,
                                    pass_turn=False,
                                    use_sp_attack=True,
                                    rotation=rot,
                                    x=x,
                                    y=y,
                                )
                            )
    return actions


def infer_actions_from_map_transition(
    map_id: str,
    before_grid: List[List[int]],
    after_grid: List[List[int]],
    p1_deck: Optional[List[int]] = None,
    p2_deck: Optional[List[int]] = None,
    p1_hand: Optional[List[int]] = None,
    p2_hand: Optional[List[int]] = None,
    p1_sp: Optional[int] = None,
    p2_sp: Optional[int] = None,
    turn: int = 1,
    max_results: int = 128,
) -> Dict[str, object]:
    """Infer legal action pairs from a before/after map transition.

    Required:
    - map_id
    - before_grid
    - after_grid

    Optional search constraints:
    - p1_deck / p2_deck
    - p1_hand / p2_hand
    - p1_sp / p2_sp

    Search order per side:
    1. hand normal
    2. hand sp
    3. deck-extra normal
    4. deck-extra sp
    5. all cards normal
    6. all cards sp
    """
    if not map_id:
        raise ValueError("map_id is required")
    if before_grid is None or after_grid is None:
        raise ValueError("before_grid and after_grid are required")

    p1_base_pool = _dedupe_keep_order((p1_hand or []) + (p1_deck or []))
    p2_base_pool = _dedupe_keep_order((p2_hand or []) + (p2_deck or []))
    if not p1_base_pool:
        p1_base_pool = _all_card_ids()
    if not p2_base_pool:
        p2_base_pool = _all_card_ids()

    state = _build_state_from_snapshot(
        map_id=map_id,
        before_grid=before_grid,
        p1_cards=p1_base_pool,
        p2_cards=p2_base_pool,
        p1_sp=int(p1_sp or 0),
        p2_sp=int(p2_sp or 0),
        turn=turn,
    )
    raw_after = _build_state_from_snapshot(
        map_id=map_id,
        before_grid=after_grid,
        p1_cards=p1_base_pool,
        p2_cards=p2_base_pool,
        p1_sp=int(p1_sp or 0),
        p2_sp=int(p2_sp or 0),
        turn=turn,
    )
    target_grid = _extract_unpadded_grid(raw_after.map)

    p1_specs = _stage_specs(p1_hand, p1_deck, p1_sp)
    p2_specs = _stage_specs(p2_hand, p2_deck, p2_sp)
    matches: List[Dict[str, object]] = []
    searched_pairs = 0

    for p1_stage in p1_specs:
        for p2_stage in p2_specs:
            trial_state = deepcopy(state)
            trial_state.players["P1"].hand = [create_card_from_id(cid) for cid in p1_stage["card_ids"]]
            trial_state.players["P2"].hand = [create_card_from_id(cid) for cid in p2_stage["card_ids"]]
            p1_actions = _enumerate_actions_for_stage(trial_state, "P1", p1_stage)
            p2_actions = _enumerate_actions_for_stage(trial_state, "P2", p2_stage)
            searched_pairs += len(p1_actions) * len(p2_actions)
            for p1_action in p1_actions:
                for p2_action in p2_actions:
                    trial = deepcopy(trial_state)
                    ok1, _, _ = step(trial, p1_action)
                    if not ok1:
                        continue
                    ok2, _, payload = step(trial, p2_action)
                    if not ok2:
                        continue
                    if _extract_unpadded_grid(trial.map) != target_grid:
                        continue
                    matches.append(
                        {
                            "p1_action": asdict(p1_action),
                            "p2_action": asdict(p2_action),
                            "p1_stage": {k: v for k, v in p1_stage.items() if k != "card_ids"},
                            "p2_stage": {k: v for k, v in p2_stage.items() if k != "card_ids"},
                            "result": payload,
                        }
                    )
                    if len(matches) >= max_results:
                        return {
                            "ok": True,
                            "truncated": True,
                            "searched_action_pairs": searched_pairs,
                            "match_count": len(matches),
                            "matches": matches,
                        }

    return {
        "ok": True,
        "truncated": False,
        "searched_action_pairs": searched_pairs,
        "match_count": len(matches),
        "matches": matches,
    }
