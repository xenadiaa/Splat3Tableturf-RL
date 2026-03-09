#!/usr/bin/env python3
"""Play 1P game with realtime terminal UI (ANSI + raw key input) and pre-game selection flow."""

from __future__ import annotations

import argparse
from collections import deque
import os
import re
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.env_core import init_state
from src.utils.deck_utils import (
    deck_display_name,
    extract_npc_strategy,
    load_deck_cards_by_rowid,
    load_npc_data,
    npc_name_zh,
)
from src.utils.player_deck_utils import (
    DECK_MAX_INDEX,
    DECK_MIN_INDEX,
    check_player_deck,
    get_player_deck_card_numbers,
    get_player_deck_name,
)
from src.view.gamepad_ui import TerminalGamepadUI


AI_TYPE_TO_BOT_STYLE = {
    "Aggressive": "aggressive",
    "Balance": "balanced",
    "AccumulateSpecial": "conservative",
}
LEVEL_TO_BOT_LEVEL = {
    0: "low",
    1: "mid",
    2: "high",
}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CSI_ARROW_RE = re.compile(r"^(?:\[|O)(?:[0-9;]*)([ABCD])")
_KEY_BUFFER: deque[str] = deque()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="1P UI play mode with keyboard mapping")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--p1-id", default="U1001")
    p.add_argument("--p1-name", default="Alice")
    p.add_argument("--line-input", action="store_true", help="Use old line-by-line input mode (debug)")
    p.add_argument("--direct", action="store_true", help="Skip selection wizard and use direct args")

    p.add_argument("--map", default="Square")
    p.add_argument("--p1-deck", type=int, default=15)
    p.add_argument("--p2-deck", type=int, default=1)
    p.add_argument("--bot-style", choices=["balanced", "aggressive", "conservative"], default="aggressive")
    p.add_argument("--bot-level", choices=["low", "mid", "high"], default="high")
    p.add_argument("--p2-id", default="BOT001")
    p.add_argument("--p2-name", default="Bot")
    return p


def _list_valid_player_decks() -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for i in range(DECK_MIN_INDEX, DECK_MAX_INDEX + 1):
        chk = check_player_deck(i, require_full_15=True)
        if chk.get("ok"):
            out.append((i, chk.get("deck_name") or get_player_deck_name(i)))
    return out


def _direct_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "map_id": args.map,
        "p1_deck_index": args.p1_deck,
        "p2_deck_ids": get_player_deck_card_numbers(args.p2_deck, require_full_15=True),
        "bot_style": args.bot_style,
        "bot_level": args.bot_level,
        "p2_id": args.p2_id,
        "p2_name": args.p2_name,
        "p2_deck_name": get_player_deck_name(args.p2_deck),
    }


@contextmanager
def _raw_mode(fileobj):
    fd = fileobj.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # Keep realtime input, but restore output newline behavior so '\n'
        # also returns carriage to column 0 (avoid diagonal/shifted layout).
        cur = termios.tcgetattr(fd)
        cur[1] |= termios.OPOST
        if hasattr(termios, "ONLCR"):
            cur[1] |= termios.ONLCR
        termios.tcsetattr(fd, termios.TCSADRAIN, cur)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_char_blocking() -> str:
    if _KEY_BUFFER:
        return _KEY_BUFFER.popleft()
    b = os.read(sys.stdin.fileno(), 1)
    if not b:
        return ""
    return b.decode(errors="ignore")


def _read_escape_tail(fd: int, max_len: int = 8, window_sec: float = 0.05) -> str:
    """Read remaining bytes of an escape sequence with a short time window."""
    out = ""
    deadline = time.monotonic() + window_sec
    while len(out) < max_len and time.monotonic() < deadline:
        ready, _, _ = select.select([sys.stdin], [], [], 0.005)
        if not ready:
            continue
        b = os.read(fd, 1)
        if not b:
            break
        out += b.decode(errors="ignore")
        # arrow sequence usually ends with A/B/C/D
        if out and out[-1] in "ABCD":
            break
    return out


def _read_key() -> str:
    ch = _read_char_blocking()
    if not ch:
        return ""
    if ch == "\x1b":
        tail = _read_escape_tail(sys.stdin.fileno())
        if tail:
            m = CSI_ARROW_RE.match(tail)
            if m:
                letter = m.group(1)
                if letter == "A":
                    return "UP"
                if letter == "B":
                    return "DOWN"
                if letter == "C":
                    return "RIGHT"
                if letter == "D":
                    return "LEFT"
            # preserve unknown tail bytes for next reads
            for c in tail:
                _KEY_BUFFER.append(c)
        return "ESC"
    if ch in ("\r", "\n"):
        return "z"
    if ch in ("i", "I"):
        return "UP"
    if ch in ("k", "K"):
        return "DOWN"
    if ch in ("j", "J"):
        return "LEFT"
    if ch in ("l", "L"):
        return "RIGHT"
    return ch


def _clear() -> None:
    sys.stdout.write("\r\033[2J\033[H")
    sys.stdout.flush()


def _menu_select_raw(title: str, items: List[str], hint: str, start_idx: int = 0) -> int:
    idx = max(0, min(start_idx, len(items) - 1)) if items else 0
    while True:
        _clear()
        print(title)
        print(hint)
        for i, item in enumerate(items):
            marker = ">" if i == idx else " "
            print(f"{marker} [{i:02d}] {item}")

        k = _read_key()
        ku = k.upper()
        if ku == "UP":
            idx = max(0, idx - 1)
        elif ku == "DOWN":
            idx = min(len(items) - 1, idx + 1)
        elif k.lower() == "z":
            return idx


def _wizard_config_raw() -> Dict[str, object]:
    npcs = load_npc_data()
    npc_labels: List[str] = []
    for n in npcs:
        en = n.get("Name", "")
        zh = npc_name_zh(en) or "未找到中文"
        npc_labels.append(f"{en} [{zh}] | order={n.get('Order')}")

    npc_idx = _menu_select_raw("[选择NPC]", npc_labels, "方向键上下移动，z确认")
    npc = npcs[npc_idx]

    strategies = sorted(extract_npc_strategy(npc), key=lambda x: x["level"])
    if not strategies:
        raise RuntimeError(f"NPC {npc.get('Name')} 没有策略数据")

    diff_labels: List[str] = []
    for s in strategies:
        diff_labels.append(
            f"level={s['level']} | ai={s['ai_type']} | map={s['map_id']} | deck={deck_display_name(s['deck_rowid'])}"
        )

    diff_idx = _menu_select_raw("[选择难度]", diff_labels, "方向键上下移动，z确认")
    strategy = strategies[diff_idx]

    valid_decks = _list_valid_player_decks()
    if not valid_decks:
        raise RuntimeError("没有可用的完整玩家牌组（0~32中无15张合法卡组）")
    deck_labels = [f"deck={idx:02d} | {name}" for idx, name in valid_decks]
    deck_pick = _menu_select_raw("[选择玩家牌组]", deck_labels, "方向键上下移动，z确认")
    p1_deck_index = valid_decks[deck_pick][0]

    p2_deck_cards = load_deck_cards_by_rowid(strategy["deck_rowid"])
    p2_deck_ids = [c["number"] for c in p2_deck_cards]

    npc_en = npc.get("Name", "NPC")
    level = int(strategy["level"])
    bot_level = LEVEL_TO_BOT_LEVEL.get(level, "high")
    bot_style = AI_TYPE_TO_BOT_STYLE.get(strategy["ai_type"], "balanced")

    return {
        "map_id": strategy["map_id"],
        "p1_deck_index": p1_deck_index,
        "p2_deck_ids": p2_deck_ids,
        "bot_style": bot_style,
        "bot_level": bot_level,
        "p2_id": f"NPC{npc.get('Order', 0)}",
        "p2_name": npc_en,
        "p2_deck_name": deck_display_name(strategy["deck_rowid"]),
    }


def _run_game_raw(args: argparse.Namespace, conf: Dict[str, object]) -> int:
    p1_idx = int(conf["p1_deck_index"])
    p1_ids = get_player_deck_card_numbers(p1_idx, require_full_15=True)
    p2_ids = list(conf["p2_deck_ids"])

    state = init_state(
        map_id=str(conf["map_id"]),
        p1_deck_ids=p1_ids,
        p2_deck_ids=p2_ids,
        seed=args.seed,
        mode="1P",
        bot_style=str(conf["bot_style"]),
        bot_level=str(conf["bot_level"]),
        p1_player_id=args.p1_id,
        p2_player_id=str(conf["p2_id"]),
        p1_player_name=args.p1_name,
        p2_player_name=str(conf["p2_name"]),
        p1_deck_name=get_player_deck_name(p1_idx),
        p2_deck_name=str(conf["p2_deck_name"]),
    )

    ui = TerminalGamepadUI(state)
    popup: str | None = None

    while True:
        _clear()
        print("键位: 方向键移动  z确认  x取消  a逆时针  s顺时针  q牌组  +=投降")
        print(ui.render())
        if popup:
            print("\n" + "-" * 32)
            print(popup)
            print("(按任意键继续)")

        if state.done and popup is None:
            popup = f"游戏结束\nwinner={state.winner}\nlog={state.log_path}"

        k = _read_key()
        if popup:
            popup = None
            if state.done:
                return 0
            continue

        out = ui.handle_key(k)
        if out:
            popup = ANSI_RE.sub("", out)


def _run_game_line_input(args: argparse.Namespace, conf: Dict[str, object]) -> int:
    p1_idx = int(conf["p1_deck_index"])
    p1_ids = get_player_deck_card_numbers(p1_idx, require_full_15=True)
    p2_ids = list(conf["p2_deck_ids"])

    state = init_state(
        map_id=str(conf["map_id"]),
        p1_deck_ids=p1_ids,
        p2_deck_ids=p2_ids,
        seed=args.seed,
        mode="1P",
        bot_style=str(conf["bot_style"]),
        bot_level=str(conf["bot_level"]),
        p1_player_id=args.p1_id,
        p2_player_id=str(conf["p2_id"]),
        p1_player_name=args.p1_name,
        p2_player_name=str(conf["p2_name"]),
        p1_deck_name=get_player_deck_name(p1_idx),
        p2_deck_name=str(conf["p2_deck_name"]),
    )

    ui = TerminalGamepadUI(state)
    print("键位: 方向键移动  z确认  x取消  a逆时针  s顺时针  q牌组  +=投降")
    while not state.done:
        print("\n" + "=" * 80)
        print(ui.render())
        raw = input("输入按键: ").strip()
        if not raw:
            continue
        out = ui.handle_key(raw)
        if out:
            print("\n" + out)
            _ = input("按回车返回对局页...")

    print("\n=== Game Over ===")
    print(ui.render())
    print(f"winner={state.winner}")
    print(f"log_path={state.log_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()

    if args.line_input:
        conf = _direct_config(args) if args.direct else _wizard_config_raw()
        return _run_game_line_input(args, conf)

    # Realtime mode: keep raw input enabled for both selection flow and in-game loop.
    with _raw_mode(sys.stdin):
        conf = _direct_config(args) if args.direct else _wizard_config_raw()
        return _run_game_raw(args, conf)


if __name__ == "__main__":
    raise SystemExit(main())
