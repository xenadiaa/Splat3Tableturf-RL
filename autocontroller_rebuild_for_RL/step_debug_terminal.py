from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocontroller_rebuild_for_RL.runtime import (
    BIT_A,
    ControllerConfig,
    MissingInterfaceError,
    RemoteStep,
    _FrameVisionPipeline,
    _resolve_serial_port,
    choose_action_from_strategy,
    compile_action_with_defaults,
    load_config,
)
from switch_connect.virtual_gamepad.serial_controller import SerialRemoteController


BIT_NAMES: Dict[int, str] = {
    0: "A",
    1: "B",
    2: "X",
    3: "Y",
    22: "DPadUp",
    23: "DPadDown",
    24: "DPadLeft",
    25: "DPadRight",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-step terminal debugger for Tableturf auto controller")
    parser.add_argument(
        "--config",
        default="autocontroller_rebuild_for_RL/runtime_config.example.json",
        help="config json path",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 means no limit",
    )
    return parser.parse_args()


def _steps_to_debug_rows(steps: List[RemoteStep]) -> List[dict]:
    rows: List[dict] = []
    for idx, step in enumerate(steps, start=1):
        pressed = [name for bit, name in BIT_NAMES.items() if step.bits & (1 << bit)]
        rows.append(
            {
                "index": idx,
                "buttons": pressed or ["<none>"],
                "bits": step.bits,
                "hold_ms": step.hold_ms,
                "gap_ms": step.gap_ms,
            }
        )
    return rows


def _wait_for_enter(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


def _print_block(title: str, payload) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, str):
        print(payload)
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_wait_a_step(config: ControllerConfig) -> List[RemoteStep]:
    return [RemoteStep(bits=(1 << BIT_A), hold_ms=config.wait_press_hold_ms, gap_ms=config.wait_press_gap_ms)]


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    serial_port = _resolve_serial_port(config.serial_port, config.pick_serial)
    vision = _FrameVisionPipeline(config)
    controller = SerialRemoteController(port=serial_port)
    turn_index = 1
    step_count = 0

    print("单步调试终端已启动。每一步都会先展示识别结果、策略输入输出、即将发送的按键。")
    print("按回车执行本步，输入 q 后回车退出。")

    try:
        while True:
            if args.max_steps and step_count >= args.max_steps:
                print("\n达到 max-steps，停止。")
                return 0

            playable_result = vision.detect_playable()
            _print_block(
                "当前检测",
                {
                    "playable": bool(playable_result.get("playable")),
                    "sp_count": vision.last_sp_count,
                    "frame_shape": playable_result.get("frame_shape"),
                    "last_frame_path": vision.last_frame_path,
                    "last_analysis_path": vision.last_analysis_path,
                    "turn_index": turn_index,
                },
            )

            if not playable_result.get("playable"):
                steps = _build_wait_a_step(config)
                _print_block(
                    "即将执行",
                    {
                        "kind": "wait_playable_press_a_once",
                        "steps": _steps_to_debug_rows(steps),
                    },
                )
                cmd = _wait_for_enter("回车执行本步，q 退出：")
                if cmd == "q":
                    return 0
                controller.run_steps(steps)
                _print_block("本步结果", {"executed": True, "kind": "wait_playable_press_a_once"})
                cmd = _wait_for_enter("回车继续下一步，q 退出：")
                if cmd == "q":
                    return 0
                step_count += 1
                continue

            state = vision.parse_turn_state(turn_index=turn_index)
            observed_state = state.to_observed_state()
            action = choose_action_from_strategy(observed_state, config.strategy_id)
            steps = compile_action_with_defaults(action, observed_state)

            _print_block(
                "识别输入/输出",
                {
                    "map_id": state.map_id,
                    "turn": state.turn,
                    "p1_sp": state.p1_sp,
                    "hand_card_numbers": state.hand_card_numbers,
                    "selected_hand_index": state.selected_hand_index,
                    "cursor_xy": state.cursor_xy,
                    "rotation": state.rotation,
                    "card_matches": state.card_matches,
                    "last_frame_path": vision.last_frame_path,
                    "last_analysis_path": vision.last_analysis_path,
                },
            )
            _print_block("传入策略", asdict(observed_state))
            _print_block(
                "策略输出",
                {
                    "player": action.player,
                    "card_number": action.card_number,
                    "pass_turn": action.pass_turn,
                    "use_sp_attack": action.use_sp_attack,
                    "rotation": action.rotation,
                    "x": action.x,
                    "y": action.y,
                },
            )
            _print_block("即将发送手柄按键", _steps_to_debug_rows(steps))

            cmd = _wait_for_enter("回车执行本步，q 退出：")
            if cmd == "q":
                return 0

            controller.run_steps(steps)
            _print_block(
                "本步结果",
                {
                    "executed": True,
                    "turn_index": turn_index,
                    "strategy_id": config.strategy_id,
                    "action_card": action.card_number,
                    "pass_turn": action.pass_turn,
                    "use_sp_attack": action.use_sp_attack,
                },
            )

            turn_index += 1
            if turn_index > config.max_turns:
                turn_index = 1
                _print_block("对战循环", {"battle_reset": True, "next_turn_index": turn_index})

            cmd = _wait_for_enter("回车继续下一步，q 退出：")
            if cmd == "q":
                return 0
            step_count += 1
    except MissingInterfaceError as exc:
        _print_block("缺失接口", {"missing_fields": exc.missing_fields})
        return 2
    finally:
        vision.close()
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
