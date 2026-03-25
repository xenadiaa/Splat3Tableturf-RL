from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocontroller_rebuild_for_RL.runtime import (
    AutoControllerRuntime,
    FrameApiAutoLauncher,
    MissingInterfaceError,
    REPO_ROOT,
    load_config,
)
from switch_connect.ui.terminal_select import choose_with_arrows
from switch_connect.virtual_gamepad.device_discovery import list_serial_port_labels, parse_device_from_label
from vision_capture.adapter import is_usb_capture_device_name, list_avfoundation_video_devices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tableturf auto controller orchestrator")
    parser.add_argument(
        "--config",
        default="autocontroller_rebuild_for_RL/runtime_config.example.json",
        help="config json path",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="play only one battle instead of looping forever",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print resolved config and exit",
    )
    parser.add_argument(
        "--target-wins",
        type=int,
        default=None,
        help="temporary target wins for this run only; overrides config and stops after reaching it",
    )
    parser.add_argument(
        "--tmp_win_target",
        action="store_true",
        help="prompt for a temporary target win count before starting; does not modify config file",
    )
    return parser.parse_args()


def _prompt_target_wins_if_needed(args: argparse.Namespace) -> int | None:
    if not bool(getattr(args, "tmp_win_target", False)):
        return None
    if args.target_wins is not None:
        return max(0, int(args.target_wins))
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        raw = input("本次临时目标胜场（直接回车表示按配置继续运行）: ").strip()
    except EOFError:
        return None
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        print(f"无效目标胜场输入：{raw}，将按配置继续运行。")
        return None


def _clear_terminal_for_runtime_ui() -> None:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    with contextlib.suppress(Exception):
        # Startup cleanup should clear both the visible screen and the scrollback
        # left by shell prompts / launch commands before the runtime UI begins.
        sys.stdout.write("\r\033[3J\033[2J\033[H")
        sys.stdout.flush()


def _load_json_obj(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_obj(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prompt_choice(options: List[str], title: str) -> str:
    if not options:
        return ""
    if sys.stdin.isatty() and sys.stdout.isatty():
        picked = choose_with_arrows(options, title)
        return str(picked or "").strip()
    return str(options[0]).strip()


def _ensure_switch_link_ready(config) -> None:
    labels = list_serial_port_labels()
    configured = str(config.serial_port or "").strip()
    if configured and any(parse_device_from_label(label) == configured for label in labels):
        config.serial_port = configured
        config.pick_serial = False
        return
    if not labels:
        raise RuntimeError("未检测到可用的 switch_link 串口，请先连接 CP2104 后再启动。")
    picked = _prompt_choice(labels, "选择可用的 switch_link 串口")
    if not picked:
        raise RuntimeError("未选择 switch_link 串口，启动已取消。")
    config.serial_port = parse_device_from_label(picked)
    config.pick_serial = False


def _usb_capture_device_names() -> List[str]:
    names = [str(name).strip() for name in list_avfoundation_video_devices()]
    names = [name for name in names if name and is_usb_capture_device_name(name)]
    seen = set()
    out: List[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _ensure_frame_api_capture_device_selected(config) -> None:
    if not str(config.frame_api_url or "").strip():
        return
    launch_config_path = Path(config.frame_api_launch_config)
    if not launch_config_path.is_absolute():
        launch_config_path = REPO_ROOT / launch_config_path
    launch_cfg = _load_json_obj(launch_config_path)
    configured_name = str(launch_cfg.get("device_name", "") or config.capture_device_name or "").strip()
    available_usb = _usb_capture_device_names()
    if configured_name and configured_name in available_usb and is_usb_capture_device_name(configured_name):
        return
    if not available_usb:
        raise RuntimeError("未检测到可用采集卡设备，已禁止回退到摄像头。")
    picked = _prompt_choice(available_usb, "选择可用的采集卡设备")
    if not picked:
        raise RuntimeError("未选择采集卡设备，启动已取消。")
    launch_cfg["device_name"] = picked
    launch_cfg["pick_device"] = False
    launch_cfg["allow_non_usb"] = False
    _write_json_obj(launch_config_path, launch_cfg)
    config.capture_device_name = picked


def _ensure_vision_ready(config) -> None:
    if not str(config.frame_api_url or "").strip():
        return
    _ensure_frame_api_capture_device_selected(config)
    launcher = FrameApiAutoLauncher(config)
    try:
        launcher.ensure_started()
        return
    except Exception:
        available_usb = _usb_capture_device_names()
        if not available_usb:
            raise RuntimeError("采集卡启动失败，且当前没有检测到可重新选择的 USB 采集卡。")
        picked = _prompt_choice(available_usb, "采集卡启动失败，请重新选择可用采集卡")
        if not picked:
            raise RuntimeError("未重新选择采集卡设备，启动已取消。")
        launch_config_path = Path(config.frame_api_launch_config)
        if not launch_config_path.is_absolute():
            launch_config_path = REPO_ROOT / launch_config_path
        launch_cfg = _load_json_obj(launch_config_path)
        launch_cfg["device_name"] = picked
        launch_cfg["pick_device"] = False
        launch_cfg["allow_non_usb"] = False
        _write_json_obj(launch_config_path, launch_cfg)
        config.capture_device_name = picked
        launcher = FrameApiAutoLauncher(config)
        launcher.ensure_started()


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    setattr(config, "_original_continuous_run", bool(config.continuous_run))
    setattr(config, "_original_target_win_count", int(config.target_win_count))
    if args.print_config:
        print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
        return 0

    _ensure_switch_link_ready(config)
    _ensure_vision_ready(config)

    override_target_wins = _prompt_target_wins_if_needed(args)
    if override_target_wins is not None:
        if int(override_target_wins) == 0:
            config.continuous_run = True
        else:
            config.continuous_run = False
            config.target_win_count = int(override_target_wins)

    _clear_terminal_for_runtime_ui()

    runtime = AutoControllerRuntime(config)
    try:
        if args.once:
            runtime.play_one_battle()
        else:
            runtime.run_forever()
        return 0
    except KeyboardInterrupt:
        return 130
    except MissingInterfaceError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "MISSING_INTERFACE_FIELDS",
                    "missing_fields": exc.missing_fields,
                    "config": str((REPO_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
