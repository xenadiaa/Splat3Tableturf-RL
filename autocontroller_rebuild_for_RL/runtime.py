from __future__ import annotations

import contextlib
import datetime as dt
import importlib
import json
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
TABLETURF_SIM_ROOT = REPO_ROOT / "tableturf_sim"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TABLETURF_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TABLETURF_SIM_ROOT))

from src.assets.tableturf_types import Map_PointBit, Map_PointMask
from switch_connect.ui.terminal_select import choose_with_arrows
from switch_connect.virtual_gamepad.device_discovery import list_serial_port_labels, parse_device_from_label
from switch_connect.virtual_gamepad.input_mapper import BIT_A, BIT_DPAD_DOWN, BIT_DPAD_LEFT, BIT_DPAD_RIGHT, BIT_DPAD_UP, BIT_X, BIT_Y, RemoteStep
from switch_connect.virtual_gamepad.serial_controller import SerialRemoteController
from tableturf_vision.hand_card_detector import SLOT_NAMES as HAND_CARD_SLOT_NAMES, detect_hand_cards
from tableturf_vision.map_state_detector import MapStateTracker, detect_map_state
from tableturf_vision.mapper_preview import _match_card
from tableturf_vision.playable_detector import detect_lose_banner, detect_playable_banner
from tableturf_vision.reference_matcher import detect_map_from_frame, load_map_info, map_name_cn_to_id
from tableturf_vision.sp_detector import get_sp_count_frame
from tableturf_vision.tableturf_mapper import _load_layout
from vision_capture.adapter import FFmpegCaptureSource, auto_detect_capture_device_name
from vision_capture.state_types import ObservedState
from src.engine.env_core import GameState, PlayerState, legal_actions
from src.engine.loaders import MAP_PADDING, load_map
from src.strategy import nn_loader
from src.strategy.registry import choose_action_from_strategy_id
from src.utils.common_utils import create_card_from_id
from src.view.gamepad_ui import _find_local_view_rightbottom_self_special_anchor


class MissingInterfaceError(RuntimeError):
    """Raised when the orchestration flow needs fields not provided by current interfaces."""

    def __init__(self, missing_fields: Sequence[str]):
        self.missing_fields = list(missing_fields)
        joined = ", ".join(self.missing_fields)
        super().__init__(f"Missing required interface fields: {joined}")


def _unique_debug_image_path(out_dir: Path, prefix: str, idx: int) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return out_dir / f"{prefix}_{ts}_{idx:05d}.png"


def _load_callable(ref: str) -> Callable[..., Any]:
    if ":" not in ref:
        raise ValueError(f"callable ref must be module:function, got: {ref}")
    module_name, func_name = ref.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"callable not found: {ref}")
    return func


def _board_label_to_mask(label: str) -> int:
    mapping = {
        "invalid": int(Map_PointMask.NotMap),
        "empty": int(Map_PointMask.Empty),
        "p1_fill": int(Map_PointMask.P1Normal),
        "p1_special": int(Map_PointMask.P1Special),
        "p1_special_activated": int(Map_PointMask.P1SpActive),
        "p2_fill": int(Map_PointMask.P2Normal),
        "p2_special": int(Map_PointMask.P2Special),
        "p2_special_activated": int(Map_PointMask.P2SpActive),
        "conflict": int(Map_PointMask.Conflict),
        "changed": int(Map_PointMask.Empty),
    }
    return mapping.get(label, int(Map_PointMask.Empty))


def _extract_board_grid(board_labels: List[List[str]]) -> List[List[int]]:
    return [[_board_label_to_mask(label) for label in row] for row in board_labels]


def _pad_board_labels_to_engine_dims(map_id: str, board_labels: List[List[str]]) -> List[List[str]]:
    game_map = load_map(map_id)
    target_h = int(game_map.height)
    target_w = int(game_map.width)
    src_h = len(board_labels)
    src_w = len(board_labels[0]) if src_h else 0
    if src_h == target_h and src_w == target_w:
        return [row[:] for row in board_labels]
    out = [["invalid" for _ in range(target_w)] for _ in range(target_h)]
    for y in range(src_h):
        for x in range(src_w):
            oy = y + MAP_PADDING
            ox = x + MAP_PADDING
            if 0 <= oy < target_h and 0 <= ox < target_w:
                out[oy][ox] = str(board_labels[y][x])
    return out


def _pad_grid_to_engine_dims(map_id: str, grid: List[List[int]]) -> List[List[int]]:
    game_map = load_map(map_id)
    target_h = int(game_map.height)
    target_w = int(game_map.width)
    src_h = len(grid)
    src_w = len(grid[0]) if src_h else 0
    if src_h == target_h and src_w == target_w:
        return [row[:] for row in grid]
    pad = MAP_PADDING
    out = [[int(Map_PointMask.NotMap) for _ in range(target_w)] for _ in range(target_h)]
    for y in range(src_h):
        for x in range(src_w):
            oy = y + pad
            ox = x + pad
            if 0 <= oy < target_h and 0 <= ox < target_w:
                out[oy][ox] = int(grid[y][x])
    return out


def _slot_rank(slot_name: str) -> int:
    try:
        return HAND_CARD_SLOT_NAMES.index(slot_name)
    except ValueError:
        return len(HAND_CARD_SLOT_NAMES)


def _map_info_by_id(map_id: str) -> Dict[str, Any]:
    for row in load_map_info():
        if str(row.get("id", "")) == str(map_id):
            return row
    raise ValueError(f"map info not found for map_id={map_id}")


def _map_state_to_board_labels(map_id: str, map_state_result: Dict[str, Any]) -> List[List[str]]:
    map_info = _map_info_by_id(map_id)
    point_type = map_info["point_type"]
    labels = [["invalid" if int(cell) == 0 else "transparent" for cell in row] for row in point_type]
    for cell in map_state_result.get("cells", []):
        row = int(cell["json_row"])
        col = int(cell["json_col"])
        labels[row][col] = str(cell["label"])
    return labels


def _pad_map_match_payload(map_id: str, map_match: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(map_match)
    template_board = out.get("board_from_template")
    if isinstance(template_board, dict):
        labels = template_board.get("labels")
        if isinstance(labels, list):
            out["board_from_template"] = {
                **template_board,
                "labels": _pad_board_labels_to_engine_dims(map_id, labels),
            }
    return out


def _normalize_hand_slots(hand_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    slots = list(hand_result.get("slots", []))
    slots.sort(key=lambda slot: _slot_rank(str(slot.get("slot", ""))))
    return slots


def _resolve_serial_port(configured_port: str, pick_serial: bool) -> str:
    if configured_port.strip():
        return configured_port.strip()
    if not pick_serial:
        labels = list_serial_port_labels()
        if len(labels) == 1:
            return parse_device_from_label(labels[0])
    labels = list_serial_port_labels()
    if not labels:
        raise RuntimeError("No serial ports found for virtual gamepad")
    picked = choose_with_arrows(labels, "Select virtual gamepad serial port")
    if not picked:
        raise RuntimeError("Serial port selection cancelled")
    return parse_device_from_label(picked)


@dataclass
class ManualVisionFields:
    selected_hand_index: Optional[int] = None
    cursor_xy: Optional[Tuple[int, int]] = None
    rotation: Optional[int] = None
    p1_sp: Optional[int] = None


@dataclass
class SupplementalState:
    selected_hand_index: Optional[int] = None
    cursor_xy: Optional[Tuple[int, int]] = None
    rotation: Optional[int] = None
    p1_sp: Optional[int] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SupplementalState":
        cursor_xy = payload.get("cursor_xy")
        cursor_tuple = tuple(cursor_xy) if isinstance(cursor_xy, (list, tuple)) and len(cursor_xy) == 2 else None
        return cls(
            selected_hand_index=payload.get("selected_hand_index"),
            cursor_xy=cursor_tuple,
            rotation=payload.get("rotation"),
            p1_sp=payload.get("p1_sp"),
        )


@dataclass
class ControllerConfig:
    frame_api_url: str = ""
    frame_api_auto_start: bool = True
    frame_api_health_url: str = ""
    frame_api_launch_script: str = "vision_capture/preview_stream_opencv.py"
    frame_api_launch_config: str = "vision_capture/capture_config.json"
    frame_api_startup_seconds: float = 15.0
    capture_device_name: str = ""
    capture_width: int = 1920
    capture_height: int = 1080
    capture_fps: int = 30
    capture_pixel_format: str = ""
    capture_read_timeout_seconds: float = 5.0
    capture_drain_ms: int = 60
    serial_port: str = ""
    pick_serial: bool = True
    wait_press_hold_ms: int = 110
    wait_press_gap_ms: int = 1000
    playable_poll_seconds: float = 0.35
    max_turns: int = 12
    continuous_run: bool = True
    target_win_count: int = 1
    progress_timeout_seconds: float = 180.0
    strategy_id: str = "default:aggressive:high"
    strategy_id_by_map: Dict[str, str] = field(default_factory=dict)
    strategy_id_by_map_name: Dict[str, str] = field(default_factory=dict)
    policy_config_json: str = "autocontroller_rebuild_for_RL/strategy_policy.example.json"
    strict_missing_interfaces: bool = True
    layout_json: str = "tableturf_vision/tableturf_layout.json"
    manual_fields: ManualVisionFields = field(default_factory=ManualVisionFields)
    supplemental_state_provider: str = ""
    debug_ui_enabled: bool = True
    save_debug_frames: bool = True
    debug_frame_dir: str = "autocontroller_rebuild_for_RL/debug_runtime"
    log_file: str = "autocontroller_rebuild_for_RL/debug_runtime/autocontroller.log"

    @classmethod
    def from_json(cls, path: Path) -> "ControllerConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        manual_fields = ManualVisionFields(**data.get("manual_fields", {}))
        payload = dict(data)
        payload["manual_fields"] = manual_fields
        return cls(**payload)


@dataclass
class ParsedTurnState:
    map_id: str
    map_name: str
    hand_card_numbers: List[int]
    map_grid: List[List[int]]
    playable_result: Dict[str, Any]
    analysis_result: Dict[str, Any]
    card_matches: List[Dict[str, Any]]
    turn: int
    selected_hand_index: Optional[int]
    cursor_xy: Optional[Tuple[int, int]]
    rotation: int
    p1_sp: int

    def to_observed_state(self) -> ObservedState:
        return ObservedState(
            map_id=self.map_id,
            hand_card_numbers=self.hand_card_numbers,
            p1_sp=self.p1_sp,
            turn=self.turn,
            map_grid=self.map_grid,
            selected_hand_index=self.selected_hand_index,
            cursor_xy=self.cursor_xy,
            rotation=self.rotation,
        )


@dataclass
class ResolvedStrategy:
    mode: str
    label: str
    strategy_id: str = ""
    checkpoint_file: str = ""
    source: str = ""


class RuntimeLogWriter:
    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, message: str) -> None:
        line = f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}\n"
        with self._lock:
            self._path.open("a", encoding="utf-8").write(line)


_STRATEGIC_MODEL_CACHE: Dict[str, Any] = {}
_BASE_MODEL_CACHE: Dict[str, Any] = {}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            "__type__": value.__class__.__name__,
            **{str(k): _json_safe(v) for k, v in vars(value).items()},
        }
    return repr(value)


class HttpJpegCaptureSource:
    def __init__(self, frame_api_url: str):
        self.frame_api_url = frame_api_url
        self.last_error: Optional[str] = None
        self._cache_path = Path(tempfile.gettempdir()) / "tableturf_http_frame_latest.jpg"

    def read_latest(self, timeout_seconds: float = 5.0, drain_ms: int = 0) -> Optional[np.ndarray]:
        del drain_ms
        req = urllib.request.Request(self.frame_api_url, headers={"Cache-Control": "no-cache"})
        try:
            with urllib.request.urlopen(req, timeout=max(0.5, timeout_seconds)) as resp:
                payload = resp.read()
        except Exception as exc:
            self.last_error = f"HTTP_FRAME_FETCH_FAILED:{exc}"
            return None
        try:
            self._cache_path.write_bytes(payload)
        except Exception as exc:
            self.last_error = f"HTTP_FRAME_SAVE_FAILED:{exc}"
            return None
        frame = cv2.imread(str(self._cache_path), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            self.last_error = "HTTP_FRAME_DECODE_FAILED"
            return None
        self.last_error = None
        return frame

    def read_with_fallbacks(
        self,
        timeout_seconds: float = 5.0,
        fallback_specs: Optional[List[Dict[str, object]]] = None,
    ) -> Optional[np.ndarray]:
        del fallback_specs
        return self.read_latest(timeout_seconds=timeout_seconds, drain_ms=0)

    def stop(self) -> None:
        return


class FrameApiAutoLauncher:
    def __init__(self, config: ControllerConfig):
        self._config = config
        self._proc: Optional[subprocess.Popen] = None

    def _health_url(self) -> str:
        explicit = str(self._config.frame_api_health_url or "").strip()
        if explicit:
            return explicit
        base = str(self._config.frame_api_url or "").strip()
        if base.endswith("/frame.jpg"):
            return base[:-10] + "/health"
        if base.endswith("/frame.jpeg"):
            return base[:-11] + "/health"
        return base.rstrip("/") + "/health"

    def is_ready(self, timeout_seconds: float = 1.0) -> bool:
        health_url = self._health_url()
        if not health_url:
            return False
        req = urllib.request.Request(health_url, headers={"Cache-Control": "no-cache"})
        try:
            with urllib.request.urlopen(req, timeout=max(0.3, timeout_seconds)) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            return bool(payload.get("has_frame"))
        except Exception:
            return False

    def ensure_started(self) -> None:
        if not str(self._config.frame_api_url or "").strip():
            return
        if self.is_ready(timeout_seconds=0.8):
            return
        if not self._config.frame_api_auto_start:
            raise RuntimeError("FRAME_API_NOT_READY_AND_AUTO_START_DISABLED")

        script_path = _resolve_repo_path(self._config.frame_api_launch_script)
        launch_config = _resolve_repo_path(self._config.frame_api_launch_config)
        cmd = [sys.executable, str(script_path), "--config", str(launch_config)]
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + max(1.0, float(self._config.frame_api_startup_seconds))
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("FRAME_API_PROCESS_EXITED_EARLY")
            if self.is_ready(timeout_seconds=0.8):
                return
            time.sleep(0.2)
        raise RuntimeError("FRAME_API_START_TIMEOUT")

    def stop(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        if proc.poll() is not None:
            return
        with contextlib.suppress(Exception):
            proc.terminate()
        try:
            proc.wait(timeout=1.5)
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"json root must be object: {path}")
    return data


def _latest_checkpoint_in_dir(dir_path: Path) -> Path:
    cands = sorted(dir_path.glob("ppo_tableturf_u*.pt"))
    if not cands:
        raise FileNotFoundError(f"no PPO checkpoint found in {dir_path}")
    return cands[-1]


def _resolve_checkpoint_from_entry(entry: Dict[str, Any]) -> Tuple[Path, str]:
    for key in ("checkpoint_file", "checkpoint", "pt"):
        value = str(entry.get(key, "")).strip()
        if value:
            resolved = _resolve_repo_path(value)
            if not resolved.exists():
                raise FileNotFoundError(f"checkpoint file not found: {resolved}")
            return resolved, key

    for key in ("training_summary", "eval_summary"):
        value = str(entry.get(key, "")).strip()
        if not value:
            continue
        summary_path = _resolve_repo_path(value)
        if not summary_path.exists():
            raise FileNotFoundError(f"summary file not found: {summary_path}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        save_dir = str(payload.get("save_dir", "")).strip()
        if save_dir:
            return _latest_checkpoint_in_dir(_resolve_repo_path(save_dir)), key
        return _latest_checkpoint_in_dir(summary_path.parent), key

    for key in ("checkpoint_dir", "save_dir", "dir"):
        value = str(entry.get(key, "")).strip()
        if value:
            resolved_dir = _resolve_repo_path(value)
            if not resolved_dir.exists():
                raise FileNotFoundError(f"checkpoint dir not found: {resolved_dir}")
            return _latest_checkpoint_in_dir(resolved_dir), key

    raise ValueError(f"ppo entry missing checkpoint path fields: {entry}")


def _resolve_policy_entry(raw: Any, source: str) -> ResolvedStrategy:
    if isinstance(raw, str):
        return ResolvedStrategy(mode="strategy_id", label=raw, strategy_id=raw, source=source)
    if not isinstance(raw, dict):
        raise ValueError(f"invalid policy entry from {source}: {raw!r}")

    mode = str(raw.get("mode", "") or raw.get("type", "") or "").strip().lower()
    strategy_id = str(raw.get("strategy_id", "")).strip()
    fallback_strategy_id = str(raw.get("fallback_strategy_id", "")).strip()
    fallback_label = str(raw.get("fallback_label", "")).strip()
    if strategy_id:
        label = str(raw.get("label", "") or strategy_id)
        return ResolvedStrategy(mode="strategy_id", label=label, strategy_id=strategy_id, source=source)

    if mode in {"ppo", "checkpoint", "nn", "strategic_ppo", "strategic", "strategic_checkpoint"} or any(
        str(raw.get(k, "")).strip() for k in ("checkpoint_file", "checkpoint", "pt", "training_summary", "eval_summary", "checkpoint_dir", "save_dir", "dir")
    ):
        try:
            checkpoint_file, path_kind = _resolve_checkpoint_from_entry(raw)
        except FileNotFoundError:
            if fallback_strategy_id:
                return ResolvedStrategy(
                    mode="strategy_id",
                    label=fallback_label or fallback_strategy_id,
                    strategy_id=fallback_strategy_id,
                    source=f"{source}:fallback_missing_checkpoint",
                )
            raise
        label = str(raw.get("label", "") or f"ppo:{checkpoint_file.name}")
        resolved_mode = "ppo_checkpoint"
        checkpoint_name = checkpoint_file.name.lower()
        if mode in {"strategic_ppo", "strategic", "strategic_checkpoint"} or checkpoint_name.startswith("strategic_ppo_"):
            resolved_mode = "strategic_ppo_checkpoint"
        return ResolvedStrategy(
            mode=resolved_mode,
            label=label,
            checkpoint_file=str(checkpoint_file),
            source=f"{source}:{path_kind}",
        )

    raise ValueError(f"unsupported policy entry from {source}: {raw}")


def _load_policy_config(config: ControllerConfig) -> Dict[str, Any]:
    path = str(config.policy_config_json or "").strip()
    if not path:
        return {}
    return _load_json_if_exists(_resolve_repo_path(path))


class ASpamWorker:
    def __init__(self, controller: SerialRemoteController, hold_ms: int, gap_ms: int):
        self._controller = controller
        self._hold_ms = hold_ms
        self._gap_ms = gap_ms
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="a-spam-worker", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        step = RemoteStep(bits=(1 << BIT_A), hold_ms=self._hold_ms, gap_ms=self._gap_ms)
        while not self._stop.is_set():
            self._controller.run_steps([step])

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _press_button(steps: List[RemoteStep], bit_index: int, hold_ms: int = 70, gap_ms: int = 130) -> None:
    steps.append(RemoteStep(bits=(1 << bit_index), hold_ms=hold_ms, gap_ms=gap_ms))


BIT_PLUS = 9
BIT_HOME = 12


def _move_axis(steps: List[RemoteStep], dx: int, dy: int, move_hold_ms: int = 70) -> None:
    if dx > 0:
        for _ in range(dx):
            _press_button(steps, BIT_DPAD_RIGHT, hold_ms=move_hold_ms, gap_ms=130)
    elif dx < 0:
        for _ in range(-dx):
            _press_button(steps, BIT_DPAD_LEFT, hold_ms=move_hold_ms, gap_ms=130)
    if dy > 0:
        for _ in range(dy):
            _press_button(steps, BIT_DPAD_DOWN, hold_ms=move_hold_ms, gap_ms=130)
    elif dy < 0:
        for _ in range(-dy):
            _press_button(steps, BIT_DPAD_UP, hold_ms=move_hold_ms, gap_ms=130)


def _card_grid_xy(index: int) -> Tuple[int, int]:
    idx = max(0, int(index))
    return (idx % 2, idx // 2)


def _move_card_selection(steps: List[RemoteStep], from_index: int, to_index: int) -> None:
    from_x, from_y = _card_grid_xy(from_index)
    to_x, to_y = _card_grid_xy(to_index)
    _move_axis(steps, dx=to_x - from_x, dy=to_y - from_y)


def _build_state_from_observation(obs: ObservedState) -> GameState:
    game_map = load_map(obs.map_id)
    if obs.map_grid is not None:
        game_map.grid = _pad_grid_to_engine_dims(obs.map_id, obs.map_grid)
    p1_hand = [create_card_from_id(n) for n in obs.hand_card_numbers]
    p1 = PlayerState(deck_ids=[], draw_pile=[], hand=p1_hand, sp=obs.p1_sp)
    p2 = PlayerState(deck_ids=[], draw_pile=[], hand=[], sp=0)
    return GameState(map=game_map, players={"P1": p1, "P2": p2}, turn=obs.turn)


def _initial_engine_anchor_for_map(obs: ObservedState) -> Tuple[int, int]:
    game_map = load_map(obs.map_id)
    pad = MAP_PADDING
    if game_map.width > pad * 2 and game_map.height > pad * 2:
        logical_w = game_map.width - pad * 2
        logical_h = game_map.height - pad * 2
        view_x0 = pad
        view_y0 = pad
    else:
        logical_w = game_map.width
        logical_h = game_map.height
        view_x0 = 0
        view_y0 = 0
    anchor_local = _find_local_view_rightbottom_self_special_anchor(
        game_map=game_map,
        is_p1=True,
        view_x0=view_x0,
        view_y0=view_y0,
        view_w=logical_w,
        view_h=logical_h,
        flip_180=False,
    )
    return (int(anchor_local[0] + view_x0), int(anchor_local[1] + view_y0))


def _engine_xy_to_ui_xy(x: int, y: int, map_id: str, map_grid: Optional[List[List[int]]] = None) -> Tuple[int, int]:
    game_map = load_map(map_id)
    target_h = int(game_map.height)
    target_w = int(game_map.width)
    raw_h = len(map_grid) if map_grid else max(0, target_h - MAP_PADDING * 2)
    raw_w = len(map_grid[0]) if map_grid and map_grid[0] else max(0, target_w - MAP_PADDING * 2)
    if target_h == raw_h and target_w == raw_w:
        return (int(x), int(y))
    return (int(x) - MAP_PADDING, int(y) - MAP_PADDING)


def _initial_ui_anchor_for_map(obs: ObservedState) -> Tuple[int, int]:
    anchor_x, anchor_y = _initial_engine_anchor_for_map(obs)
    return _engine_xy_to_ui_xy(anchor_x, anchor_y, obs.map_id, obs.map_grid)


def _action_target_ui_xy(action: Any, obs: ObservedState) -> Tuple[int, int]:
    if action.x is None or action.y is None:
        raise ValueError("non-pass action requires x/y")
    return _engine_xy_to_ui_xy(int(action.x), int(action.y), obs.map_id, obs.map_grid)


def choose_action_from_strategy(obs: ObservedState, strategy_id: str):
    state = _build_state_from_observation(obs)
    return choose_action_from_strategy_id(state=state, player="P1", strategy_id=strategy_id)


def _base_load_model(checkpoint_file: str):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torch import failed: {exc}") from exc

    ckpt = str(checkpoint_file)
    if ckpt in _BASE_MODEL_CACHE:
        return _BASE_MODEL_CACHE[ckpt], torch

    from gamestrategy_RL.networks import PolicyValueNet

    model = PolicyValueNet(map_channels=6, scalar_dim=6, action_feature_dim=12)
    obj = torch.load(ckpt, map_location="cpu")
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not isinstance(state_dict, dict):
        raise RuntimeError("invalid checkpoint content")
    model.load_state_dict(state_dict)
    model.eval()
    _BASE_MODEL_CACHE[ckpt] = model
    return model, torch


def _base_encode_state(state: GameState, player: str) -> Tuple[np.ndarray, np.ndarray]:
    game_map = state.map
    obs = np.zeros((6, game_map.height, game_map.width), dtype=np.float32)
    is_p1_agent = player == "P1"
    for y in range(game_map.height):
        for x in range(game_map.width):
            mask = int(game_map.get_point(x, y))
            is_p1 = (mask & int(Map_PointBit.IsP1)) != 0
            is_p2 = (mask & int(Map_PointBit.IsP2)) != 0
            obs[0, y, x] = 1.0 if (mask & int(Map_PointBit.IsValid)) else 0.0
            obs[1, y, x] = 1.0 if (is_p1 if is_p1_agent else is_p2) else 0.0
            obs[2, y, x] = 1.0 if (is_p2 if is_p1_agent else is_p1) else 0.0
            obs[3, y, x] = 1.0 if (mask & int(Map_PointBit.IsSp)) else 0.0
            obs[4, y, x] = 1.0 if (mask & int(Map_PointBit.IsSupplySp)) else 0.0
            obs[5, y, x] = 1.0 if (is_p1 and is_p2) else 0.0

    p1_score, p2_score = _strategic_compute_scores(state)
    own = state.players[player]
    opp = state.players["P2" if player == "P1" else "P1"]
    own_score = p1_score if is_p1_agent else p2_score
    opp_score = p2_score if is_p1_agent else p1_score
    scalar = np.array(
        [
            state.turn / max(1, state.max_turns),
            own.sp / 20.0,
            opp.sp / 20.0,
            (own_score - opp_score) / 100.0,
            len(own.draw_pile) / 15.0,
            len(opp.draw_pile) / 15.0,
        ],
        dtype=np.float32,
    )
    return obs, scalar


def _base_encode_actions(state: GameState, player: str, legal_action_dicts: List[Dict[str, Any]]) -> np.ndarray:
    ps = state.players[player]
    hand = ps.hand
    hand_index = {card.Number: idx for idx, card in enumerate(hand)}
    w = max(1, state.map.width - 1)
    h = max(1, state.map.height - 1)
    feats: List[np.ndarray] = []
    for action in legal_action_dicts:
        card_no = int(action.get("card_number"))
        card = next((c for c in hand if c.Number == card_no), None)
        if card is None:
            raise RuntimeError(f"card #{card_no} not in hand")
        rotation = int(action.get("rotation", 0))
        x = action.get("x")
        y = action.get("y")
        cell_count, sp_count = nn_loader._card_cell_stats(card, rotation)
        feats.append(
            np.array(
                [
                    hand_index[card.Number] / 3.0,
                    1.0 if bool(action.get("pass_turn", False)) else 0.0,
                    1.0 if bool(action.get("use_sp_attack", False)) else 0.0,
                    rotation / 3.0,
                    (float(x) / w) if x is not None else 0.0,
                    (float(y) / h) if y is not None else 0.0,
                    card.CardPoint / 20.0,
                    card.SpecialCost / 10.0,
                    cell_count / 64.0,
                    sp_count / 64.0,
                    ps.sp / 20.0,
                    state.turn / max(1, state.max_turns),
                ],
                dtype=np.float32,
            )
        )
    if not feats:
        return np.zeros((1, 12), dtype=np.float32)
    return np.stack(feats, axis=0)


def _choose_action_from_base_checkpoint(state: GameState, checkpoint_file: str, player: str = "P1") -> Dict[str, Any]:
    legal = legal_actions(state, player)
    if not legal:
        raise RuntimeError("legal_actions is empty")
    legal_action_dicts = [{
        "player": a.player,
        "card_number": a.card_number,
        "surrender": a.surrender,
        "pass_turn": a.pass_turn,
        "use_sp_attack": a.use_sp_attack,
        "rotation": a.rotation,
        "x": a.x,
        "y": a.y,
    } for a in legal]
    model, torch = _base_load_model(checkpoint_file)
    map_obs, scalar_obs = _base_encode_state(state, player)
    action_feats = _base_encode_actions(state, player, legal_action_dicts)
    with torch.no_grad():
        logits, _ = model.forward_single(
            torch.as_tensor(map_obs, dtype=torch.float32),
            torch.as_tensor(scalar_obs, dtype=torch.float32),
            torch.as_tensor(action_feats, dtype=torch.float32),
        )
        idx = int(torch.argmax(logits).item())
    idx = max(0, min(idx, len(legal_action_dicts) - 1))
    return dict(legal_action_dicts[idx])


def _strategic_load_model(checkpoint_file: str):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torch import failed: {exc}") from exc

    ckpt = str(checkpoint_file)
    if ckpt in _STRATEGIC_MODEL_CACHE:
        return _STRATEGIC_MODEL_CACHE[ckpt], torch

    from gamestrategy_RL.strategic_networks import StrategicPolicyValueNet

    model = StrategicPolicyValueNet(map_channels=6, scalar_dim=14, action_feature_dim=12)
    obj = torch.load(ckpt, map_location="cpu")
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not isinstance(state_dict, dict):
        raise RuntimeError("invalid strategic checkpoint content")
    model.load_state_dict(state_dict)
    model.eval()
    _STRATEGIC_MODEL_CACHE[ckpt] = model
    return model, torch


def _strategic_is_valid(mask: int) -> bool:
    return (mask & int(Map_PointBit.IsValid)) != 0


def _strategic_has_owner(mask: int, player: str) -> bool:
    bit = Map_PointBit.IsP1 if player == "P1" else Map_PointBit.IsP2
    return (mask & int(bit)) != 0


def _strategic_is_empty(mask: int) -> bool:
    if not _strategic_is_valid(mask):
        return False
    return not _strategic_has_owner(mask, "P1") and not _strategic_has_owner(mask, "P2")


def _strategic_is_sp(mask: int) -> bool:
    return (mask & int(Map_PointBit.IsSp)) != 0


def _iter_state_cells(state: GameState) -> Iterable[Tuple[int, int, int]]:
    game_map = state.map
    for y in range(game_map.height):
        for x in range(game_map.width):
            yield x, y, int(game_map.get_point(x, y))


def _strategic_neighbors4(x: int, y: int) -> Iterable[Tuple[int, int]]:
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _strategic_compute_scores(state: GameState) -> Tuple[int, int]:
    p1 = 0
    p2 = 0
    for _, _, mask in _iter_state_cells(state):
        if not _strategic_is_valid(mask):
            continue
        is_p1 = _strategic_has_owner(mask, "P1")
        is_p2 = _strategic_has_owner(mask, "P2")
        if is_p1 and not is_p2:
            p1 += 1
        elif is_p2 and not is_p1:
            p2 += 1
    return p1, p2


def _strategic_reachable_empty_stats(state: GameState, player: str) -> Tuple[int, int, int]:
    reachable_total = 0
    largest_reachable = 0
    largest_locked = 0
    visited: Set[Tuple[int, int]] = set()
    game_map = state.map

    for y in range(game_map.height):
        for x in range(game_map.width):
            if (x, y) in visited:
                continue
            mask = int(game_map.get_point(x, y))
            if not _strategic_is_empty(mask):
                continue
            queue = deque([(x, y)])
            visited.add((x, y))
            size = 0
            touches_player = False
            while queue:
                cx, cy = queue.popleft()
                size += 1
                for nx, ny in _strategic_neighbors4(cx, cy):
                    if nx < 0 or ny < 0 or nx >= game_map.width or ny >= game_map.height:
                        continue
                    nm = int(game_map.get_point(nx, ny))
                    if _strategic_has_owner(nm, player):
                        touches_player = True
                    if (nx, ny) in visited or not _strategic_is_empty(nm):
                        continue
                    visited.add((nx, ny))
                    queue.append((nx, ny))
            if touches_player:
                reachable_total += size
                largest_reachable = max(largest_reachable, size)
            else:
                largest_locked = max(largest_locked, size)
    return reachable_total, largest_reachable, largest_locked


def _strategic_frontier_count(state: GameState, player: str) -> int:
    other = "P2" if player == "P1" else "P1"
    count = 0
    for x, y, mask in _iter_state_cells(state):
        if not _strategic_has_owner(mask, player):
            continue
        for nx, ny in _strategic_neighbors4(x, y):
            if nx < 0 or ny < 0 or nx >= state.map.width or ny >= state.map.height:
                continue
            nm = int(state.map.get_point(nx, ny))
            if _strategic_is_empty(nm) or _strategic_has_owner(nm, other):
                count += 1
                break
    return count


def _strategic_is_frontier_cell(state: GameState, x: int, y: int, player: str) -> bool:
    other = "P2" if player == "P1" else "P1"
    for nx, ny in _strategic_neighbors4(x, y):
        if nx < 0 or ny < 0 or nx >= state.map.width or ny >= state.map.height:
            continue
        nm = int(state.map.get_point(nx, ny))
        if _strategic_is_empty(nm) or _strategic_has_owner(nm, other):
            return True
    return False


def _strategic_sp_breach_risk(state: GameState, attacker: str, defender: str) -> float:
    attacker_sp = float(state.players[attacker].sp)
    if attacker_sp < 3:
        return 0.0
    attack_scale = 2.0 if attacker_sp >= 6 else 1.0
    risk = 0.0
    for x, y, mask in _iter_state_cells(state):
        if not _strategic_has_owner(mask, defender):
            continue
        if not _strategic_is_frontier_cell(state, x, y, defender):
            continue
        adjacent_enemy_sp = False
        for nx, ny in _strategic_neighbors4(x, y):
            if nx < 0 or ny < 0 or nx >= state.map.width or ny >= state.map.height:
                continue
            nm = int(state.map.get_point(nx, ny))
            if _strategic_has_owner(nm, attacker) and _strategic_is_sp(nm):
                adjacent_enemy_sp = True
                break
        if not adjacent_enemy_sp:
            continue
        support = 0
        for nx, ny in _strategic_neighbors4(x, y):
            if nx < 0 or ny < 0 or nx >= state.map.width or ny >= state.map.height:
                continue
            nm = int(state.map.get_point(nx, ny))
            if _strategic_has_owner(nm, defender):
                support += 1
        if support <= 1:
            risk += 1.5 * attack_scale
        elif support == 2:
            risk += 0.8 * attack_scale
        else:
            risk += 0.25 * attack_scale
    return risk


def _strategic_compute_metrics(state: GameState) -> Dict[str, float]:
    valid_cells = 0
    empty_cells = 0
    for _, _, mask in _iter_state_cells(state):
        if not _strategic_is_valid(mask):
            continue
        valid_cells += 1
        if _strategic_is_empty(mask):
            empty_cells += 1
    p1_reach, p1_largest, p1_locked = _strategic_reachable_empty_stats(state, "P1")
    p2_reach, p2_largest, p2_locked = _strategic_reachable_empty_stats(state, "P2")
    p1_score, p2_score = _strategic_compute_scores(state)
    return {
        "valid_cells": float(valid_cells),
        "empty_cells": float(empty_cells),
        "turn_ratio": state.turn / max(1, state.max_turns),
        "p1_sp": float(state.players["P1"].sp),
        "p2_sp": float(state.players["P2"].sp),
        "p1_score": float(p1_score),
        "p2_score": float(p2_score),
        "score_diff": float(p1_score - p2_score),
        "p1_reachable_empty": float(p1_reach),
        "p2_reachable_empty": float(p2_reach),
        "p1_largest_reachable": float(p1_largest),
        "p2_largest_reachable": float(p2_largest),
        "p1_largest_locked": float(p1_locked),
        "p2_largest_locked": float(p2_locked),
        "p1_frontier": float(_strategic_frontier_count(state, "P1")),
        "p2_frontier": float(_strategic_frontier_count(state, "P2")),
        "enemy_breach_risk": float(_strategic_sp_breach_risk(state, attacker="P2", defender="P1")),
        "own_breach_chance": float(_strategic_sp_breach_risk(state, attacker="P1", defender="P2")),
        "p1_draw_ratio": len(state.players["P1"].draw_pile) / 15.0,
        "p2_draw_ratio": len(state.players["P2"].draw_pile) / 15.0,
    }


def _strategic_encode_state(state: GameState, player: str) -> Tuple[np.ndarray, np.ndarray]:
    game_map = state.map
    obs = np.zeros((6, game_map.height, game_map.width), dtype=np.float32)
    metrics = _strategic_compute_metrics(state)
    is_p1_agent = player == "P1"
    for y in range(game_map.height):
        for x in range(game_map.width):
            mask = int(game_map.get_point(x, y))
            is_p1 = (mask & int(Map_PointBit.IsP1)) != 0
            is_p2 = (mask & int(Map_PointBit.IsP2)) != 0
            obs[0, y, x] = 1.0 if _strategic_is_valid(mask) else 0.0
            obs[1, y, x] = 1.0 if (is_p1 if is_p1_agent else is_p2) else 0.0
            obs[2, y, x] = 1.0 if (is_p2 if is_p1_agent else is_p1) else 0.0
            obs[3, y, x] = 1.0 if (mask & int(Map_PointBit.IsSp)) else 0.0
            obs[4, y, x] = 1.0 if (mask & int(Map_PointBit.IsSupplySp)) else 0.0
            obs[5, y, x] = 1.0 if (is_p1 and is_p2) else 0.0

    valid = max(1.0, metrics["valid_cells"])
    scalar = np.array(
        [
            metrics["turn_ratio"],
            metrics["p1_sp"] / 20.0,
            metrics["p2_sp"] / 20.0,
            metrics["score_diff"] / 100.0,
            metrics["p1_draw_ratio"],
            metrics["p2_draw_ratio"],
            metrics["p1_reachable_empty"] / valid,
            metrics["p2_reachable_empty"] / valid,
            metrics["p1_largest_reachable"] / valid,
            metrics["p2_largest_reachable"] / valid,
            metrics["enemy_breach_risk"] / valid,
            metrics["own_breach_chance"] / valid,
            metrics["p1_largest_locked"] / valid,
            metrics["p2_largest_locked"] / valid,
        ],
        dtype=np.float32,
    )
    return obs, scalar


def _strategic_encode_actions(state: GameState, player: str, legal_action_dicts: List[Dict[str, Any]]) -> np.ndarray:
    ps = state.players[player]
    hand = ps.hand
    hand_index = {card.Number: idx for idx, card in enumerate(hand)}
    feats: List[np.ndarray] = []
    for action in legal_action_dicts:
        card_no = int(action.get("card_number"))
        card = next((c for c in hand if c.Number == card_no), None)
        if card is None:
            raise RuntimeError(f"card #{card_no} not in hand")
        rotation = int(action.get("rotation", 0))
        x = action.get("x")
        y = action.get("y")
        cell_count, sp_count = nn_loader._card_cell_stats(card, rotation)
        feats.append(
            np.array(
                [
                    hand_index[card.Number] / 3.0,
                    1.0 if bool(action.get("pass_turn", False)) else 0.0,
                    1.0 if bool(action.get("use_sp_attack", False)) else 0.0,
                    rotation / 3.0,
                    (float(x) / max(1, state.map.width - 1)) if x is not None else 0.0,
                    (float(y) / max(1, state.map.height - 1)) if y is not None else 0.0,
                    card.CardPoint / 20.0,
                    card.SpecialCost / 10.0,
                    cell_count / 64.0,
                    sp_count / 64.0,
                    ps.sp / 20.0,
                    state.turn / max(1, state.max_turns),
                ],
                dtype=np.float32,
            )
        )
    if not feats:
        return np.zeros((1, 12), dtype=np.float32)
    return np.stack(feats, axis=0)


def _choose_action_from_strategic_checkpoint(state: GameState, checkpoint_file: str, player: str = "P1") -> Dict[str, Any]:
    legal = legal_actions(state, player)
    if not legal:
        raise RuntimeError("legal_actions is empty")
    legal_action_dicts = [{
        "player": a.player,
        "card_number": a.card_number,
        "surrender": a.surrender,
        "pass_turn": a.pass_turn,
        "use_sp_attack": a.use_sp_attack,
        "rotation": a.rotation,
        "x": a.x,
        "y": a.y,
    } for a in legal]
    model, torch = _strategic_load_model(checkpoint_file)
    map_obs, scalar_obs = _strategic_encode_state(state, player)
    action_feats = _strategic_encode_actions(state, player, legal_action_dicts)
    with torch.no_grad():
        logits, _ = model.forward_single(
            torch.as_tensor(map_obs, dtype=torch.float32),
            torch.as_tensor(scalar_obs, dtype=torch.float32),
            torch.as_tensor(action_feats, dtype=torch.float32),
        )
        idx = int(torch.argmax(logits).item())
    idx = max(0, min(idx, len(legal_action_dicts) - 1))
    return dict(legal_action_dicts[idx])


def choose_action_from_resolved_strategy(obs: ObservedState, resolved: ResolvedStrategy):
    state = _build_state_from_observation(obs)
    if resolved.mode == "strategy_id":
        return choose_action_from_strategy_id(state=state, player="P1", strategy_id=resolved.strategy_id)
    if resolved.mode == "ppo_checkpoint":
        payload = _choose_action_from_base_checkpoint(state, resolved.checkpoint_file, player="P1")
        for action in legal_actions(state, "P1"):
            if (
                action.card_number == payload.get("card_number")
                and action.pass_turn == bool(payload.get("pass_turn", False))
                and action.use_sp_attack == bool(payload.get("use_sp_attack", False))
                and action.rotation == int(payload.get("rotation", 0))
                and action.x == payload.get("x")
                and action.y == payload.get("y")
            ):
                return action
        raise RuntimeError(f"ppo strategy returned non-legal action: {resolved.checkpoint_file}")
    if resolved.mode == "strategic_ppo_checkpoint":
        payload = _choose_action_from_strategic_checkpoint(state, resolved.checkpoint_file, player="P1")
        for action in legal_actions(state, "P1"):
            if (
                action.card_number == payload.get("card_number")
                and action.pass_turn == bool(payload.get("pass_turn", False))
                and action.use_sp_attack == bool(payload.get("use_sp_attack", False))
                and action.rotation == int(payload.get("rotation", 0))
                and action.x == payload.get("x")
                and action.y == payload.get("y")
            ):
                return action
        raise RuntimeError(f"strategic ppo returned non-legal action: {resolved.checkpoint_file}")
    raise ValueError(f"unsupported resolved strategy mode: {resolved.mode}")


def resolve_strategy_id(
    config: ControllerConfig,
    map_id: str,
    map_name: str = "",
) -> str:
    map_id_use = str(map_id or "")
    map_name_use = str(map_name or "")
    if map_id_use and map_id_use in config.strategy_id_by_map:
        return str(config.strategy_id_by_map[map_id_use])
    if map_name_use and map_name_use in config.strategy_id_by_map_name:
        return str(config.strategy_id_by_map_name[map_name_use])
    return str(config.strategy_id)


def resolve_strategy(
    config: ControllerConfig,
    map_id: str,
    map_name: str = "",
) -> ResolvedStrategy:
    policy = _load_policy_config(config)
    maps = policy.get("maps", {}) if isinstance(policy.get("maps", {}), dict) else {}
    defaults = policy.get("default")
    source_base = str(config.policy_config_json or "")

    for key in (str(map_id or ""), str(map_name or "")):
        if key and key in maps:
            return _resolve_policy_entry(maps[key], f"{source_base}:{key}")

    if defaults is not None:
        return _resolve_policy_entry(defaults, f"{source_base}:default")

    strategy_id = resolve_strategy_id(config, map_id=map_id, map_name=map_name)
    return ResolvedStrategy(mode="strategy_id", label=strategy_id, strategy_id=strategy_id, source="embedded_config")


def _sp_pick_pool(obs: ObservedState) -> List[int]:
    state = _build_state_from_observation(obs)
    actions = legal_actions(state, "P1")
    seen: set[int] = set()
    ordered: List[int] = []
    for hand_card in obs.hand_card_numbers:
        for action in actions:
            card_number = action.card_number
            if (
                card_number == hand_card
                and action.use_sp_attack
                and card_number not in seen
            ):
                ordered.append(int(card_number))
                seen.add(int(card_number))
                break
    return ordered


def compile_action_with_defaults(action, obs: ObservedState) -> List[RemoteStep]:
    if bool(getattr(action, "surrender", False)):
        return [
            RemoteStep(bits=(1 << BIT_PLUS), hold_ms=100, gap_ms=1000),
            RemoteStep(bits=(1 << BIT_DPAD_RIGHT), hold_ms=100, gap_ms=1000),
            RemoteStep(bits=(1 << BIT_A), hold_ms=110, gap_ms=60),
        ]

    hand = obs.hand_card_numbers
    if not hand:
        raise ValueError("hand_card_numbers is empty")
    action_card = action.card_number if action.card_number is not None else hand[0]
    if action_card not in hand:
        raise ValueError(f"card {action.card_number} not in observed hand {hand}")

    steps: List[RemoteStep] = []
    if action.pass_turn:
        target_idx = hand.index(action_card)
        _move_axis(steps, dx=0, dy=2)
        _press_button(steps, BIT_A, hold_ms=110, gap_ms=60)
        _move_card_selection(steps, from_index=0, to_index=target_idx)
        _press_button(steps, BIT_A, hold_ms=110, gap_ms=60)
        return steps

    if action.use_sp_attack:
        sp_pool = _sp_pick_pool(obs)
        if action_card not in sp_pool:
            raise ValueError(f"card {action_card} not available in sp pick pool {sp_pool}")
        target_idx = sp_pool.index(action_card)
        _move_axis(steps, dx=1, dy=2)
        _press_button(steps, BIT_A, hold_ms=110, gap_ms=60)
        _move_card_selection(steps, from_index=0, to_index=target_idx)
        _press_button(steps, BIT_A, hold_ms=110, gap_ms=60)
    else:
        target_idx = hand.index(action_card)
        _move_card_selection(steps, from_index=int(obs.selected_hand_index or 0), to_index=target_idx)
        _press_button(steps, BIT_A, hold_ms=110, gap_ms=60)

    cursor_x, cursor_y = _initial_ui_anchor_for_map(obs)

    cw_steps = int(action.rotation) % 4
    ccw_steps = (-int(action.rotation)) % 4
    if cw_steps <= ccw_steps:
        for _ in range(cw_steps):
            _press_button(steps, BIT_X)
    else:
        for _ in range(ccw_steps):
            _press_button(steps, BIT_Y)

    target_x, target_y = _action_target_ui_xy(action, obs)
    _move_axis(steps, dx=int(target_x) - int(cursor_x), dy=int(target_y) - int(cursor_y))
    _press_button(steps, BIT_A, hold_ms=110, gap_ms=60)
    return steps


class TerminalDebugUI:
    def __init__(self, runtime: "AutoControllerRuntime"):
        self._runtime = runtime
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._stdin_fd: Optional[int] = None
        self._stdin_old_attrs = None
        self._interactive = False

    def start(self) -> None:
        if not self._runtime.config.debug_ui_enabled:
            return
        self._interactive = bool(sys.stdin.isatty() and sys.stdout.isatty())
        if self._interactive:
            import termios
            import tty

            self._stdin_fd = sys.stdin.fileno()
            self._stdin_old_attrs = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
        self._thread = threading.Thread(target=self._run, name="terminal-debug-ui", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        if self._interactive and self._stdin_fd is not None and self._stdin_old_attrs is not None:
            import termios

            with contextlib.suppress(Exception):
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_attrs)
        if self._interactive:
            with contextlib.suppress(Exception):
                sys.stdout.write("\x1b[2J\x1b[H")
                sys.stdout.flush()

    def _poll_key(self) -> None:
        if not self._interactive or self._stdin_fd is None:
            return
        ready, _, _ = select.select([self._stdin_fd], [], [], 0.0)
        if not ready:
            return
        chars = os.read(self._stdin_fd, 16).decode("utf-8", errors="ignore")
        for ch in chars:
            if ch == "=":
                self._runtime.adjust_turn_index(+1)
                continue
            if ch == "-":
                self._runtime.adjust_turn_index(-1)
                continue
            key = ch.lower()
            if key == "p":
                self._runtime.toggle_pause()
            elif key == "r":
                self._runtime.resume()
            elif key == "q":
                self._runtime.request_stop("user_requested_quit")

    def _render(self) -> None:
        state = self._runtime.debug_snapshot()
        lines = [
            "Tableturf AutoController Debug",
            "keys: p=pause/resume  r=resume  q=quit  -=turn-1  ==turn+1",
            "",
            f"status: {state['status']}",
            f"phase: {state['phase']}",
            f"turn: {state['turn']}",
            f"map: {state['map_id']}",
            f"wins: {state['wins']}",
            f"battles: {state['battles']}",
            f"pending result: {state['pending_result_check']}",
            f"playable: {state['playable']}",
            f"hand: {state['hand']}",
            f"sp: {state['p1_sp']}",
            f"action: {state['last_action']}",
            f"strategy: {state['strategy_id']}",
            f"strategy source: {state['strategy_source']}",
            f"serial: {state['serial_port']}",
            f"frame: {state['last_frame_path']}",
            f"analysis: {state['last_analysis_path']}",
            f"last error: {state['last_error']}",
            f"updated: {state['updated_at']}",
            "",
            "recent events:",
        ]
        for item in state["events"]:
            lines.append(f"  {item}")
        if self._interactive:
            sys.stdout.write("\x1b[2J\x1b[H" + "\n".join(lines) + "\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()

    def _run(self) -> None:
        last_render = 0.0
        while not self._stop.is_set():
            self._poll_key()
            now = time.monotonic()
            if now - last_render >= 0.2:
                self._render()
                last_render = now
            time.sleep(0.05)


class AutoControllerRuntime:
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.serial_port = _resolve_serial_port(config.serial_port, config.pick_serial)
        self.controller = SerialRemoteController(port=self.serial_port)
        self.vision = _FrameVisionPipeline(config)
        log_path = Path(config.log_file)
        if not log_path.is_absolute():
            log_path = REPO_ROOT / log_path
        self._logger = RuntimeLogWriter(log_path)
        self.turn_index = 1
        self.win_count = 0
        self.battle_count = 0
        self._pending_result_check = False
        self._battle_started = False
        self._last_progress_ts = time.monotonic()
        self._wait_a_enabled = True
        self._wait_silent_logged = False
        self._closed = False
        self._paused = threading.Event()
        self._stop_requested = threading.Event()
        self._status_lock = threading.Lock()
        self._status: Dict[str, Any] = {
            "status": "initializing",
            "phase": "boot",
            "turn": 0,
            "map_id": "",
            "wins": 0,
            "battles": 0,
            "pending_result_check": False,
            "playable": False,
            "hand": "",
            "p1_sp": 0,
            "last_action": "",
            "strategy_id": self.config.strategy_id,
            "strategy_source": "",
            "serial_port": self.serial_port,
            "last_frame_path": "",
            "last_analysis_path": "",
            "last_error": "",
            "updated_at": "",
            "events": [],
        }
        self._debug_ui = TerminalDebugUI(self)
        self._debug_ui.start()
        self._logger.write("自动控制器启动，已初始化串口、采集与调试界面。")
        self._push_event("runtime_ready")
        self._set_status(status="running", phase="idle")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._debug_ui.stop()
        self.vision.close()
        self.controller.close()

    def _timestamp(self) -> str:
        return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _set_status(self, **kwargs: Any) -> None:
        with self._status_lock:
            self._status.update(kwargs)
            self._status["updated_at"] = self._timestamp()

    def _push_event(self, message: str) -> None:
        with self._status_lock:
            events = list(self._status.get("events", []))
            events.append(f"{self._timestamp()} {message}")
            self._status["events"] = events[-8:]
            self._status["updated_at"] = self._timestamp()
        self._logger.write(message)

    def _mark_progress(self, reason: str) -> None:
        self._last_progress_ts = time.monotonic()
        self._logger.write(f"进度推进：{reason}")

    def _check_progress_timeout(self, phase: str) -> None:
        if not self._battle_started:
            return
        timeout = max(1.0, float(self.config.progress_timeout_seconds))
        elapsed = time.monotonic() - self._last_progress_ts
        if elapsed < timeout:
            return
        self._push_event(f"progress_timeout phase={phase} elapsed={elapsed:.1f}s，触发投降并重开。")
        self._run_surrender_sequence()
        self._battle_started = False
        self._pending_result_check = False
        self._set_status(pending_result_check=False)
        self.vision.reset_battle_context()
        raise RuntimeError("BATTLE_PROGRESS_TIMEOUT_SURRENDER")

    def _wait_for_next_turn_playable(self) -> None:
        self._push_event("进入下一回合确认阶段：固定等待 2 秒，不进行 playable 检测。")
        time.sleep(2.0)
        while True:
            self._wait_if_paused()
            self._ensure_not_stopped()
            self._check_progress_timeout("wait_next_turn_playable")
            self._set_status(phase="waiting_next_turn_playable", turn=self.turn_index)
            playable_result = self.vision.detect_playable()
            self._set_status(
                playable=bool(playable_result.get("playable")),
                p1_sp=self.vision.last_sp_count,
                last_frame_path=self.vision.last_frame_path,
                last_analysis_path=self.vision.last_analysis_path,
            )
            if playable_result.get("playable"):
                self._push_event("已重新检测到 playable，确认进入下一回合。")
                self._mark_progress("重新检测到可出牌状态，确认本回合已成功推进。")
                return
            time.sleep(max(0.1, self.config.playable_poll_seconds))

    def debug_snapshot(self) -> Dict[str, Any]:
        with self._status_lock:
            return dict(self._status)

    def toggle_pause(self) -> None:
        if self._paused.is_set():
            self.resume()
        else:
            self._paused.set()
            self._set_status(status="paused")
            self._push_event("paused_by_user")

    def resume(self) -> None:
        if self._paused.is_set():
            self._paused.clear()
            self._set_status(status="running")
            self._push_event("resumed_by_user")

    def adjust_turn_index(self, delta: int) -> None:
        if delta == 0:
            return
        old_turn = int(self.turn_index)
        new_turn = max(1, min(int(self.config.max_turns), old_turn + int(delta)))
        if new_turn == old_turn:
            self._push_event(f"turn_adjust_ignored current={old_turn} delta={delta}")
            return
        self.turn_index = new_turn
        self._set_status(turn=self.turn_index)
        self._push_event(f"turn_adjusted {old_turn}->{new_turn}")
        self._logger.write(f"用户手动调整当前回合：从第 {old_turn} 回合改为第 {new_turn} 回合。")

    def request_stop(self, reason: str) -> None:
        self._stop_requested.set()
        self._set_status(status="stopping", last_error=reason)
        self._push_event(reason)

    def _press_home_and_stop(self, reason: str) -> None:
        self._set_status(phase="stopping", last_error=reason)
        self._push_event(reason)
        self.controller.send_smart_sequence_csv("HOME,100")
        time.sleep(0.3)
        self.request_stop(reason)

    def _run_surrender_sequence(self) -> None:
        self._push_event("执行投降：按 PLUS，等待 1 秒，再按右，等待 1 秒，最后按 A。")
        self.controller.run_steps(
            [
                RemoteStep(bits=(1 << BIT_PLUS), hold_ms=100, gap_ms=1000),
                RemoteStep(bits=(1 << BIT_DPAD_RIGHT), hold_ms=100, gap_ms=1000),
                RemoteStep(bits=(1 << BIT_A), hold_ms=110, gap_ms=60),
            ]
        )

    def _finalize_previous_battle_as_not_win(self, reason: str) -> None:
        if not self._pending_result_check:
            return
        self._pending_result_check = False
        self._set_status(pending_result_check=False)
        self._push_event(reason)
        self._logger.write("上一局在重新进入可出牌前未检测到敌方战败标志，按未获胜处理。")

    def _record_battle_result_from_frame(self, lose_result: Dict[str, Any]) -> None:
        if not self._pending_result_check:
            return
        self._pending_result_check = False
        if bool(lose_result.get("lose")):
            self.win_count += 1
            self._set_status(wins=self.win_count, pending_result_check=False)
            self._push_event(f"battle_result=win total_wins={self.win_count}")
            self._mark_progress("检测到敌方战败标志，本局记为胜利。")
            if (not self.config.continuous_run) and self.win_count >= max(1, int(self.config.target_win_count)):
                self._press_home_and_stop("target_win_count_reached")
            return

        self._set_status(pending_result_check=False)
        self._push_event("battle_result=not_win")
        self._logger.write("当前结算检查帧未出现敌方战败标志，本局暂记为未获胜。")

    def _wait_if_paused(self) -> None:
        while self._paused.is_set() and not self._stop_requested.is_set():
            time.sleep(0.1)

    def _ensure_not_stopped(self) -> None:
        if self._stop_requested.is_set():
            raise KeyboardInterrupt("stop requested")

    def wait_until_playable(self) -> Dict[str, Any]:
        self._set_status(phase="waiting_playable", playable=False)
        self._push_event("开始等待进入可出牌状态。每次按 A 前都会先检查当前帧，避免在已可出牌时误按 A 选中第一张卡。")
        wait_a_step = [RemoteStep(bits=(1 << BIT_A), hold_ms=self.config.wait_press_hold_ms, gap_ms=0)]
        while True:
            self._wait_if_paused()
            self._ensure_not_stopped()
            frame, playable_result, lose_result = self.vision.inspect_wait_frame()
            self._set_status(
                phase="waiting_playable",
                playable=bool(playable_result.get("playable")),
                last_frame_path=self.vision.last_frame_path,
                last_analysis_path=self.vision.last_analysis_path,
                p1_sp=self.vision.last_sp_count,
                pending_result_check=self._pending_result_check,
                wins=self.win_count,
                battles=self.battle_count,
            )
            if self._pending_result_check:
                self._record_battle_result_from_frame(lose_result)
                self._ensure_not_stopped()
            if playable_result.get("playable"):
                if self._pending_result_check:
                    self._finalize_previous_battle_as_not_win("battle_result=unknown_assume_not_win_on_playable")
                self._push_event("playable_detected")
                self._mark_progress("检测到可出牌状态，停止继续按 A。")
                self._wait_a_enabled = False
                self._wait_silent_logged = False
                return playable_result
            if self._wait_a_enabled:
                self._push_event("当前未进入可出牌状态，执行一次 A 以推进到下一界面。")
                self.controller.run_steps(wait_a_step)
            else:
                if not self._wait_silent_logged:
                    self._push_event("当前未进入可出牌状态，但已进入静默等待阶段，本轮不再自动按 A。")
                    self._wait_silent_logged = True
            time.sleep(max(0.05, self.config.wait_press_gap_ms / 1000.0, self.config.playable_poll_seconds))

    def play_one_battle(self) -> None:
        self.turn_index = 1
        self.vision.reset_battle_context()
        self.wait_until_playable()
        self._battle_started = True
        self._mark_progress("新对局开始。")
        self._set_status(phase="battle_started", turn=1)
        self._push_event("battle_started")
        try:
            while self.turn_index <= self.config.max_turns:
                self._wait_if_paused()
                self._ensure_not_stopped()
                self._check_progress_timeout("battle_turn_loop")
                state = self.vision.parse_turn_state(turn_index=self.turn_index)
                observed_state = state.to_observed_state()
                resolved_strategy = resolve_strategy(self.config, state.map_id, state.map_name)
                self._logger.write(
                    f"第 {self.turn_index} 回合识别完成：地图={state.map_name}({state.map_id})，手牌={state.hand_card_numbers}，SP={state.p1_sp}。"
                )
                self._set_status(
                    phase="planning_action",
                    turn=self.turn_index,
                    map_id=state.map_id,
                    playable=bool(state.playable_result.get("playable")),
                    hand=",".join(str(n) for n in state.hand_card_numbers),
                    p1_sp=state.p1_sp,
                    last_frame_path=self.vision.last_frame_path,
                    last_analysis_path=self.vision.last_analysis_path,
                    last_error="",
                    strategy_id=resolved_strategy.label,
                    strategy_source=resolved_strategy.source,
                )
                action = choose_action_from_resolved_strategy(observed_state, resolved_strategy)
                action_text = (
                    f"card={action.card_number} rot={action.rotation} "
                    f"xy=({action.x},{action.y}) pass={action.pass_turn} sp={action.use_sp_attack} surrender={action.surrender}"
                )
                self._set_status(last_action=action_text)
                self._push_event(f"action_turn_{self.turn_index}: {action_text}")
                self._logger.write(f"第 {self.turn_index} 回合采用策略 {resolved_strategy.label}，来源={resolved_strategy.source}，动作={action_text}。")
                steps = compile_action_with_defaults(action, observed_state)
                self._set_status(phase="executing_action")
                self._push_event("开始执行手柄按键序列：执行过程中不进行 playable 检测。")
                self.controller.run_steps(steps)
                if self.turn_index < self.config.max_turns:
                    self._wait_for_next_turn_playable()
                self.turn_index += 1
                self._mark_progress(f"已完成一次动作执行，进入回合索引 {self.turn_index}。")
                self._set_status(phase="turn_complete", turn=min(self.turn_index, self.config.max_turns))
                time.sleep(max(0.1, self.config.playable_poll_seconds))
            self._push_event("battle_complete")
            self.battle_count += 1
            self._pending_result_check = True
            self._battle_started = False
            self._wait_a_enabled = True
            self._wait_silent_logged = False
            self._logger.write(f"12 回合结束，进入结算检查阶段。当前累计局数={self.battle_count}，累计胜场={self.win_count}。")
            self._set_status(phase="battle_complete", battles=self.battle_count, pending_result_check=True)
        except RuntimeError as exc:
            if str(exc) == "BATTLE_PROGRESS_TIMEOUT_SURRENDER":
                self._set_status(phase="battle_timeout_restarting", last_error=str(exc), pending_result_check=False)
                self._push_event("对局因长时间无推进而投降，已重置状态，准备重新从等待阶段开始。")
                self._wait_a_enabled = True
                self._wait_silent_logged = False
                return
            self._set_status(last_error=str(exc), phase="error")
            self._push_event(f"error: {exc}")
            raise
        except Exception as exc:
            self._set_status(last_error=str(exc), phase="error")
            self._push_event(f"error: {exc}")
            raise

    def run_forever(self) -> None:
        try:
            while True:
                self._ensure_not_stopped()
                self.play_one_battle()
        except KeyboardInterrupt:
            self._push_event("runtime_stopped")
            self._set_status(status="stopped", phase="stopped")


class _FrameVisionPipeline:
    def __init__(self, config: ControllerConfig):
        self._config = config
        layout_path = Path(config.layout_json)
        if not layout_path.is_absolute():
            layout_path = REPO_ROOT / layout_path
        self._layout = _load_layout(layout_path)
        frame_api_url = str(config.frame_api_url or "").strip()
        if frame_api_url:
            self._frame_api_launcher = FrameApiAutoLauncher(config)
            self._frame_api_launcher.ensure_started()
            self._capture = HttpJpegCaptureSource(frame_api_url=frame_api_url)
            device_name = f"http:{frame_api_url}"
        else:
            self._frame_api_launcher = None
            device_name = config.capture_device_name.strip() or auto_detect_capture_device_name(prefer_usb=True)
            if not device_name:
                raise RuntimeError("No capture device available")
            self._capture = FFmpegCaptureSource(
                device_name=device_name,
                width=config.capture_width,
                height=config.capture_height,
                fps=config.capture_fps,
                pixel_format=config.capture_pixel_format,
                strict_usb_only=False,
            )
        self._supplemental_provider = (
            _load_callable(config.supplemental_state_provider) if config.supplemental_state_provider else None
        )
        self._map_id: Optional[str] = None
        self._map_name: Optional[str] = None
        self._map_tracker: Optional[MapStateTracker] = None
        self.last_sp_count = 0
        debug_dir = Path(config.debug_frame_dir)
        if not debug_dir.is_absolute():
            debug_dir = REPO_ROOT / debug_dir
        self._debug_dir = debug_dir
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self.last_frame_path = ""
        self.last_analysis_path = ""
        self._playable_shot_index = 0

    def close(self) -> None:
        self._capture.stop()
        if self._frame_api_launcher is not None:
            self._frame_api_launcher.stop()

    def reset_battle_context(self) -> None:
        self._map_id = None
        self._map_name = None
        self._map_tracker = None

    def _read_latest_frame(self):
        fallback_specs = [
            {"width": self._config.capture_width, "height": self._config.capture_height, "pixel_format": self._config.capture_pixel_format},
            {"width": 1920, "height": 1080, "pixel_format": ""},
            {"width": 1280, "height": 720, "pixel_format": ""},
            {"width": 1280, "height": 720, "pixel_format": self._config.capture_pixel_format},
            {"width": 1920, "height": 1080, "pixel_format": "uyvy422"},
            {"width": 1280, "height": 720, "pixel_format": "uyvy422"},
            {"width": 1280, "height": 720, "pixel_format": "nv12"},
            {"width": 1920, "height": 1080, "pixel_format": "nv12"},
            {"width": 1280, "height": 720, "pixel_format": "yuyv422"},
            {"width": 1920, "height": 1080, "pixel_format": "yuyv422"},
        ]
        frame = self._capture.read_with_fallbacks(
            timeout_seconds=self._config.capture_read_timeout_seconds,
            fallback_specs=fallback_specs,
        )
        if frame is None:
            raise RuntimeError(f"Capture frame unavailable: {self._capture.last_error or 'unknown error'}")
        return frame

    def _save_debug_snapshot(self, frame, payload: Dict[str, Any]) -> None:
        if not self._config.save_debug_frames:
            return
        latest_png = self._debug_dir / "latest_frame.png"
        latest_json = self._debug_dir / "latest_analysis.json"
        cv2.imwrite(str(latest_png), frame)
        latest_json.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        self.last_frame_path = str(latest_png)
        self.last_analysis_path = str(latest_json)

    def _save_playable_capture(self, frame) -> None:
        if not self._config.save_debug_frames:
            return
        self._playable_shot_index += 1
        path = _unique_debug_image_path(self._debug_dir, "capture", self._playable_shot_index)
        cv2.imwrite(str(path), frame)

    def detect_playable(self) -> Dict[str, Any]:
        frame = self._read_latest_frame()
        result = detect_playable_banner(frame)
        self.last_sp_count = int(get_sp_count_frame(frame))
        result["frame_shape"] = [int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])]
        result["p1_sp"] = int(self.last_sp_count)
        if result.get("playable"):
            self._save_playable_capture(frame)
        self._save_debug_snapshot(
            frame,
            {
                "kind": "playable_poll",
                "playable_result": result,
                "p1_sp": int(self.last_sp_count),
            },
        )
        return result

    def inspect_wait_frame(self):
        frame = self._read_latest_frame()
        playable_result = detect_playable_banner(frame)
        lose_result = detect_lose_banner(frame)
        self.last_sp_count = int(get_sp_count_frame(frame))
        if playable_result.get("playable"):
            self._save_playable_capture(frame)
        self._save_debug_snapshot(
            frame,
            {
                "kind": "wait_poll",
                "playable_result": playable_result,
                "lose_result": lose_result,
                "p1_sp": int(self.last_sp_count),
            },
        )
        return frame, playable_result, lose_result

    def _detect_map_identity(self, frame) -> Dict[str, Any]:
        return detect_map_from_frame(frame, layout=self._layout)

    def _detect_map_identity_stable(self, first_frame) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        frames = [first_frame]
        for _ in range(2):
            time.sleep(0.08)
            frames.append(self._read_latest_frame())
        for frame in frames:
            result = dict(self._detect_map_identity(frame))
            attempts.append(result)

        best_by_map: Dict[str, Dict[str, Any]] = {}
        for idx, item in enumerate(attempts):
            map_id = str(item.get("map_id", "") or "")
            score = float(item.get("score", 0.0) or 0.0)
            if not map_id:
                continue
            slot = best_by_map.setdefault(
                map_id,
                {"count": 0, "best_score": -1.0, "best": item, "attempt_indices": []},
            )
            slot["count"] = int(slot["count"]) + 1
            slot["attempt_indices"].append(idx)
            if score > float(slot["best_score"]):
                slot["best_score"] = score
                slot["best"] = item

        if not best_by_map:
            fallback = attempts[0] if attempts else {}
            fallback["stability_attempts"] = attempts
            return fallback

        ranked = sorted(
            best_by_map.values(),
            key=lambda row: (int(row["count"]), float(row["best_score"])),
            reverse=True,
        )
        chosen = dict(ranked[0]["best"])
        chosen["stability_attempts"] = attempts
        chosen["stability_vote_count"] = int(ranked[0]["count"])
        chosen["stability_best_score"] = float(ranked[0]["best_score"])
        return chosen

    def _supplemental_state(self, frame, analysis_result: Dict[str, Any], turn_index: int) -> SupplementalState:
        if self._supplemental_provider is not None:
            payload = self._supplemental_provider(
                frame=frame,
                analysis_result=analysis_result,
                turn_index=turn_index,
                map_id=self._map_id,
            )
            if payload is None:
                return SupplementalState()
            if not isinstance(payload, dict):
                raise ValueError("supplemental_state_provider must return a dict or None")
            return SupplementalState.from_payload(payload)
        return SupplementalState(
            selected_hand_index=self._config.manual_fields.selected_hand_index,
            cursor_xy=self._config.manual_fields.cursor_xy,
            rotation=self._config.manual_fields.rotation,
            p1_sp=self._config.manual_fields.p1_sp,
        )

    def parse_turn_state(self, turn_index: int) -> ParsedTurnState:
        frame = self._read_latest_frame()
        playable_result = detect_playable_banner(frame)
        map_match: Dict[str, Any] = {}
        if self._map_id is None:
            map_match = dict(self._detect_map_identity_stable(frame))
            self._map_id = str(map_match.get("map_id", "") or "")
            self._map_name = str(map_match.get("map_name_zh", "") or "")
            if self._map_id:
                map_match = _pad_map_match_payload(self._map_id, map_match)
            if self._map_name:
                self._map_tracker = MapStateTracker(self._map_name)
        if not self._map_id:
            raise MissingInterfaceError(["map_id(tableturf_vision map match failed)"])
        if not self._map_name:
            name_map = {str(v): str(k) for k, v in map_name_cn_to_id().items()}
            self._map_name = name_map.get(self._map_id, "")
        if not self._map_name:
            raise MissingInterfaceError(["map_name(tableturf_vision map name missing)"])
        if self._map_tracker is None:
            self._map_tracker = MapStateTracker(self._map_name)

        hand_result = detect_hand_cards(frame)
        ordered_slots = _normalize_hand_slots(hand_result)
        card_matches = []
        hand_card_numbers: List[int] = []
        for slot in ordered_slots:
            match = _match_card(slot["matrix"])
            card_matches.append(
                {
                    "slot": slot.get("slot"),
                    "counts": slot.get("counts"),
                    "matrix": slot.get("matrix"),
                    "match": match,
                }
            )
            if match.get("number") is not None:
                hand_card_numbers.append(int(match["number"]))
        if len(hand_card_numbers) != 4:
            raise MissingInterfaceError(["hand_card_numbers(card recognition incomplete)"])

        self.last_sp_count = int(get_sp_count_frame(frame))
        raw_map_state_result = detect_map_state(frame, self._map_name)
        map_state_result = self._map_tracker.update_frame(frame)
        board_labels_raw = _map_state_to_board_labels(self._map_id, map_state_result)
        board_labels = _pad_board_labels_to_engine_dims(self._map_id, board_labels_raw)
        engine_map_grid = _extract_board_grid(board_labels)
        analysis_result = {
            "kind": "turn_state",
            "map_match": map_match or {"map_id": self._map_id, "map_name_zh": self._map_name},
            "raw_map_state": raw_map_state_result,
            "map_state": map_state_result,
            "board": {
                "raw_labels": board_labels_raw,
                "labels": board_labels,
            },
            "hand_cards": {
                "slots": ordered_slots,
            },
            "sp": {
                "p1_sp": int(self.last_sp_count),
            },
        }
        self._save_debug_snapshot(frame, analysis_result)

        supplemental = self._supplemental_state(frame, analysis_result, turn_index)
        missing: List[str] = []
        selected_hand_index = supplemental.selected_hand_index if supplemental.selected_hand_index is not None else 0
        rotation = supplemental.rotation if supplemental.rotation is not None else 0
        p1_sp = supplemental.p1_sp if supplemental.p1_sp is not None else int(self.last_sp_count)
        cursor_xy = supplemental.cursor_xy if supplemental.cursor_xy is not None else _initial_ui_anchor_for_map(
            ObservedState(
                map_id=self._map_id,
                hand_card_numbers=hand_card_numbers,
                p1_sp=int(p1_sp),
                turn=turn_index,
                map_grid=engine_map_grid,
            )
        )
        if missing and self._config.strict_missing_interfaces:
            raise MissingInterfaceError(missing)

        return ParsedTurnState(
            map_id=self._map_id,
            map_name=self._map_name,
            hand_card_numbers=hand_card_numbers,
            map_grid=engine_map_grid,
            playable_result=playable_result,
            analysis_result=analysis_result,
            card_matches=card_matches,
            turn=turn_index,
            selected_hand_index=int(selected_hand_index),
            cursor_xy=(int(cursor_xy[0]), int(cursor_xy[1])),
            rotation=int(rotation),
            p1_sp=int(p1_sp),
        )


def load_config(path: str) -> ControllerConfig:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    return ControllerConfig.from_json(cfg_path)
