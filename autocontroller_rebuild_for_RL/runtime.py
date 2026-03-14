from __future__ import annotations

import contextlib
import datetime as dt
import importlib
import json
import os
import select
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
TABLETURF_SIM_ROOT = REPO_ROOT / "tableturf_sim"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TABLETURF_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TABLETURF_SIM_ROOT))

from src.assets.tableturf_types import Map_PointMask
from switch_connect.policies.router import choose_action
from switch_connect.ui.terminal_select import choose_with_arrows
from switch_connect.virtual_gamepad.device_discovery import list_serial_port_labels, parse_device_from_label
from switch_connect.virtual_gamepad.input_mapper import BIT_A, RemoteStep, compile_action_to_remote_steps
from switch_connect.virtual_gamepad.serial_controller import SerialRemoteController
from tableturf_vision.mapper_preview import _match_card
from tableturf_vision.playable_detector import detect_playable_banner
from tableturf_vision.reference_matcher import match_map_by_reference_board_labels
from tableturf_vision.tableturf_mapper import _load_layout, analyze_image
from vision_capture.adapter import FFmpegCaptureSource, auto_detect_capture_device_name
from vision_capture.state_types import ObservedState


class MissingInterfaceError(RuntimeError):
    """Raised when the orchestration flow needs fields not provided by current interfaces."""

    def __init__(self, missing_fields: Sequence[str]):
        self.missing_fields = list(missing_fields)
        joined = ", ".join(self.missing_fields)
        super().__init__(f"Missing required interface fields: {joined}")


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
    capture_device_name: str = ""
    capture_width: int = 1920
    capture_height: int = 1080
    capture_fps: int = 30
    capture_read_timeout_seconds: float = 5.0
    capture_drain_ms: int = 60
    serial_port: str = ""
    pick_serial: bool = True
    wait_press_hold_ms: int = 110
    wait_press_gap_ms: int = 1000
    playable_poll_seconds: float = 0.35
    max_turns: int = 12
    policy: str = "engine"
    style: str = ""
    level: str = "high"
    nn_module: str = ""
    nn_command: str = ""
    strict_missing_interfaces: bool = True
    layout_json: str = "tableturf_vision/tableturf_layout.json"
    manual_fields: ManualVisionFields = field(default_factory=ManualVisionFields)
    supplemental_state_provider: str = ""
    debug_ui_enabled: bool = True
    save_debug_frames: bool = True
    debug_frame_dir: str = "autocontroller_rebuild_for_RL/debug_runtime"

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
            "keys: p=pause/resume  r=resume  q=quit",
            "",
            f"status: {state['status']}",
            f"phase: {state['phase']}",
            f"turn: {state['turn']}",
            f"map: {state['map_id']}",
            f"playable: {state['playable']}",
            f"hand: {state['hand']}",
            f"action: {state['last_action']}",
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
        self.turn_index = 1
        self._closed = False
        self._paused = threading.Event()
        self._stop_requested = threading.Event()
        self._status_lock = threading.Lock()
        self._status: Dict[str, Any] = {
            "status": "initializing",
            "phase": "boot",
            "turn": 0,
            "map_id": "",
            "playable": False,
            "hand": "",
            "last_action": "",
            "serial_port": self.serial_port,
            "last_frame_path": "",
            "last_analysis_path": "",
            "last_error": "",
            "updated_at": "",
            "events": [],
        }
        self._debug_ui = TerminalDebugUI(self)
        self._debug_ui.start()
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

    def request_stop(self, reason: str) -> None:
        self._stop_requested.set()
        self._set_status(status="stopping", last_error=reason)
        self._push_event(reason)

    def _wait_if_paused(self) -> None:
        while self._paused.is_set() and not self._stop_requested.is_set():
            time.sleep(0.1)

    def _ensure_not_stopped(self) -> None:
        if self._stop_requested.is_set():
            raise KeyboardInterrupt("stop requested")

    def wait_until_playable(self) -> Dict[str, Any]:
        worker = ASpamWorker(
            controller=self.controller,
            hold_ms=self.config.wait_press_hold_ms,
            gap_ms=self.config.wait_press_gap_ms,
        )
        self._set_status(phase="waiting_playable", playable=False)
        self._push_event("start_waiting_playable")
        worker.start()
        try:
            while True:
                self._wait_if_paused()
                self._ensure_not_stopped()
                playable_result = self.vision.detect_playable()
                self._set_status(
                    phase="waiting_playable",
                    playable=bool(playable_result.get("playable")),
                    last_frame_path=self.vision.last_frame_path,
                    last_analysis_path=self.vision.last_analysis_path,
                )
                if playable_result.get("playable"):
                    self._push_event("playable_detected")
                    return playable_result
                time.sleep(max(0.05, self.config.playable_poll_seconds))
        finally:
            worker.stop()

    def play_one_battle(self) -> None:
        self.turn_index = 1
        self.wait_until_playable()
        self._set_status(phase="battle_started", turn=1)
        self._push_event("battle_started")
        try:
            while self.turn_index <= self.config.max_turns:
                self._wait_if_paused()
                self._ensure_not_stopped()
                state = self.vision.parse_turn_state(turn_index=self.turn_index)
                observed_state = state.to_observed_state()
                self._set_status(
                    phase="planning_action",
                    turn=self.turn_index,
                    map_id=state.map_id,
                    playable=bool(state.playable_result.get("playable")),
                    hand=",".join(str(n) for n in state.hand_card_numbers),
                    last_frame_path=self.vision.last_frame_path,
                    last_analysis_path=self.vision.last_analysis_path,
                    last_error="",
                )
                action = choose_action(
                    obs=observed_state,
                    policy=self.config.policy,
                    style=self.config.style or None,
                    level=self.config.level,
                    nn_module=self.config.nn_module,
                    nn_command=self.config.nn_command,
                )
                action_text = (
                    f"card={action.card_number} rot={action.rotation} "
                    f"xy=({action.x},{action.y}) pass={action.pass_turn} sp={action.use_sp_attack}"
                )
                self._set_status(last_action=action_text)
                self._push_event(f"action_turn_{self.turn_index}: {action_text}")
                steps = compile_action_to_remote_steps(action, observed_state)
                self._set_status(phase="executing_action")
                self.controller.run_steps(steps)
                self.turn_index += 1
                self._set_status(phase="turn_complete", turn=min(self.turn_index, self.config.max_turns))
                time.sleep(max(0.1, self.config.playable_poll_seconds))
            self._push_event("battle_complete")
            self._set_status(phase="battle_complete")
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
        device_name = config.capture_device_name.strip() or auto_detect_capture_device_name(prefer_usb=True)
        if not device_name:
            raise RuntimeError("No capture device available")
        self._capture = FFmpegCaptureSource(
            device_name=device_name,
            width=config.capture_width,
            height=config.capture_height,
            fps=config.capture_fps,
            strict_usb_only=False,
        )
        self._supplemental_provider = (
            _load_callable(config.supplemental_state_provider) if config.supplemental_state_provider else None
        )
        self._map_id: Optional[str] = None
        debug_dir = Path(config.debug_frame_dir)
        if not debug_dir.is_absolute():
            debug_dir = REPO_ROOT / debug_dir
        self._debug_dir = debug_dir
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self.last_frame_path = ""
        self.last_analysis_path = ""

    def close(self) -> None:
        self._capture.stop()

    def _read_latest_frame(self):
        frame = self._capture.read_latest(
            timeout_seconds=self._config.capture_read_timeout_seconds,
            drain_ms=self._config.capture_drain_ms,
        )
        if frame is None:
            raise RuntimeError(f"Capture frame unavailable: {self._capture.last_error or 'unknown error'}")
        return frame

    def detect_playable(self) -> Dict[str, Any]:
        frame = self._read_latest_frame()
        result = detect_playable_banner(frame)
        result["frame_shape"] = [int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])]
        return result

    def _analyze_frame(self, frame) -> Dict[str, Any]:
        import cv2
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp") as tmp:
            tmp_path = Path(tmp.name)
        try:
            ok = cv2.imwrite(str(tmp_path), frame)
            if not ok:
                raise RuntimeError(f"Failed to snapshot frame to temp path: {tmp_path}")
            result = analyze_image(tmp_path, layout=self._layout, write_overlay=False, out_dir=tmp_path.parent)
            if self._config.save_debug_frames:
                latest_png = self._debug_dir / "latest_frame.png"
                latest_json = self._debug_dir / "latest_analysis.json"
                cv2.imwrite(str(latest_png), frame)
                latest_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                self.last_frame_path = str(latest_png)
                self.last_analysis_path = str(latest_json)
            return result
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

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
        analysis_result = self._analyze_frame(frame)
        if self._map_id is None:
            map_match = match_map_by_reference_board_labels(analysis_result["board"]["labels"], self._layout)
            self._map_id = str(map_match.get("map_id", "") or "")
        if not self._map_id:
            raise MissingInterfaceError(["map_id(tableturf_vision map match failed)"])

        card_matches = [_match_card(card["labels"]) for card in analysis_result["cards"]]
        hand_card_numbers = [int(match["number"]) for match in card_matches if match.get("number") is not None]
        if len(hand_card_numbers) != 4:
            raise MissingInterfaceError(["hand_card_numbers(card recognition incomplete)"])

        supplemental = self._supplemental_state(frame, analysis_result, turn_index)
        missing: List[str] = []
        if supplemental.selected_hand_index is None:
            missing.append("selected_hand_index")
        if supplemental.cursor_xy is None:
            missing.append("cursor_xy")
        if supplemental.rotation is None:
            missing.append("rotation")
        if supplemental.p1_sp is None:
            missing.append("p1_sp")
        if missing and self._config.strict_missing_interfaces:
            raise MissingInterfaceError(missing)

        return ParsedTurnState(
            map_id=self._map_id,
            hand_card_numbers=hand_card_numbers,
            map_grid=_extract_board_grid(analysis_result["board"]["labels"]),
            playable_result=playable_result,
            analysis_result=analysis_result,
            card_matches=card_matches,
            turn=turn_index,
            selected_hand_index=supplemental.selected_hand_index,
            cursor_xy=supplemental.cursor_xy,
            rotation=int(supplemental.rotation or 0),
            p1_sp=int(supplemental.p1_sp or 0),
        )


def load_config(path: str) -> ControllerConfig:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    return ControllerConfig.from_json(cfg_path)
