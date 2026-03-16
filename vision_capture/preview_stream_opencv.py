from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from switch_connect.ui.terminal_select import choose_with_arrows
from vision_capture.adapter import (
    auto_detect_capture_device_name,
    is_usb_capture_device_name,
    list_avfoundation_video_device_rows,
    list_avfoundation_video_devices,
)


CAPTURE_PROFILES: List[Dict[str, object]] = [
    {"width": 1920, "height": 1080, "pixel_format": "uyvy422", "label": "1920x1080 / uyvy422"},
    {"width": 1280, "height": 720, "pixel_format": "uyvy422", "label": "1280x720 / uyvy422"},
    {"width": 1920, "height": 1080, "pixel_format": "nv12", "label": "1920x1080 / nv12"},
    {"width": 1280, "height": 720, "pixel_format": "nv12", "label": "1280x720 / nv12"},
    {"width": 1920, "height": 1080, "pixel_format": "yuyv422", "label": "1920x1080 / yuyv422"},
    {"width": 1280, "height": 720, "pixel_format": "yuyv422", "label": "1280x720 / yuyv422"},
]

DEFAULT_CONFIG: Dict[str, object] = {
    "device_name": "UGREEN 35287",
    "pick_device": False,
    "allow_non_usb": False,
    "preview_spec": "1920x1080 / uyvy422",
    "fps": 30,
    "window_title": "Capture Card Preview",
    "probe_seconds": 2.0,
    "frame_api_host": "127.0.0.1",
    "frame_api_port": 8765,
    "frame_jpeg_quality": 90,
}


def _load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    data = json.loads(path.read_text(encoding="utf-8"))
    out = dict(DEFAULT_CONFIG)
    if isinstance(data, dict):
        out.update(data)
    return out


def _pick_video_device(prefer_usb_only: bool) -> str:
    devices = list_avfoundation_video_devices()
    if prefer_usb_only:
        usb_devices = [d for d in devices if is_usb_capture_device_name(d)]
        if usb_devices:
            devices = usb_devices
    if not devices:
        return ""
    picked = choose_with_arrows(devices, "Select capture video device")
    return picked or ""


def _resolve_device(args: argparse.Namespace) -> str:
    manual = (args.device_name or "").strip()
    if args.pick_device:
        return _pick_video_device(prefer_usb_only=not args.allow_non_usb)
    if manual:
        return manual
    return auto_detect_capture_device_name(prefer_usb=not args.allow_non_usb) or ""


def _resolve_video_input_name(device_name: str) -> str:
    name = str(device_name or "").strip()
    if name.isdigit():
        return f"{name}:none"
    for row in list_avfoundation_video_device_rows():
        if row["name"] == name:
            return f"{row['index']}:none"
    return f"{name}:none"


def _profile_from_spec(spec: str) -> Dict[str, object]:
    size, pixel_format = [part.strip() for part in spec.split("/", 1)]
    width_str, height_str = [part.strip() for part in size.lower().split("x", 1)]
    return {
        "width": int(width_str),
        "height": int(height_str),
        "pixel_format": pixel_format,
        "label": f"{int(width_str)}x{int(height_str)} / {pixel_format}",
    }


def _build_profiles(spec: str) -> List[Dict[str, object]]:
    if spec:
        return [_profile_from_spec(spec)]
    return [dict(profile) for profile in CAPTURE_PROFILES]

def _resolve_video_index(device_name: str) -> Optional[int]:
    name = str(device_name or "").strip()
    if name.isdigit():
        return int(name)
    for row in list_avfoundation_video_device_rows():
        if row["name"] == name:
            return int(row["index"])
    return None


def _open_video_capture(
    device_name: str,
    profiles: List[Dict[str, object]],
    fps: int,
    timeout_seconds: float,
) -> Tuple[Optional[cv2.VideoCapture], Optional[Dict[str, object]], Optional[np.ndarray], str]:
    last_error = ""
    device_index = _resolve_video_index(device_name)
    if device_index is None:
        return None, None, None, f"DEVICE_INDEX_NOT_FOUND:{device_name}"

    for profile in profiles:
        width = int(profile["width"])
        height = int(profile["height"])
        cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            last_error = f"OPENCV_OPEN_FAILED:{device_index}"
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        deadline = time.monotonic() + max(0.5, timeout_seconds)
        frame: Optional[np.ndarray] = None
        while time.monotonic() < deadline:
            ok, grabbed = cap.read()
            if ok and grabbed is not None and grabbed.size > 0:
                frame = grabbed
                break
            time.sleep(0.03)

        if frame is not None:
            return cap, profile, frame, ""

        last_error = f"{profile['label']}: FRAME_TIMEOUT({timeout_seconds}s)"
        cap.release()

    return None, None, None, last_error


@dataclass
class FrameState:
    jpeg_quality: int
    frame: Optional[np.ndarray] = None
    jpeg_bytes: Optional[bytes] = None
    last_frame_ts: float = 0.0
    profile_label: str = ""
    device_name: str = ""
    frame_count: int = 0
    last_error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_frame(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        with self.lock:
            self.frame = frame.copy()
            self.jpeg_bytes = encoded.tobytes()
            self.last_frame_ts = time.time()
            self.frame_count += 1
            self.last_error = ""

    def snapshot_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return None if self.jpeg_bytes is None else bytes(self.jpeg_bytes)

    def snapshot_metadata(self) -> Dict[str, object]:
        with self.lock:
            return {
                "device_name": self.device_name,
                "profile_label": self.profile_label,
                "frame_count": self.frame_count,
                "last_frame_ts": self.last_frame_ts,
                "has_frame": self.jpeg_bytes is not None,
                "last_error": self.last_error,
            }


def _make_handler(state: FrameState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path in {"/health", "/healthz"}:
                body = json.dumps(state.snapshot_metadata(), ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path in {"/frame.jpg", "/frame.jpeg"}:
                body = state.snapshot_jpeg()
                if body is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No frame available yet")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/frame.json":
                body = json.dumps(state.snapshot_metadata(), ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Supported paths: /health, /frame.jpg, /frame.json")

    return Handler


def _overlay_status(frame: np.ndarray, profile_label: str, device_name: str, api_url: str) -> None:
    lines = [
        f"Device: {device_name}",
        f"Capture: {profile_label}",
        f"API: {api_url}/frame.jpg",
        "Keys: q/ESC quit",
    ]
    for idx, text in enumerate(lines):
        y = 30 + idx * 28
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Open an OpenCV preview window and expose the latest frame over HTTP.")
    parser.add_argument("--config", default="vision_capture/capture_config.json", help="json config file path")
    parser.add_argument("--device-name", default=None, help="manual AVFoundation device name")
    parser.add_argument("--pick-device", action="store_true", help="choose device via arrow keys")
    parser.add_argument("--allow-non-usb", action="store_true", help="allow non-capture-card cameras")
    parser.add_argument("--spec", default=None, help="single capture spec like '1920x1080 / uyvy422'")
    parser.add_argument("--fps", type=int, default=None, help="capture frame rate")
    parser.add_argument("--window-title", default=None, help="OpenCV window title")
    parser.add_argument("--probe-seconds", type=float, default=None, help="timeout for opening first frame")
    parser.add_argument("--host", default=None, help="HTTP bind host")
    parser.add_argument("--port", type=int, default=None, help="HTTP bind port")
    parser.add_argument("--jpeg-quality", type=int, default=None, help="JPEG quality for /frame.jpg")
    parser.add_argument("--list-profiles", action="store_true", help="print supported capture profiles and exit")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = _load_config(cfg_path)

    if args.list_profiles:
        for profile in CAPTURE_PROFILES:
            print(profile["label"])
        return 0

    if args.device_name is None:
        args.device_name = str(cfg.get("device_name", DEFAULT_CONFIG["device_name"]))
    if not args.pick_device:
        args.pick_device = bool(cfg.get("pick_device", DEFAULT_CONFIG["pick_device"]))
    if not args.allow_non_usb:
        args.allow_non_usb = bool(cfg.get("allow_non_usb", DEFAULT_CONFIG["allow_non_usb"]))
    if args.spec is None:
        args.spec = str(cfg.get("preview_spec", DEFAULT_CONFIG["preview_spec"]))
    if args.fps is None:
        args.fps = int(cfg.get("fps", DEFAULT_CONFIG["fps"]))
    if args.window_title is None:
        args.window_title = str(cfg.get("window_title", DEFAULT_CONFIG["window_title"]))
    if args.probe_seconds is None:
        args.probe_seconds = float(cfg.get("probe_seconds", DEFAULT_CONFIG["probe_seconds"]))
    if args.host is None:
        args.host = str(cfg.get("frame_api_host", DEFAULT_CONFIG["frame_api_host"]))
    if args.port is None:
        args.port = int(cfg.get("frame_api_port", DEFAULT_CONFIG["frame_api_port"]))
    if args.jpeg_quality is None:
        args.jpeg_quality = int(cfg.get("frame_jpeg_quality", DEFAULT_CONFIG["frame_jpeg_quality"]))

    device_name = _resolve_device(args)
    if not device_name:
        print("No capture device selected/detected.")
        return 2

    profiles = _build_profiles(args.spec or "")
    cap, active_profile, first_frame, error = _open_video_capture(
        device_name=device_name,
        profiles=profiles,
        fps=args.fps,
        timeout_seconds=args.probe_seconds,
    )
    if cap is None or active_profile is None or first_frame is None:
        print(f"Unable to open preview stream. last_error={error}")
        return 1

    state = FrameState(jpeg_quality=max(30, min(100, args.jpeg_quality)))
    state.device_name = device_name
    state.profile_label = str(active_profile["label"])
    state.update_frame(first_frame)
    api_url = f"http://{args.host}:{args.port}"

    server = ThreadingHTTPServer((args.host, args.port), _make_handler(state))
    server_thread = threading.Thread(target=server.serve_forever, name="frame-api-server", daemon=True)
    server_thread.start()
    print(f"Device: {device_name}")
    print(f"Input index: {_resolve_video_index(device_name)}")
    print(f"Streaming: {active_profile['label']}")
    print(f"Frame API: {api_url}/frame.jpg")
    print(f"Health API: {api_url}/health")
    last_frame_ts = time.monotonic()

    try:
        cv2.namedWindow(args.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            args.window_title,
            min(int(active_profile["width"]), 1280),
            min(int(active_profile["height"]), 720),
        )

        while True:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                state.update_frame(frame)
                last_frame_ts = time.monotonic()

            if time.monotonic() - last_frame_ts >= max(1.0, args.probe_seconds):
                state.last_error = f"FRAME_TIMEOUT({args.probe_seconds}s)"
                print(f"No frame received within {args.probe_seconds}s. Check input signal/resolution.")
                break

            if state.frame is None:
                continue
            display_frame = state.frame.copy()
            _overlay_status(display_frame, str(active_profile["label"]), device_name, api_url)
            cv2.imshow(args.window_title, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        server.shutdown()
        server.server_close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
