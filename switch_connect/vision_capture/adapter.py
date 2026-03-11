from __future__ import annotations

import json
import os
import re
import select
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .state_types import ObservedState


class VisionAdapter:
    """
    Frame -> ObservedState adapter.
    Implement `detect_state` with your OpenCV pipeline.
    """

    def detect_state(self, frame: Any) -> ObservedState:  # pragma: no cover
        raise NotImplementedError("implement OpenCV detection here")

    @staticmethod
    def load_state_json(path: str) -> ObservedState:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return ObservedState(**data)

    @staticmethod
    def dump_state_json(state: ObservedState, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")


def list_avfoundation_video_devices() -> List[str]:
    """
    Return AVFoundation video device names from ffmpeg listing.
    """
    cmd = ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    text = (proc.stderr or "") + (proc.stdout or "")
    names: List[str] = []
    in_video = False
    for line in text.splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            in_video = False
            continue
        if not in_video:
            continue
        # e.g. [AVFoundation indev @ ...] [0] UGREEN 35287
        m = re.search(r"\[\d+\]\s+(.+)$", line.strip())
        if m:
            names.append(m.group(1).strip())
    return names


def is_usb_capture_device_name(name: str) -> bool:
    n = (name or "").lower()
    keywords = ("ugreen", "capture", "uvc", "hdmi")
    return any(k in n for k in keywords)


def rank_capture_device_name(name: str) -> int:
    n = (name or "").lower()
    score = 0
    if "ugreen" in n:
        score += 100
    if "capture" in n:
        score += 40
    if "uvc" in n:
        score += 20
    if "hdmi" in n:
        score += 10
    return score


def auto_detect_capture_device_name(prefer_usb: bool = True) -> Optional[str]:
    """
    Pick best available capture-card-like video device from AVFoundation list.
    """
    names = list_avfoundation_video_devices()
    if not names:
        return None

    candidates = names
    if prefer_usb:
        filtered = [n for n in names if is_usb_capture_device_name(n)]
        if filtered:
            candidates = filtered

    candidates = sorted(candidates, key=rank_capture_device_name, reverse=True)
    return candidates[0] if candidates else None


class FFmpegCaptureSource:
    """
    Capture frames from AVFoundation by device name (preferred for USB capture card).
    This avoids relying on OpenCV camera index mapping.
    """

    def __init__(
        self,
        device_name: str = "UGREEN 35287",
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        strict_usb_only: bool = True,
    ):
        self.device_name = device_name
        self.width = width
        self.height = height
        self.fps = fps
        self.strict_usb_only = strict_usb_only
        self._proc: Optional[subprocess.Popen] = None
        self._frame_bytes = self.width * self.height * 3
        self._rx_buffer = bytearray()
        self.last_error: Optional[str] = None

    def start(self) -> None:
        if self._proc is not None:
            return
        if self.strict_usb_only and not is_usb_capture_device_name(self.device_name):
            self.last_error = f"DEVICE_REJECTED_NOT_USB_CAPTURE:{self.device_name}"
            raise ValueError(self.last_error)
        inp = f"{self.device_name}:none"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "avfoundation",
            "-framerate",
            str(self.fps),
            "-video_size",
            f"{self.width}x{self.height}",
            "-i",
            inp,
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=self._frame_bytes * 2,
        )
        if self._proc.stdout is not None:
            os.set_blocking(self._proc.stdout.fileno(), False)
        if self._proc.stderr is not None:
            os.set_blocking(self._proc.stderr.fileno(), False)

    def _stderr_tail(self, max_bytes: int = 4096) -> str:
        if self._proc is None or self._proc.stderr is None:
            return ""
        fd = self._proc.stderr.fileno()
        chunks = bytearray()
        while len(chunks) < max_bytes:
            try:
                part = os.read(fd, min(1024, max_bytes - len(chunks)))
            except BlockingIOError:
                break
            if not part:
                break
            chunks.extend(part)
        return chunks.decode("utf-8", errors="ignore").strip()

    def _read_from_pipe_once(self, timeout_seconds: float) -> bool:
        if self._proc is None or self._proc.stdout is None:
            return False
        out_fd = self._proc.stdout.fileno()
        ready, _, _ = select.select([out_fd], [], [], max(0.0, timeout_seconds))
        if not ready:
            return False
        try:
            chunk = os.read(out_fd, self._frame_bytes * 4)
        except BlockingIOError:
            return False
        if not chunk:
            return False
        self._rx_buffer.extend(chunk)
        return True

    def _pop_frame(self) -> Optional[np.ndarray]:
        if len(self._rx_buffer) < self._frame_bytes:
            return None
        frame_bytes = bytes(self._rx_buffer[: self._frame_bytes])
        del self._rx_buffer[: self._frame_bytes]
        return np.frombuffer(frame_bytes, dtype=np.uint8).reshape((self.height, self.width, 3))

    def read(self, timeout_seconds: float = 5.0) -> Optional[np.ndarray]:
        if self._proc is None:
            self.start()
        assert self._proc is not None
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        while True:
            frame = self._pop_frame()
            if frame is not None:
                self.last_error = None
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.last_error = f"FRAME_TIMEOUT({timeout_seconds}s)"
                return None
            if not self._read_from_pipe_once(remaining):
                rc = self._proc.poll()
                tail = self._stderr_tail()
                if rc is None and remaining > 0:
                    # No new bytes yet; continue polling until timeout.
                    continue
                if rc is None:
                    self.last_error = "NO_DATA_FROM_CAPTURE_SOURCE"
                else:
                    self.last_error = f"FFMPEG_EXITED({rc}) {tail}".strip()
                return None

    def read_latest(self, timeout_seconds: float = 5.0, drain_ms: int = 50) -> Optional[np.ndarray]:
        """
        Return the freshest frame by draining buffered stale frames.
        """
        frame = self.read(timeout_seconds=timeout_seconds)
        if frame is None:
            return None
        latest = frame

        drain_deadline = time.monotonic() + max(0.0, drain_ms / 1000.0)
        while time.monotonic() < drain_deadline:
            got = self._read_from_pipe_once(0.0)
            if not got:
                break
            while True:
                newer = self._pop_frame()
                if newer is None:
                    break
                latest = newer
        self.last_error = None
        return latest

    def stop(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        try:
            proc.terminate()
        except Exception:
            pass

        # Avoid blocking forever in wait() when ffmpeg is stuck.
        exited = False
        for _ in range(20):
            rc = proc.poll()
            if rc is not None:
                exited = True
                break
            try:
                time.sleep(0.05)
            except KeyboardInterrupt:
                break

        if not exited:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=0.5)
            except Exception:
                pass
        if proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass
        if proc.stderr:
            try:
                proc.stderr.close()
            except Exception:
                pass
