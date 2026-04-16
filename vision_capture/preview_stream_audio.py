from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision_capture.adapter import list_avfoundation_video_device_rows


DEFAULT_CONFIG: Dict[str, object] = {
    "device_name": "UGREEN 35287",
    "audio_device": "auto",
    "audio_rate": 48000,
    "audio_channels": 2,
    "volume_level": 10,
}


def _load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    data = json.loads(path.read_text(encoding="utf-8"))
    out = dict(DEFAULT_CONFIG)
    if isinstance(data, dict):
        out.update(data)
    return out


def _list_avfoundation_audio_device_rows() -> List[Dict[str, str]]:
    cmd = ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    text = (proc.stderr or "") + (proc.stdout or "")
    rows: List[Dict[str, str]] = []
    in_audio = False
    for line in text.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if not in_audio:
            continue
        m = re.search(r"\[(\d+)\]\s+(.+)$", line.strip())
        if m:
            rows.append({"index": m.group(1).strip(), "name": m.group(2).strip()})
    return rows


def _resolve_video_name(device_name: str) -> str:
    value = str(device_name or "").strip()
    if not value:
        return ""
    if value.isdigit():
        for row in list_avfoundation_video_device_rows():
            if row["index"] == value:
                return row["name"]
    return value


def _resolve_audio_index(
    audio_device_setting: str,
    target_video_name: str,
    rows: List[Dict[str, str]],
) -> Optional[str]:
    setting = str(audio_device_setting or "").strip()
    if setting == "__same_as_video__":
        if not target_video_name:
            return None
        for row in rows:
            if row["name"] == target_video_name:
                return row["index"]
        return None

    if setting and setting.lower() == "none":
        return None

    if setting.isdigit():
        for row in rows:
            if row["index"] == setting:
                return setting
        return None

    if setting and setting.lower() not in {"auto", "none"}:
        for row in rows:
            if row["name"] == setting:
                return row["index"]

    if target_video_name:
        for row in rows:
            if row["name"] == target_video_name:
                return row["index"]

    return rows[0]["index"] if rows else None


def _build_capture_cmd(input_spec: str, rate: int, channels: int) -> List[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-thread_queue_size",
        "512",
        "-f",
        "avfoundation",
        "-i",
        input_spec,
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(rate),
        "-f",
        "s16le",
        "pipe:1",
    ]


def _build_play_cmd(rate: int, channels: int, volume_percent: int) -> List[str]:
    gain = max(0.0, min(2.0, float(volume_percent) / 100.0))
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-f",
        "s16le",
        "-ac",
        str(channels),
        "-ar",
        str(rate),
        "-i",
        "pipe:0",
        "-filter:a",
        f"volume={gain:.4f}",
        "-f",
        "audiotoolbox",
        "-",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audio-only real-time preview for capture card via ffmpeg capture + ffmpeg audiotoolbox play")
    parser.add_argument("--config", default="vision_capture/capture_config.json", help="json config path")
    parser.add_argument("--audio-device", default=None, help="audio device index/name/auto/none (override config)")
    parser.add_argument("--sample-rate", type=int, default=None, help="audio sample rate")
    parser.add_argument("--channels", type=int, default=None, help="audio channels")
    parser.add_argument("--volume", type=int, default=None, help="playback volume percent (0-100)")
    parser.add_argument("--list-audio-devices", action="store_true", help="list avfoundation audio devices and exit")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = _load_config(cfg_path)

    rows = _list_avfoundation_audio_device_rows()
    if args.list_audio_devices:
        if not rows:
            print("No AVFoundation audio device found.")
            return 1
        for row in rows:
            print(f"[{row['index']}] {row['name']}")
        return 0

    if not rows:
        print("No AVFoundation audio device found.")
        return 1

    target_video_name = _resolve_video_name(str(cfg.get("device_name", DEFAULT_CONFIG["device_name"])))
    if args.audio_device is None:
        audio_setting = "__same_as_video__"
    else:
        audio_setting = str(args.audio_device).strip()
    audio_index = _resolve_audio_index(audio_setting, target_video_name, rows)
    if audio_index is None:
        print(f"Audio unresolved for video device: {target_video_name or '<empty>'}")
        print("Tip: pass --audio-device <index|name> to override.")
        return 1

    rate = max(8000, int(args.sample_rate if args.sample_rate is not None else cfg.get("audio_rate", 48000)))
    channels = max(1, int(args.channels if args.channels is not None else cfg.get("audio_channels", 2)))
    level = int(cfg.get("volume_level", 10))
    volume_percent = max(0, min(100, int(args.volume if args.volume is not None else level * 10)))

    selected_name = ""
    for row in rows:
        if row["index"] == audio_index:
            selected_name = row["name"]
            break

    input_spec = f":{audio_index}"
    capture_cmd = _build_capture_cmd(input_spec=input_spec, rate=rate, channels=channels)
    play_cmd = _build_play_cmd(rate=rate, channels=channels, volume_percent=volume_percent)

    print(f"Config: {cfg_path}")
    print(f"Target video: {target_video_name or '<empty>'}")
    print(f"Selected audio: [{audio_index}] {selected_name or '<unknown>'}")
    print(f"Input spec: {input_spec}")
    print(f"Audio: rate={rate}, channels={channels}, volume={volume_percent}%")
    print("Running... Press Ctrl+C to stop.")

    cap_proc: Optional[subprocess.Popen] = None
    play_proc: Optional[subprocess.Popen] = None
    try:
        cap_proc = subprocess.Popen(
            capture_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=0,
        )
        if cap_proc.stdout is None:
            print("Failed to start audio capture pipe.")
            return 1
        play_proc = subprocess.Popen(
            play_cmd,
            stdin=cap_proc.stdout,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            bufsize=0,
        )
        cap_proc.stdout.close()

        while True:
            cap_rc = cap_proc.poll()
            play_rc = play_proc.poll()
            if cap_rc is not None or play_rc is not None:
                cap_tail = ""
                play_tail = ""
                if cap_proc.stderr is not None:
                    cap_tail = (cap_proc.stderr.read() or b"").decode("utf-8", errors="ignore").strip()
                if play_proc.stderr is not None:
                    play_tail = (play_proc.stderr.read() or b"").decode("utf-8", errors="ignore").strip()
                print(f"Stopped. capture_rc={cap_rc}, play_rc={play_rc}")
                if cap_tail:
                    print(f"[capture] {cap_tail.replace(chr(10), ' | ')}")
                if play_tail:
                    print(f"[play] {play_tail.replace(chr(10), ' | ')}")
                return 1 if (cap_rc not in (None, 0) or play_rc not in (None, 0)) else 0
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    finally:
        for proc in (play_proc, cap_proc):
            if proc is None:
                continue
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                except Exception:
                    pass
        for proc in (play_proc, cap_proc):
            if proc is None:
                continue
            if proc.poll() is None:
                try:
                    proc.wait(timeout=0.8)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass


if __name__ == "__main__":
    raise SystemExit(main())
