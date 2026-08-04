from __future__ import annotations

import os
import shutil
import sys
from typing import List, Optional, Sequence

if os.name == "nt":
    import msvcrt
else:
    import termios
    import tty


class _RawMode:
    def __enter__(self):
        if os.name == "nt":
            return self
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if os.name != "nt":
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def _clear_screen() -> None:
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def _read_key() -> str:
    if os.name == "nt":
        ch = msvcrt.getwch()
        if ch in {"\r", "\n"}:
            return "enter"
        if ch == "\x03":
            return "ctrl_c"
        if ch in {"\x00", "\xe0"}:
            extended = msvcrt.getwch()
            if extended == "H":
                return "up"
            if extended == "P":
                return "down"
        return "other"

    ch = os.read(sys.stdin.fileno(), 1)
    if not ch:
        return ""
    if ch == b"\r" or ch == b"\n":
        return "enter"
    if ch == b"\x03":
        return "ctrl_c"
    if ch == b"\x1b":
        seq = os.read(sys.stdin.fileno(), 2)
        if seq == b"[A":
            return "up"
        if seq == b"[B":
            return "down"
        return "esc"
    return "other"


def choose_with_arrows(
    options: Sequence[str],
    title: str,
    footer: str = "Use ↑/↓ then Enter. Ctrl+C to cancel.",
) -> Optional[str]:
    if not options:
        return None
    idx = 0

    with _RawMode():
        while True:
            _clear_screen()
            width = shutil.get_terminal_size((80, 24)).columns
            lines: List[str] = []
            for line in str(title).splitlines() or [""]:
                lines.append(line[:width])
            lines.append("-" * min(width, 80))
            for i, opt in enumerate(options):
                if i == idx:
                    lines.append(f">  {opt}")
                else:
                    lines.append(f"   {opt}")
            if str(footer).strip():
                lines.append("")
                for line in str(footer).splitlines() or [""]:
                    lines.append(line[:width])
            body = "\r\n".join(lines) + "\r\n"
            sys.stdout.write(body)
            sys.stdout.flush()

            key = _read_key()
            if key == "up":
                idx = (idx - 1) % len(options)
            elif key == "down":
                idx = (idx + 1) % len(options)
            elif key == "enter":
                _clear_screen()
                return options[idx]
            elif key == "ctrl_c":
                _clear_screen()
                return None
