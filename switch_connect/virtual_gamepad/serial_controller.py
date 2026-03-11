from __future__ import annotations

import struct
import time
from typing import Iterable

from .input_mapper import RemoteStep
from .smart_program_compat import SMART_HEX_VERSION, encode_smart_sequence, encode_smart_sequence_csv

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None


class SerialRemoteController:
    """Talk to AutoController smart-program firmware via 0xFF remote mode."""

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 0.1):
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        self._ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)

    def close(self) -> None:
        self._ser.close()

    def send_bits(self, bits: int) -> None:
        payload = bytes([0xFF]) + struct.pack("<I", bits & 0xFFFFFFFF)
        self._ser.write(payload)

    def release(self) -> None:
        self.send_bits(0)

    def run_steps(self, steps: Iterable[RemoteStep]) -> None:
        for step in steps:
            self.send_bits(step.bits)
            time.sleep(max(0.0, step.hold_ms / 1000.0))
            self.release()
            time.sleep(max(0.0, step.gap_ms / 1000.0))

    def read_bytes(self, max_len: int = 64) -> bytes:
        return self._ser.read(max_len)

    def send_smart_sequence_payload(self, payload: bytes) -> None:
        """
        Send already encoded sequence payload.
        payload must start with 0xFE and include full command table.
        """
        if not payload or payload[0] != 0xFE:
            raise ValueError("smart sequence payload must start with 0xFE")
        self._ser.write(payload)

    def send_smart_sequence(self, commands) -> None:
        payload = encode_smart_sequence(commands)
        self.send_smart_sequence_payload(payload)

    def send_smart_sequence_csv(self, command_csv: str) -> None:
        payload = encode_smart_sequence_csv(command_csv)
        self.send_smart_sequence_payload(payload)

    def wait_smart_sequence_start_ack(self, timeout_seconds: float = 2.0) -> bool:
        """
        First command completion ack is SMART_HEX_VERSION (8) when accepted.
        """
        deadline = time.time() + max(0.1, timeout_seconds)
        while time.time() < deadline:
            b = self._ser.read(1)
            if not b:
                continue
            if b[0] == SMART_HEX_VERSION:
                return True
            # Some firmware replies 0xFF for generic command completion.
            if b[0] == 0xFF:
                return True
        return False
