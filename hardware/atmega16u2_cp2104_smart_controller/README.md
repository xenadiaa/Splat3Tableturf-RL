# ATmega16U2 + CP2104 AutoController Smart Controller

Status: schematic specification draft. Do not fabricate before ERC, footprint and USB routing review.

## Architecture

```
PC USB -> CP2104 -> UART_TX/RX -> ATmega16U2 -> Switch USB
```

- U1 ATmega16U2-AU runs the AutoController/LUFA Switch HID firmware.
- U2 CP2104-F03-GM exposes the existing 9600-baud bidirectional Smart serial protocol to the PC.
- The two USB VBUS rails must never be tied together. Only GND and the UART signals cross the two domains.
- The first prototype should use USB Micro-B receptacles to avoid USB-C CC circuitry and reduce variables.

## Required functional blocks

### Switch-side ATmega16U2

- U1: ATmega16U2-AU, TQFP32, 5 V, 16 MHz.
- J1: USB Micro-B to Switch dock.
- J1 VBUS -> F1 500 mA resettable fuse -> `+5V_SWITCH`.
- J1 D+/D- -> U3 USBLC6-2SC6 ESD array -> 22 ohm series resistors -> U1 USB D+/D-.
- Y1: 16 MHz crystal with two 22 pF load capacitors.
- Every VCC/AVCC/UVCC pin: local 100 nF decoupling; one 4.7 uF bulk capacitor on `+5V_SWITCH`.
- UCAP: 1 uF to GND as required by the AVR USB regulator.
- RESET: 10 k pull-up, reset button to GND and ICSP header.
- HWB: 10 k pull-up and DFU button to GND so the board can enter the same DFU workflow as UNO R3.
- J3: 2x3 AVR ICSP header for recovery/programming.

### PC-side CP2104

- U2: CP2104-F03-GM, QFN24.
- J2: USB Micro-B to PC.
- J2 VBUS -> `+5V_PC`; connect to CP2104 REGIN and VBUS only.
- J2 D+/D- -> U4 USBLC6-2SC6 ESD array -> CP2104 D+/D-.
- CP2104 VDD regulator output: 1 uF and 100 nF to GND.
- CP2104 VIO: connect to VDD for 3.45 V UART signaling.
- CP2104 RST: 4.7 k pull-up to VIO.
- CP2104 VPP: 4.7 uF to GND for configuration-ROM programming.

### UART bridge

The original UNO setup works through its existing series-resistor network. The custom board keeps that behavior but adds explicit pads so voltage levels can be verified before population:

- CP2104 TXD -> R21 1 k -> ATmega16U2 RXD.
- ATmega16U2 TXD -> R22 1 k -> CP2104 RXD.
- Common GND.
- Do not connect `+5V_PC` to `+5V_SWITCH`.
- Populate footprints for an optional two-channel level translator if measurements show the selected CP2104 revision/module is not tolerant of the 5 V AVR TX level. Direct 5 V into a CP2104 input must not be assumed safe without checking the exact device revision/data sheet.

## Firmware compatibility

Keep the U1 clock, USB pins, HWB/RESET and UART assignment identical to the UNO R3 ATmega16U2 AutoController target. The PC protocol remains:

- 9600 baud, 8N1.
- `0xFF + uint32_le` for live bit state.
- `0xFE + 30 * (command_char + uint16_le duration)` for Smart sequences.

## Bring-up order

1. Populate only U1, J1, clock, reset/DFU, ICSP and U1 decoupling.
2. Flash the known-good UNO R3 ATmega16U2 AutoController HEX.
3. Confirm Switch enumeration and basic A-button operation.
4. Populate U2 and J2; confirm CP2104 enumeration on the PC.
5. Verify UART idle levels with a multimeter/oscilloscope before fitting R21/R22.
6. Fit UART resistors and run the existing `probe_firmware()` command.
7. Only after the schematic passes review, assign verified footprints and route the PCB.

