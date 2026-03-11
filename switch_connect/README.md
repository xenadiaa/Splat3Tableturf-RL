# switch_connect

Realtime switch bridge without changing existing engine files.
Pipeline:
`vision_capture -> ObservedState -> policy(engine/nn) -> action -> virtual_gamepad -> serial`

## Layout

- `vision_capture/`
  - `types.py`: observed game state schema
  - `adapter.py`: AVFoundation capture-card adapter (`ffmpeg` by device name, default `UGREEN 35287`)
- `virtual_gamepad/`
  - `input_mapper.py`: action -> button step mapping
  - `serial_controller.py`: 0xFF remote mode sender
  - `smart_program_compat.py`: AutoController smart-program command encoding (0xFE)
- `strategy_mapper.py`
  - uses existing engine rules/scoring:
    - `src.engine.env_core.legal_actions`
    - `src.engine.env_core._score_bot_action`
- `policies/`
  - `router.py`: policy switch
  - `engine_policy.py`: built-in engine strategy
  - `nn_policy.py`: neural policy adapters (python module or external command)
- `bridge_runner.py`
  - CLI entrypoint for json state, action selection, optional serial output

## Quick Start

```bash
cd /Users/xenadia/Documents/GitHub/Splat3Tableturf-RL
python3 switch_connect/bridge_runner.py \
  --state-json switch_connect/examples/state_example.json \
  --print-steps
```

## Config-driven Capture (Recommended Daily Use)

Configure once in:
- [`capture_config.json`](/Users/xenadia/Documents/GitHub/Splat3Tableturf-RL/switch_connect/capture_config.json)

Then run one command:

```bash
cd /Users/xenadia/Documents/GitHub/Splat3Tableturf-RL
./.venv/bin/python switch_connect/capture_runner.py
```

Behavior:
- Select capture device with arrow keys (if `pick_device=true`)
- Capture screenshots using configured parameters
- Save to configured output directory
- If `pick_device=false`, it uses `device_name` from config.
- If `pick_device=false` but `device_name` is empty, it auto-enters picker.
- Every manual selection updates `device_name` in config automatically.

Probe capture card frame (by device name, not camera index):

```bash
python3 switch_connect/vision_capture/probe_video.py \
  --dump-devices \
  --device-name "UGREEN 35287"
```

Auto-detect capture card (recommended):

```bash
python3 switch_connect/vision_capture/probe_video.py \
  --dump-devices \
  --auto-device
```

Manual selection with arrow keys (video device):

```bash
python3 switch_connect/vision_capture/probe_video.py --pick-device
```

Continuous capture with unique filenames, skip duplicates, and keep card active:

```bash
python3 switch_connect/vision_capture/probe_video.py \
  --device-name "UGREEN 35287" \
  --shots 20 \
  --interval-ms 1000 \
  --warmup-seconds 5 \
  --prefix capture \
  --keep-active-seconds 0
```

Debug mode (continuous screenshots until Ctrl+C):

```bash
python3 switch_connect/vision_capture/probe_video.py \
  --device-name "UGREEN 35287" \
  --debug \
  --interval-ms 1000 \
  --prefix capture
```

Press Enter to capture (keep capture active; `q` + Enter to quit):

```bash
python3 switch_connect/vision_capture/capture_on_enter.py
```

This command reads `switch_connect/capture_config.json`.

Notes:
- Filenames are always unique (`prefix_YYYYMMDD_HHMMSS_microsec_idx.jpg`).
- Duplicate consecutive frames are skipped by default.
- `--debug` disables duplicate-skip and keeps saving continuously.
- Snapshot uses latest-frame drain to avoid saving stale buffered frames.
- Default is restart-per-shot to force fresh frame each screenshot.
- `--keep-active-seconds 0` keeps capture active until `Ctrl+C`.
- Use `--no-keep-active` to stop immediately after snapshot phase.

Use NN policy (python module callable):

```bash
python3 switch_connect/bridge_runner.py \
  --state-json switch_connect/examples/state_example.json \
  --policy nn-module \
  --nn-module switch_connect.policies.examples.simple_nn_policy:infer \
  --print-steps
```

Use NN policy (external process):

```bash
python3 switch_connect/bridge_runner.py \
  --state-json switch_connect/examples/state_example.json \
  --policy nn-command \
  --nn-command "python3 your_infer.py" \
  --print-steps
```

Send to serial:

```bash
python3 switch_connect/bridge_runner.py \
  --state-json switch_connect/examples/state_example.json \
  --serial-port /dev/cu.SLAB_USBtoUART \
  --print-steps
```

Send AutoController-compatible smart sequence (0xFE mode):

```bash
python3 switch_connect/virtual_gamepad/send_smart_sequence.py \
  --serial-port /dev/cu.SLAB_USBtoUART \
  --commands "A,1,Nothing,20,DRight,1,ASpam,10"
```

Manual selection with arrow keys (serial port):

```bash
python3 switch_connect/virtual_gamepad/send_smart_sequence.py \
  --pick-serial \
  --commands "A,1,Nothing,20,DRight,1,ASpam,10"
```

## Policy I/O contract

- policy input: `ObservedState` JSON
- policy output: Action JSON object:
  - `player` (`"P1"`)
  - `card_number` (int or null)
  - `pass_turn` (bool)
  - `use_sp_attack` (bool)
  - `rotation` (0..3)
  - `x`, `y` (int, nullable when pass)

## Smart Program Compatibility

- Firmware mode:
  - `0xFF + uint32`: remote bit-control
  - `0xFE + 30*(char + uint16 duration)`: smart sequence table
- Implemented from `AutoController_swsh/SourceCode/Bots/Others_SmartProgram/Others_SmartProgram.c`
- Compatible tokens include:
  - `A/B/X/Y/L/R/ZL/ZR/Home/Capture/Plus/Minus`
  - `DUp/DDown/DLeft/DRight`
  - `LUp/LDown/LLeft/LRight` and diagonals
  - `ASpam/BSpam/Loop`
  - combo tokens such as `DRightR`, `ZLBX`, `ZLA`, `BY`, `LUpClick`
