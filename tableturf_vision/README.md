# tableturf_vision

Tableturf-specific visual parsing and judge tools.

Responsibilities:
- parse board and card state from captured frames
- render terminal previews
- build fixed-point judge profiles
- validate green-point annotations
- detect whether the top-left playable banner is visible

Current map recognition stage:
- match maps against PNG templates in `tableturf_vision/参照基础`
- use only the center board-grid region defined by `board_roi_norm`
- exclude the left hand-card area and the right-side play/placement UI from map matching
- do not use `tableturf_vision/参照基础_坐标点确定` yet until point-confirmation assets are complete

Playable-banner detection:
- the playable banner ROI is currently fixed from the 15 reference PNGs
- confirmed absolute ROI on 1920x1080 frames: `x=20, y=111, w=262, h=159`
- use `tableturf_vision.playable_detector.is_playable_frame(...)` or
  `tableturf_vision.playable_detector.is_playable_image_path(...)`

This package intentionally excludes device capture and controller output.
