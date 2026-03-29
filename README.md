# Splat3Tableturf-RL
Try to rebuild Splat3 Tableturf core and run RL on it. Get stable winning strategy and used it on auto controller.

自留自用指令接口

整体流程：
自动对战：
cd /Users/xenadia/Documents/GitHub/Splat3Tableturf-RL
.venv/bin/python autocontroller_rebuild_for_RL/main.py --config autocontroller_rebuild_for_RL/runtime_config.local.json --tmp_win_target

单步调试：
cd /Users/xenadia/Documents/GitHub/Splat3Tableturf-RL
.venv/bin/python autocontroller_rebuild_for_RL/step_debug_terminal.py --config autocontroller_rebuild_for_RL/runtime_config.local.json

克隆水母对战
cd /Users/xenadia/Documents/GitHub/Splat3Tableturf-RL
.venv/bin/python autocontroller_rebuild_for_RL/clone_jelly_main.py --config autocontroller_rebuild_for_RL/runtime_config.local.json


占地斗士启动服务端：
局域网：
python3 /Users/xenadia/Documents/GitHub/Splat3Tableturf-RL/tableturf_sim/tools/play_service.py --bind 0.0.0.0

本地：
python3 /Users/xenadia/Documents/GitHub/Splat3Tableturf-RL/tableturf_sim/tools/play_service.py --bind 127.0.0.1

占地斗士启动客户端：
python3 /Users/xenadia/Documents/GitHub/Splat3Tableturf-RL/tableturf_sim/tools/play_client.py --name Host


TOOLS：
自动截屏：
cd /Users/xenadia/Documents/GitHub/Splat3Tableturf-RL
.venv/bin/python vision_capture/capture_runner.py

手动回车截屏：
cd /Users/xenadia/Documents/GitHub/Splat3Tableturf-RL
.venv/bin/python vision_capture/capture_on_enter.py

视频流展示：
cd /Users/xenadia/Documents/GitHub/Splat3Tableturf-RL
.venv/bin/python vision_capture/preview_stream_opencv.py
* http://127.0.0.1:8765/frame.jpg
* http://127.0.0.1:8765/frame.json
* http://127.0.0.1:8765/health

