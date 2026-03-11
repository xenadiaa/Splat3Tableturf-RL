# Tableturf Strategy RL

This folder provides a PPO training pipeline for learning Tableturf strategy with `tableturf_sim`.

## Files

- `rl_env.py`: Single-agent wrapper over `tableturf_sim` (`P1` is RL agent, `P2` is built-in bot).
- `networks.py`: Dynamic-action policy/value neural network.
- `ppo_trainer.py`: PPO rollout + update logic.
- `train.py`: CLI entry.

## Requirements

- Python 3.10+
- `numpy`
- `torch`

Install example:

```bash
pip install numpy torch
```

## Train

From repo root (`Splat3Tableturf-RL`):

```bash
python -m gamestrategy_RL.train --total-steps 100000 --map-id ManySp --bot-style balanced --bot-level mid
```

List map IDs:

```bash
python -m gamestrategy_RL.train --list-maps
```

List preset deck row IDs:

```bash
python -m gamestrategy_RL.train --list-decks
```

Use a specific deck by row id or numeric index:

```bash
python -m gamestrategy_RL.train --p1-deck MiniGame_Aori --p2-deck 3
```

Auto deck selection by map (derived from NPC map/deck relations):

```bash
python -m gamestrategy_RL.train --list-map-decks
```

Train all maps (one model directory per map):

```bash
python -m gamestrategy_RL.train --train-all-maps --total-steps 50000 --save-dir gamestrategy_RL/checkpoints_all_maps
```

Resume from checkpoint:

```bash
python -m gamestrategy_RL.train --map-id Cross --resume-checkpoint gamestrategy_RL/checkpoints/ppo_tableturf_u0010.pt
```

Training logs are written into each `save-dir`:

- `training_metrics.jsonl`: per-update records (loss/reward/win_rate/elapsed)
- `training_metrics.csv`: per-update table
- `training_summary.json`: final summary

Checkpoints are written to `gamestrategy_RL/checkpoints/`.
