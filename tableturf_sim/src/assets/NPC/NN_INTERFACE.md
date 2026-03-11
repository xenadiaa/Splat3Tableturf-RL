# NPC NN Strategy Interface

Place per-NPC NN strategy files in this folder with name:

- `NPCName_nn.json`
- or `NPCName_nn.py`

If both are absent, engine falls back to default 3x3 bot strategy.

## JSON mode (`NPCName_nn.json`)

```json
{
  "type": "file_action_json",
  "action_file": "/absolute/path/to/one_action.json"
}
```

`one_action.json` must be one legal action payload:

```json
{
  "card_number": 58,
  "pass_turn": false,
  "use_sp_attack": false,
  "rotation": 0,
  "x": 3,
  "y": 4
}
```

## Python mode (`NPCName_nn.py`)

Implement function `choose_action`:

```python
def choose_action(state, player, legal_actions, context):
    # legal_actions: list[dict]
    # return one action dict that matches one item in legal_actions
    return legal_actions[0]
```

Returned action must match one legal action exactly, otherwise engine falls back.
