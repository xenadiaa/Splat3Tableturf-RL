"""加载卡牌、地图等资源的工具函数。"""

import json
from pathlib import Path
from typing import Optional

from ..assets.tpyes import GameMap

# 全局缓存
_map_data_cache: Optional[list] = None
_map_dict_by_id: Optional[dict] = None


def _load_map_info() -> list:
    """加载地图 JSON 数据（带缓存）。"""
    global _map_data_cache
    if _map_data_cache is None:
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        json_path = project_root / "data" / "maps" / "MiniGameMapInfo.json"

        if not json_path.exists():
            raise FileNotFoundError(f"地图数据文件不存在: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            _map_data_cache = json.load(f)

    return _map_data_cache


def _get_map_dict_by_id() -> dict:
    """获取按 id 索引的地图字典（带缓存）。"""
    global _map_dict_by_id
    if _map_dict_by_id is None:
        maps_data = _load_map_info()
        _map_dict_by_id = {m["id"]: m for m in maps_data}
    return _map_dict_by_id


def load_map(map_id: str) -> GameMap:
    """
    根据地图 ID 从 MiniGameMapInfo.json 加载地图，返回 GameMap 对象。

    Args:
        map_id: 地图 id（如 "WDiamond", "Square", "ManySp" 等）

    Returns:
        GameMap 对象

    Raises:
        KeyError: 找不到指定 id 的地图
    """
    map_dict = _get_map_dict_by_id()

    if map_id not in map_dict:
        raise KeyError(f"找不到 id 为 {map_id!r} 的地图")

    data = map_dict[map_id]
    # 深拷贝 grid，避免外部修改影响缓存
    grid = [row[:] for row in data["point_type"]]

    return GameMap(
        map_id=data["id"],
        name=data["name"],
        ename=data["ename"],
        width=data["width"],
        height=data["height"],
        grid=grid,
    )


def load_cards():
    """加载卡牌（待实现）。"""
    raise NotImplementedError("load_cards 尚未实现")
