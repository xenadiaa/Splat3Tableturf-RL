def get_collision_mask(m1, x1, y1, m2, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    # 如果偏移超过 8 格，物理上不可能重叠
    if abs(dx) >= 8 or abs(dy) >= 8:
        return 0
    
    # 将 M2 对齐到 M1 的坐标系
    # 假设 M2 的第 0 位是 (0,0)，第 63 位是 (7,7)
    # 行偏移 dy 相当于移动 8 * dy 位，列偏移 dx 相当于移动 dx 位
    shift_amount = dy * 8 + dx
    
    if shift_amount > 0:
        shifted_m2 = m2 << shift_amount
    else:
        shifted_m2 = m2 >> abs(shift_amount)

    # 关键：位移后可能会有“环绕”污染（比如第1行移到了第2行）
    # 需要根据 dx 使用掩码清空无效位
    col_mask = 0xFFFFFFFFFFFFFFFF
    if dx > 0:
        # 左移，清空左侧 dx 列
        for i in range(dx): col_mask &= ~0x0101010101010101 << i
    elif dx < 0:
        # 右移，清空右侧 dx 列
        for i in range(abs(dx)): col_mask &= ~0x8080808080808080 >> i
        
    shifted_m2 &= col_mask
    
    # 最终冲突位（在 M1 坐标系下）
    return m1 & shifted_m2

#比较单格归属
def compare_single_box_final_result(player, enemy):
    # 假设 Epision 是一个预定义的极小值常数
    epsilon = 0.000001 
    
    # 计算优先级
    player_priority = 30.0 + 0.5*player['is_using_sp'] - player['card_count'] + 30*player['is_sp_box']
    enemy_priority = 30.0 + 0.5*enemy['is_using_sp'] - enemy['card_count'] + 30*player['is_sp_box']
    
    # 逻辑判断
    if abs(player_priority - enemy_priority) < epsilon:
        return "BT_Conflict"
    elif player_priority > enemy_priority:
        return player['box_type']
    else:
        return enemy['box_type']

#存在相邻卡牌返回True，否则返回False
def have_neighbor_with_card(card, pos, map):
    #TODO：判断相邻卡牌逻辑
    return True

def have_neighbor_with_special(card, pos, map):
    #TODO：判断相邻卡牌逻辑
    return True

def is_able_to_place_card(card, map):
    #TODO：前端用于判断是否有可以放置卡牌的位置
    return True

def is_able_to_place_withsp(card, map):
    #TODO：前端用于判断是否有可以放置使用sp的卡牌位置
    return True

def rotate_cell(r: int, c: int, rotation: int) -> tuple[int, int]:
    """(r,c) 旋转后所在的新坐标。rotation: 0,1,2,3 → 0°,90°,180°,270° CW"""
    if rotation == 0:
        return (r, c)
    if rotation == 1:   # 90° CW
        return (c, 7 - r)
    if rotation == 2:   # 180°
        return (7 - r, 7 - c)
    if rotation == 3:   # 270° CW
        return (7 - c, r)
    raise ValueError("rotation must be 0-3")

    return newrc

#锚点选择，json中第一个非empty点