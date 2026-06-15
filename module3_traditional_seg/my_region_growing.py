"""
手动实现区域生长分割 — 实验9
支持种子点选取 + 灰度差阈值准则 + 八邻域生长
"""
import numpy as np


def my_region_grow(image, seed_point, threshold=10, connectivity=8):
    """
    区域生长算法
    原理:
    1. 从种子点开始，检查其邻域像素
    2. 若邻域像素与已包含区域均值之差 ≤ threshold，则加入区域
    3. 递归处理新加入的像素，直到没有像素满足条件
    4. 使用队列实现广度优先搜索(BFS)，避免递归栈溢出

    :param image: 灰度图像
    :param seed_point: 种子点 (row, col)
    :param threshold: 灰度差阈值 — 像素须满足 |I(p) - mean(region)| ≤ threshold
    :param connectivity: 邻域类型 — 4 或 8
    :return: 二值掩膜 (True=分割区域)
    """
    img = np.asarray(image, dtype=np.float64)
    H, W = img.shape
    mask = np.zeros((H, W), dtype=bool)

    # 种子点
    seed_r, seed_c = int(seed_point[0]), int(seed_point[1])
    if not (0 <= seed_r < H and 0 <= seed_c < W):
        raise ValueError(f"种子点超出图像范围: ({seed_r}, {seed_c})")

    # 邻域偏移 (8邻域)
    if connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
    else:  # 4邻域
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    # 初始化: 队列(BFS) + 区域像素列表
    queue = [(seed_r, seed_c)]
    mask[seed_r, seed_c] = True
    region_pixels = [img[seed_r, seed_c]]

    while queue:
        r, c = queue.pop(0)

        for dr, dc in offsets:
            nr, nc = r + dr, c + dc

            # 检查边界
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            # 检查是否已访问
            if mask[nr, nc]:
                continue

            # 检查灰度差
            region_mean = np.mean(region_pixels)
            pixel_val = img[nr, nc]
            if abs(pixel_val - region_mean) <= threshold:
                mask[nr, nc] = True
                region_pixels.append(pixel_val)
                queue.append((nr, nc))

    return mask


def my_region_grow_multi_seed(image, seed_points, threshold=10, connectivity=8):
    """
    多种子点区域生长: 多个种子点的并集
    :param seed_points: 种子点列表 [(r1,c1), (r2,c2), ...]
    :return: 合并后的二值掩膜
    """
    combined_mask = np.zeros_like(image, dtype=bool)
    for sp in seed_points:
        mask = my_region_grow(image, sp, threshold, connectivity)
        combined_mask = combined_mask | mask
    return combined_mask


def my_region_grow_adaptive(image, seed_point, initial_threshold=5, max_threshold=50, step=2):
    """
    自适应区域生长: 逐步放宽阈值直到区域面积稳定
    用于处理灰度渐变区域
    """
    best_mask = None
    best_area = 0
    prev_area = 0

    for thresh in range(initial_threshold, max_threshold + 1, step):
        mask = my_region_grow(image, seed_point, threshold=thresh)
        area = mask.sum()

        # 如果面积增长放缓，说明已到达自然边界
        if prev_area > 0 and area > 0:
            growth_ratio = (area - prev_area) / prev_area
            if growth_ratio > 0.5:  # 突然增长 > 50%
                break

        best_mask = mask
        prev_area = area

    return best_mask if best_mask is not None else np.zeros_like(image, dtype=bool)
