"""
窗宽窗位调节算法 — 实验12核心
CT图像窗宽(WW)/窗位(WL)调节，预设骨窗/软组织窗/肺窗
"""
import numpy as np
from config import WINDOW_PRESETS


def apply_window_level(image, window_width, window_level):
    """
    应用窗宽窗位调节算法
    :param image: 输入CT图像 (HU值)
    :param window_width: 窗宽 (WW)
    :param window_level: 窗位 (WL)
    :return: 归一化到0-255的uint8图像

    原理：
    - 窗宽决定显示范围，窗位决定显示中心
    - 低于 WL - WW/2 的像素为黑色(0)
    - 高于 WL + WW/2 的像素为白色(255)
    - 中间的像素线性映射到 0-255
    """
    low = window_level - window_width / 2.0
    high = window_level + window_width / 2.0

    image = np.asarray(image, dtype=np.float64)
    # 线性映射 + 截断
    result = (image - low) / (high - low) * 255.0
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def apply_preset(image, preset_name):
    """应用预设窗宽窗位"""
    if preset_name not in WINDOW_PRESETS:
        raise ValueError(f"未知预设: {preset_name}，可选: {list(WINDOW_PRESETS.keys())}")
    ww, wl = WINDOW_PRESETS[preset_name]
    return apply_window_level(image, ww, wl)


def get_window_presets():
    """返回所有预设名称列表"""
    return list(WINDOW_PRESETS.keys())


def auto_window(image):
    """
    自动计算最优窗宽窗位（基于图像直方图1%~99%分位数）
    """
    image = np.asarray(image, dtype=np.float64)
    p1 = np.percentile(image, 1)
    p99 = np.percentile(image, 99)
    ww = p99 - p1
    wl = (p99 + p1) / 2.0
    return int(ww), int(wl)
