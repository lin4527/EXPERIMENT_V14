"""
手动实现色彩空间转换与伪彩色 — 实验11
- RGB ↔ HSI 色彩空间转换
- 灰度转伪彩色映射
- 彩色图像通道分离与单通道滤波
"""
import numpy as np
import math


# ===================== RGB ↔ HSI 色彩空间转换 =====================

def my_rgb_to_hsi(rgb_image):
    """
    手动实现 RGB → HSI 转换
    原理:
    - Intensity (I): I = (R+G+B)/3
    - Saturation (S): S = 1 - 3*min(R,G,B)/(R+G+B)
    - Hue (H): 基于 RGB 分量相对于参考轴的角度

    算法步骤:
    1. 归一化 RGB 到 [0,1]
    2. 计算 I (亮度) = 均值
    3. 计算 S (饱和度) = 1 - min/(R+G+B)
    4. 计算 H (色调) = arccos(...)，根据 G vs B 调整

    :param rgb_image: RGB 图像 (H, W, 3), uint8 [0,255]
    :return: HSI 图像 (H, W, 3), H∈[0,2π], S∈[0,1], I∈[0,1]
    """
    img = np.asarray(rgb_image, dtype=np.float64) / 255.0
    H, W = img.shape[:2]

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Intensity
    I = (R + G + B) / 3.0

    # Saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    sum_rgb = R + G + B + 1e-10
    S = 1.0 - 3.0 * min_rgb / sum_rgb

    # Hue (使用标准公式)
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-10
    theta = np.arccos(np.clip(numerator / denominator, -1, 1))
    H = np.where(B <= G, theta, 2 * np.pi - theta)

    return np.stack([H, S, I], axis=2)


def my_hsi_to_rgb(hsi_image):
    """
    手动实现 HSI → RGB 转换
    原理: 根据 H 所在扇区 (0°, 120°, 240°)，分别使用不同的变换公式
    :param hsi_image: HSI 图像 (H, W, 3), H∈[0,2π], S∈[0,1], I∈[0,1]
    :return: RGB 图像 (H, W, 3), uint8 [0,255]
    """
    H = hsi_image[:, :, 0].copy()
    S = hsi_image[:, :, 1]
    I = hsi_image[:, :, 2]

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    # RG 扇区 (0° ≤ H < 120°)
    mask = (H >= 0) & (H < 2 * np.pi / 3)
    H_m = H[mask]
    B[mask] = I[mask] * (1 - S[mask])
    R[mask] = I[mask] * (1 + S[mask] * np.cos(H_m) / np.cos(np.pi / 3 - H_m))
    G[mask] = 3 * I[mask] - (R[mask] + B[mask])

    # GB 扇区 (120° ≤ H < 240°)
    mask = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)
    H_m = H[mask] - 2 * np.pi / 3
    R[mask] = I[mask] * (1 - S[mask])
    G[mask] = I[mask] * (1 + S[mask] * np.cos(H_m) / np.cos(np.pi / 3 - H_m))
    B[mask] = 3 * I[mask] - (R[mask] + G[mask])

    # BR 扇区 (240° ≤ H < 360°)
    mask = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)
    H_m = H[mask] - 4 * np.pi / 3
    G[mask] = I[mask] * (1 - S[mask])
    B[mask] = I[mask] * (1 + S[mask] * np.cos(H_m) / np.cos(np.pi / 3 - H_m))
    R[mask] = 3 * I[mask] - (G[mask] + B[mask])

    # 灰度无色调情况 (S ≈ 0)
    gray_mask = S < 0.01
    R[gray_mask] = I[gray_mask]
    G[gray_mask] = I[gray_mask]
    B[gray_mask] = I[gray_mask]

    rgb = np.stack([R, G, B], axis=2)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


# ===================== 伪彩色映射 =====================

def my_pseudocolor_map(gray_image, colormap='jet'):
    """
    灰度转伪彩色映射
    原理: 将灰度值 [0,255] 通过颜色查找表映射为 RGB 彩色
    手动实现常见的 colormap 方案
    :param gray_image: 灰度图像 (H, W)
    :param colormap: 'jet', 'hot', 'cool', 'bone', 'rainbow'
    :return: RGB 伪彩色图像 (H, W, 3)
    """
    img = np.asarray(gray_image, dtype=np.float64)
    if img.max() > 1:
        img = img / 255.0
    img = np.clip(img, 0, 1)

    H, W = img.shape

    if colormap == 'jet':
        # Jet: 蓝→青→绿→黄→红
        R, G, B = _jet_colormap(img)
    elif colormap == 'hot':
        # Hot: 黑→红→黄→白
        R = np.clip(img / 0.375, 0, 1)
        G = np.clip((img - 0.375) / 0.375, 0, 1)
        B = np.clip((img - 0.75) / 0.25, 0, 1)
    elif colormap == 'cool':
        R = img
        G = 1.0 - img
        B = np.ones_like(img)
    elif colormap == 'bone':
        # Bone: 黑→灰→白 (带蓝调)
        R = (img * 0.85 + 0.15)
        G = (img * 0.85 + 0.15)
        B = (img * 0.7 + 0.3)
    elif colormap == 'rainbow':
        R, G, B = _rainbow_colormap(img)
    else:
        R = G = B = img

    rgb = np.stack([R, G, B], axis=2)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _jet_colormap(t):
    """Jet colormap 的分段插值实现"""
    # 蓝→青 (0→0.25)
    # 青→绿 (0.25→0.5)
    # 绿→黄 (0.5→0.75)
    # 黄→红 (0.75→1.0)
    R = np.clip(np.where(t < 0.5,
                         np.where(t < 0.25, 0, (t - 0.25) * 4),
                         np.where(t < 0.75, 1, (1 - t) * 4 + 1)), 0, 1)
    G = np.clip(np.where(t < 0.25, t * 4,
                         np.where(t < 0.75, 1,
                                  (1 - t) * 4)), 0, 1)
    B = np.clip(np.where(t < 0.25, 1,
                         np.where(t < 0.5, (0.5 - t) * 4 + 1,
                                  np.where(t < 0.75, 0, 0))), 0, 1)
    return R, G, B


def _rainbow_colormap(t):
    """Rainbow colormap: 利用正弦函数生成彩虹色"""
    R = np.clip(np.sin(t * 2 * np.pi - np.pi / 2) * 0.5 + 0.5, 0, 1)
    G = np.clip(np.sin(t * 2 * np.pi + np.pi / 6) * 0.5 + 0.5, 0, 1)
    B = np.clip(np.sin(t * 2 * np.pi + 5 * np.pi / 6) * 0.5 + 0.5, 0, 1)
    return R, G, B


# ===================== 彩色图像通道操作 =====================

def my_channel_separate(color_image):
    """
    彩色图像通道分离
    :param color_image: RGB 图像 (H, W, 3)
    :return: dict {'R': ..., 'G': ..., 'B': ...}
    """
    img = np.asarray(color_image)
    return {
        'R': img[:, :, 0],
        'G': img[:, :, 1],
        'B': img[:, :, 2],
    }


def my_channel_filter(color_image, channel='R', filter_func=None):
    """
    单通道滤波：提取某通道 → 滤波 → 替换回原图
    :param color_image: RGB 图像
    :param channel: 'R', 'G', 'B'
    :param filter_func: 滤波函数 (接受单通道图像, 返回同尺寸结果)
    :return: 滤波后的 RGB 图像
    """
    img = np.asarray(color_image, dtype=np.float64).copy()
    ch_map = {'R': 0, 'G': 1, 'B': 2}
    ch_idx = ch_map.get(channel, 0)

    if filter_func is not None:
        img[:, :, ch_idx] = filter_func(img[:, :, ch_idx])

    return img


def my_channel_merge(channels_dict):
    """通道融合: 将分离的 R,G,B 通道合并为彩色图像"""
    return np.stack([channels_dict['R'], channels_dict['G'], channels_dict['B']], axis=2)
