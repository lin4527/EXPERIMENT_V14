"""
手动实现灰度变换算法 — 实验3
- 线性灰度变换 (斜率/截距可调)
- 对数变换
- Gamma 幂次变换
- 分段线性变换
所有函数名以 my_ 开头，不直接调用库的现成变换函数
"""
import numpy as np


def my_linear_transform(image, slope=1.0, intercept=0.0):
    """
    线性灰度变换: output = slope * input + intercept
    :param image: 输入图像 (numpy array, 支持任意范围)
    :param slope: 斜率 (对比度调节)
    :param intercept: 截距 (亮度调节)
    :return: 变换后的图像 (保持原范围)
    """
    img = np.asarray(image, dtype=np.float64)
    result = slope * img + intercept
    # 保持原始数据范围
    result = np.clip(result, img.min(), img.max())
    return result


def my_log_transform(image, c=1.0):
    """
    对数变换: s = c * log(1 + r)
    压缩高灰度值，扩展低灰度值 — 适用于增强暗部细节
    原理: 对数函数对低值区斜率大（增强），高值区斜率小（压缩）
    :param image: 输入图像
    :param c: 尺度常数
    :return: 变换后的图像
    """
    img = np.asarray(image, dtype=np.float64)
    # 确保非负
    img_min = img.min()
    if img_min < 0:
        img = img - img_min
    result = c * np.log1p(img)  # log1p(x) = log(1+x)，数值稳定
    return result


def my_gamma_transform(image, gamma=1.0, c=1.0):
    """
    Gamma 幂次变换: s = c * r^gamma
    原理:
    - gamma < 1: 扩展暗部，压缩亮部（增强暗区）
    - gamma > 1: 压缩暗部，扩展亮部（增强亮区）
    - gamma = 1: 线性恒等变换
    :param image: 输入图像
    :param gamma: 幂次指数
    :param c: 尺度常数
    :return: 变换后的图像
    """
    img = np.asarray(image, dtype=np.float64)
    # 归一化到 [0, 1]
    img_min = img.min()
    img_max = img.max()
    if img_max == img_min:
        return img.copy()
    img_norm = (img - img_min) / (img_max - img_min)
    # Gamma 校正
    result_norm = c * np.power(img_norm, gamma)
    # 反归一化
    result = result_norm * (img_max - img_min) + img_min
    return result


def my_piecewise_linear_transform(image, breakpoints):
    """
    分段线性变换
    原理: 在不同灰度区间使用不同的线性映射，实现对比度拉伸、灰度切割等效果
    :param image: 输入图像
    :param breakpoints: 断点列表 [(in1, out1), (in2, out2), ...]
                        自动补齐 (0,0) 和 (max, max)
    :return: 变换后的图像
    """
    img = np.asarray(image, dtype=np.float64)
    img_min, img_max = img.min(), img.max()

    # 构建完整断点表 (包含起点和终点)
    pts = sorted(breakpoints, key=lambda x: x[0])
    if len(pts) == 0 or pts[0][0] > img_min:
        pts.insert(0, (img_min, img_min))
    if pts[-1][0] < img_max:
        pts.append((img_max, img_max))

    result = np.zeros_like(img)
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        mask = (img >= x0) & (img <= x1)
        if x1 != x0:
            result[mask] = y0 + (img[mask] - x0) * (y1 - y0) / (x1 - x0)
        else:
            result[mask] = y0

    return result


def my_contrast_stretch(image, low_percent=2, high_percent=98):
    """
    对比度拉伸: 基于百分位数的自动分段线性变换
    将 [low_percent%, high_percent%] 映射到整个显示范围
    """
    img = np.asarray(image, dtype=np.float64)
    low_val = np.percentile(img, low_percent)
    high_val = np.percentile(img, high_percent)
    if high_val == low_val:
        return img
    result = (img - low_val) / (high_val - low_val) * (img.max() - img.min()) + img.min()
    result = np.clip(result, img.min(), img.max())
    return result
