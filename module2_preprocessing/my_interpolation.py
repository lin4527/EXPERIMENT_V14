"""
手动实现插值与重采样算法 — 实验2
- 最近邻插值 (Nearest Neighbor)
- 双线性插值 (Bilinear Interpolation)
- 图像重采样
- 灰度级量化调整
"""
import numpy as np
import math


def my_nearest_neighbor(image, new_shape):
    """
    最近邻插值
    原理: 对每个目标像素，找到在源图像中最近的原像素，直接复制其灰度值
    :param image: 源图像 (H, W)
    :param new_shape: 目标尺寸 (new_H, new_W)
    :return: 插值后的图像
    """
    src = np.asarray(image, dtype=np.float64)
    src_h, src_w = src.shape
    dst_h, dst_w = new_shape

    # 计算缩放比例
    scale_h = src_h / dst_h
    scale_w = src_w / dst_w

    result = np.zeros((dst_h, dst_w), dtype=src.dtype)

    for i in range(dst_h):
        for j in range(dst_w):
            # 计算源图像对应坐标
            src_i = int((i + 0.5) * scale_h - 0.5)
            src_j = int((j + 0.5) * scale_w - 0.5)
            # 边界裁切
            src_i = max(0, min(src_i, src_h - 1))
            src_j = max(0, min(src_j, src_w - 1))
            result[i, j] = src[src_i, src_j]

    return result


def my_bilinear_interpolation(image, new_shape):
    """
    双线性插值
    原理:
    1. 找到目标像素在源图像中对应坐标 (x, y)
    2. 找到周围4个邻域像素 Q11(x1,y1), Q21(x2,y1), Q12(x1,y2), Q22(x2,y2)
    3. 先在 x 方向线性插值, 再在 y 方向线性插值:
       f(x,y) = (1-alpha)(1-beta)*f(x1,y1) + alpha(1-beta)*f(x2,y1)
              + (1-alpha)*beta*f(x1,y2) + alpha*beta*f(x2,y2)
       其中 alpha = x - x1, beta = y - y1
    :param image: 源图像 (H, W)
    :param new_shape: 目标尺寸 (new_H, new_W)
    :return: 双线性插值结果
    """
    src = np.asarray(image, dtype=np.float64)
    src_h, src_w = src.shape
    dst_h, dst_w = new_shape

    scale_h = src_h / dst_h
    scale_w = src_w / dst_w

    result = np.zeros((dst_h, dst_w), dtype=np.float64)

    for i in range(dst_h):
        for j in range(dst_w):
            # 源图像浮点坐标 (中心对齐)
            src_y = (i + 0.5) * scale_h - 0.5
            src_x = (j + 0.5) * scale_w - 0.5

            # 4个最近邻整数坐标
            x1 = int(math.floor(src_x))
            y1 = int(math.floor(src_y))
            x2 = x1 + 1
            y2 = y1 + 1

            # 截断
            x1 = max(0, min(x1, src_w - 1))
            x2 = max(0, min(x2, src_w - 1))
            y1 = max(0, min(y1, src_h - 1))
            y2 = max(0, min(y2, src_h - 1))

            # 插值权重
            alpha = src_x - x1
            beta = src_y - y1

            # 双线性插值公式
            val = ((1 - alpha) * (1 - beta) * src[y1, x1] +
                   alpha * (1 - beta) * src[y1, x2] +
                   (1 - alpha) * beta * src[y2, x1] +
                   alpha * beta * src[y2, x2])

            result[i, j] = val

    return result


def my_image_resample(image, scale_factor, method='bilinear'):
    """
    图像重采样 (空间分辨率调整)
    :param image: 源图像
    :param scale_factor: 缩放因子 (>1 放大, <1 缩小)
    :param method: 'nearest' 或 'bilinear'
    :return: 重采样后的图像
    """
    src = np.asarray(image, dtype=np.float64)
    src_h, src_w = src.shape
    new_h = int(src_h * scale_factor)
    new_w = int(src_w * scale_factor)
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    if method == 'nearest':
        return my_nearest_neighbor(src, (new_h, new_w))
    else:
        return my_bilinear_interpolation(src, (new_h, new_w))


def my_grayscale_quantize(image, levels=256):
    """
    灰度级量化调整
    原理: 将连续灰度值映射到有限的 levels 个灰度级
    s = round(r / step) * step, 其中 step = 256 / levels
    :param image: 输入图像 (0-255)
    :param levels: 目标灰度级数 (2, 4, 8, 16, 32, 64, 128, 256)
    :return: 量化后的图像
    """
    img = np.asarray(image, dtype=np.float64)
    # 归一化到 [0, 1]
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-10)
    # 量化
    step = 1.0 / (levels - 1) if levels > 1 else 1.0
    quantized = np.round(img_norm / step) * step
    # 反归一化
    result = quantized * (img.max() - img.min()) + img.min()
    return result
