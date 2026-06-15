"""
手动实现边缘检测算子 — 实验6
- 一阶: Roberts, Sobel, Prewitt, Canny
- 二阶: Laplacian, LoG (Laplacian of Gaussian)
所有卷积核手动定义，卷积操作手动实现
"""
import numpy as np
import math
from scipy.ndimage import gaussian_filter


def _conv2d(image, kernel):
    """
    手动实现 2D 卷积 (valid 模式)
    :param image: 2D 数组
    :param kernel: 卷积核 (2D)
    :return: 卷积结果
    """
    img = np.asarray(image, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    kh, kw = k.shape
    ih, iw = img.shape
    out_h = ih - kh + 1
    out_w = iw - kw + 1
    result = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            result[i, j] = np.sum(img[i:i+kh, j:j+kw] * k)
    return result


def _conv2d_same(image, kernel):
    """same 模式卷积 (输出尺寸=输入尺寸), 使用零填充"""
    img = np.asarray(image, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    kh, kw = k.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)
    return _conv2d(padded, k)


def _gradient_magnitude(gx, gy):
    """计算梯度幅值"""
    return np.sqrt(gx.astype(np.float64)**2 + gy.astype(np.float64)**2)


def _gradient_direction(gx, gy):
    """计算梯度方向 (弧度)"""
    return np.arctan2(gy, gx)


# ===================== 一阶边缘检测 =====================

def my_roberts(image):
    """
    Roberts 交叉梯度算子
    使用 2×2 对角线差分核:
    Gx = [[1, 0], [0, -1]]
    Gy = [[0, 1], [-1, 0]]
    特点: 简单快速，但对噪声敏感
    """
    gx_kernel = np.array([[1, 0], [0, -1]], dtype=np.float64)
    gy_kernel = np.array([[0, 1], [-1, 0]], dtype=np.float64)

    gx = _conv2d(image, gx_kernel)
    gy = _conv2d(image, gy_kernel)
    return _gradient_magnitude(gx, gy)


def my_sobel(image):
    """
    Sobel 算子 (3×3)
    核定义:
    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Gy = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]
    特点: 结合了高斯平滑和差分，对噪声有一定抑制
    """
    gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    gx = _conv2d_same(image, gx_kernel)
    gy = _conv2d_same(image, gy_kernel)
    return _gradient_magnitude(gx, gy)


def my_prewitt(image):
    """
    Prewitt 算子 (3×3)
    核定义:
    Gx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    Gy = [[-1,-1,-1], [ 0, 0, 0], [ 1, 1, 1]]
    特点: 与 Sobel 类似，但无中心权重
    """
    gx_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    gy_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)

    gx = _conv2d_same(image, gx_kernel)
    gy = _conv2d_same(image, gy_kernel)
    return _gradient_magnitude(gx, gy)


def my_canny(image, low_threshold=50, high_threshold=150, sigma=1.4):
    """
    Canny 边缘检测 完整四步流程
    1. 高斯平滑
    2. 计算梯度幅值和方向 (Sobel)
    3. 非极大值抑制 (NMS)
    4. 双阈值检测 + 边缘连接
    :param image: 灰度图像
    :param low_threshold: 低阈值
    :param high_threshold: 高阈值
    :param sigma: 高斯平滑 σ
    :return: 二值边缘图 (0/255)
    """
    img = np.asarray(image, dtype=np.float64)

    # Step 1: 高斯平滑
    smoothed = gaussian_filter(img, sigma=sigma)

    # Step 2: 梯度计算 (Sobel)
    gx = _conv2d_same(smoothed, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64))
    gy = _conv2d_same(smoothed, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64))
    magnitude = _gradient_magnitude(gx, gy)
    direction = _gradient_direction(gx, gy)

    # Step 3: 非极大值抑制
    # 将梯度方向量化到 4 个方向: 0°, 45°, 90°, 135°
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    H, W = magnitude.shape
    nms = np.zeros((H, W), dtype=np.float64)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = angle[i, j]
            mag = magnitude[i, j]

            # 方向 0° (水平边缘)
            if (a < 22.5 or a >= 157.5):
                q = magnitude[i, j + 1] if j + 1 < W else 0
                r = magnitude[i, j - 1] if j - 1 >= 0 else 0
            # 方向 45°
            elif 22.5 <= a < 67.5:
                q = magnitude[i + 1, j - 1] if (i + 1 < H and j - 1 >= 0) else 0
                r = magnitude[i - 1, j + 1] if (i - 1 >= 0 and j + 1 < W) else 0
            # 方向 90° (垂直边缘)
            elif 67.5 <= a < 112.5:
                q = magnitude[i + 1, j] if i + 1 < H else 0
                r = magnitude[i - 1, j] if i - 1 >= 0 else 0
            # 方向 135°
            else:
                q = magnitude[i - 1, j - 1] if (i - 1 >= 0 and j - 1 >= 0) else 0
                r = magnitude[i + 1, j + 1] if (i + 1 < H and j + 1 < W) else 0

            if mag >= q and mag >= r:
                nms[i, j] = mag

    # Step 4: 双阈值检测与边缘连接 (滞后阈值)
    strong = np.zeros((H, W), dtype=np.uint8)
    weak = np.zeros((H, W), dtype=np.uint8)

    strong[nms >= high_threshold] = 255
    weak[(nms >= low_threshold) & (nms < high_threshold)] = 128

    # 边缘连接: 与强边缘相邻的弱边缘变为强边缘
    result = strong.copy()
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if weak[i, j] == 128:
                # 检查8邻域是否有强边缘
                if np.any(strong[i-1:i+2, j-1:j+2] == 255):
                    result[i, j] = 255
                else:
                    result[i, j] = 0

    return result


# ===================== 二阶边缘检测 =====================

def my_laplacian(image):
    """
    Laplacian 二阶微分算子 (4邻域)
    核: [[0,  1, 0],
         [1, -4, 1],
         [0,  1, 0]]
    二阶导数为零的位置对应边缘中心
    特点: 对噪声极度敏感，通常需要先平滑
    """
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    lap = _conv2d_same(image, kernel)
    return np.abs(lap)


def my_laplacian_8(image):
    """Laplacian (8邻域增强)"""
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)
    lap = _conv2d_same(image, kernel)
    return np.abs(lap)


def my_log_edge(image, sigma=1.4):
    """
    LoG (Laplacian of Gaussian) 边缘检测
    原理:
    1. 先用高斯滤波平滑: G(x,y) = exp(-(x²+y²)/(2σ²))
    2. 再求 Laplacian: ∇²G
    LoG 核公式: ∇²G(x,y) = -(1/(πσ⁴)) * (1 - (x²+y²)/(2σ²)) * exp(-(x²+y²)/(2σ²))
    零交叉点即为边缘位置
    :param image: 灰度图像
    :param sigma: 高斯标准差
    :return: 边缘检测结果
    """
    img = np.asarray(image, dtype=np.float64)

    # 生成 LoG 核
    size = int(2 * math.ceil(3 * sigma) + 1)
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1

    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    sigma_sq = sigma ** 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            r_sq = x**2 + y**2
            # LoG 公式
            kernel[i, j] = -(1.0 / (np.pi * sigma_sq ** 2)) * \
                           (1.0 - r_sq / (2.0 * sigma_sq)) * \
                           np.exp(-r_sq / (2.0 * sigma_sq))

    # 归一化 (使核的和为零)
    kernel = kernel - np.mean(kernel)

    # 卷积
    log_result = _conv2d_same(img, kernel)

    # 零交叉检测
    H, W = log_result.shape
    edges = np.zeros((H, W), dtype=np.uint8)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            patch = log_result[i-1:i+2, j-1:j+2]
            if patch.max() > 0 and patch.min() < 0:
                edges[i, j] = 255

    return edges
