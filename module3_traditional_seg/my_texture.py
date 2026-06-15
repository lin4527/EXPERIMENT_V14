"""
手动实现纹理特征提取 — 实验7
- GLCM 灰度共生矩阵 (4方向: 0°/45°/90°/135°)
- 5种纹理特征: 对比度/能量/熵/逆差分矩/自相关
- LBP 局部二值模式
- Gabor 小波纹理提取
"""
import numpy as np
import math
from scipy.ndimage import convolve


# ===================== GLCM 灰度共生矩阵 =====================

def my_glcm(image, distance=1, angle=0, levels=256, symmetric=True, normed=True):
    """
    手动实现灰度共生矩阵 (GLCM)
    原理: 统计图像中满足特定空间关系的像素对(i,j)出现的频次
    - 对于任意位置 (x,y)，检查其 (x+dx, y+dy) 处的像素
    - 方向角度定义: 0°(1,0), 45°(1,-1), 90°(0,-1), 135°(-1,-1)

    :param image: 灰度图像 (0-255)
    :param distance: 像素对距离
    :param angle: 方向角度 (度), 0°/45°/90°/135°
    :param levels: 灰度级数 (默认256, 可量化减小)
    :param symmetric: 是否生成对称GLCM
    :param normed: 是否归一化
    :return: GLCM 矩阵 (levels × levels)
    """
    img = np.asarray(image, dtype=np.uint8)

    # 可选: 量化到更少灰度级以减少GLCM尺寸
    if levels < 256:
        img = (img / (256 / levels)).astype(np.uint8)

    H, W = img.shape

    # 方向偏移量
    angle_offsets = {
        0:   (0, distance),        # 0° 水平向右
        45:  (-distance, distance), # 45° 右上
        90:  (-distance, 0),       # 90° 垂直向上
        135: (-distance, -distance), # 135° 左上
    }
    dy, dx = angle_offsets.get(angle, (0, distance))

    # 构建 GLCM
    glcm = np.zeros((levels, levels), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            ref_val = img[i, j]
            if ref_val >= levels:
                continue

            # 邻居像素位置
            ni = i + dy
            nj = j + dx
            if 0 <= ni < H and 0 <= nj < W:
                nei_val = img[ni, nj]
                if nei_val < levels:
                    glcm[ref_val, nei_val] += 1

    if symmetric:
        glcm = glcm + glcm.T

    if normed and glcm.sum() > 0:
        glcm = glcm / glcm.sum()

    return glcm


def my_glcm_features(glcm):
    """
    从 GLCM 提取 5 种 Haralick 纹理特征
    :param glcm: 归一化后的 GLCM 矩阵
    :return: dict {contrast, energy, entropy, idm, correlation}
    """
    glcm = np.asarray(glcm, dtype=np.float64)
    N = glcm.shape[0]

    # 辅助向量
    i = np.arange(N)
    j = np.arange(N)
    I, J = np.meshgrid(i, j, indexing='ij')

    # 边际概率
    p_i = np.sum(glcm, axis=1)  # 行和
    p_j = np.sum(glcm, axis=0)  # 列和

    # 均值
    mu_i = np.sum(I[:, 0] * p_i)
    mu_j = np.sum(J[0, :] * p_j)

    # 标准差
    sigma_i = np.sqrt(np.sum((I[:, 0] - mu_i)**2 * p_i) + 1e-10)
    sigma_j = np.sqrt(np.sum((J[0, :] - mu_j)**2 * p_j) + 1e-10)

    # 1) 对比度 (Contrast): Σ(i-j)² × p(i,j)
    contrast = np.sum((I - J)**2 * glcm)

    # 2) 能量 (Energy / Angular Second Moment): Σ p(i,j)²
    energy = np.sum(glcm**2)

    # 3) 熵 (Entropy): -Σ p(i,j) × log₂(p(i,j))
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

    # 4) 逆差分矩 (IDM / Homogeneity): Σ p(i,j) / (1 + (i-j)²)
    idm = np.sum(glcm / (1 + (I - J)**2))

    # 5) 自相关 (Correlation): Σ ((i-μi)(j-μj) × p(i,j)) / (σi × σj)
    correlation = np.sum((I - mu_i) * (J - mu_j) * glcm) / (sigma_i * sigma_j + 1e-10)

    return {
        "contrast": float(contrast),
        "energy": float(energy),
        "entropy": float(entropy),
        "idm": float(idm),
        "correlation": float(correlation),
    }


def my_glcm_4direction_features(image, distance=1, levels=256):
    """
    计算四个方向 (0°/45°/90°/135°) 的 GLCM 特征并取平均
    """
    angles = [0, 45, 90, 135]
    all_features = []
    for angle in angles:
        glcm = my_glcm(image, distance=distance, angle=angle, levels=levels)
        feat = my_glcm_features(glcm)
        all_features.append(feat)

    # 各特征取平均
    avg_features = {}
    for key in all_features[0]:
        avg_features[key] = float(np.mean([f[key] for f in all_features]))

    return avg_features


# ===================== LBP 局部二值模式 =====================

def my_lbp(image, radius=1, n_points=8):
    """
    手动实现 LBP (Local Binary Pattern)
    原理: 对每个像素，比较其邻域像素与中心像素的灰度值，生成二进制模式
    LBP = Σ(s(g_p - g_c) * 2^p)，其中 s(x) = 1 if >= 0 else 0

    :param image: 灰度图像
    :param radius: 邻域半径
    :param n_points: 邻域采样点数
    :return: LBP 特征图 (值范围 0~2^n_points - 1)
    """
    img = np.asarray(image, dtype=np.float64)
    H, W = img.shape
    result = np.zeros((H, W), dtype=np.uint8)

    # 预计算采样点偏移
    offsets = []
    for p in range(n_points):
        theta = 2 * np.pi * p / n_points
        dy = -radius * math.sin(theta)
        dx = radius * math.cos(theta)
        offsets.append((dy, dx))

    for i in range(radius, H - radius):
        for j in range(radius, W - radius):
            center_val = img[i, j]
            code = 0
            for p, (dy, dx) in enumerate(offsets):
                # 双线性插值采样邻域像素
                ni = i + dy
                nj = j + dx
                ni_int = int(math.floor(ni))
                nj_int = int(math.floor(nj))
                ni_frac = ni - ni_int
                nj_frac = nj - nj_int

                if radius == 1 and n_points == 8:
                    # 标准 LBP: 8邻域直接用整数坐标
                    ni_r = int(round(ni))
                    nj_r = int(round(nj))
                    nei_val = img[max(0, min(H-1, ni_r)), max(0, min(W-1, nj_r))]
                else:
                    # 双线性插值
                    try:
                        a = img[ni_int, nj_int]
                        b = img[min(ni_int + 1, H - 1), nj_int]
                        c = img[ni_int, min(nj_int + 1, W - 1)]
                        d = img[min(ni_int + 1, H - 1), min(nj_int + 1, W - 1)]
                        nei_val = (a * (1 - ni_frac) * (1 - nj_frac) +
                                   b * ni_frac * (1 - nj_frac) +
                                   c * (1 - ni_frac) * nj_frac +
                                   d * ni_frac * nj_frac)
                    except IndexError:
                        nei_val = center_val

                if nei_val >= center_val:
                    code |= (1 << p)

            result[i, j] = code

    return result


def my_lbp_histogram(lbp_image, n_bins=256):
    """计算 LBP 直方图作为特征向量"""
    lbp = np.asarray(lbp_image, dtype=np.uint8).flatten()
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    # 归一化
    hist = hist.astype(np.float64)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


# ===================== Gabor 小波 =====================

def my_gabor_filter(image, theta=0, frequency=0.1, sigma_x=4.0, sigma_y=4.0,
                     phase_offset=0, size=None):
    """
    Gabor 小波滤波器
    原理: Gabor 核 = 高斯包络 × 正弦平面波
    g(x,y) = exp(-(x'²/(2σx²) + y'²/(2σy²))) × cos(2πfx' + φ)
    其中 x' = x×cos(θ) + y×sin(θ), y' = -x×sin(θ) + y×cos(θ)

    :param image: 输入灰度图像
    :param theta: Gabor方向 (弧度), 控制滤波器朝向
    :param frequency: 正弦波频率, 控制纹理尺度
    :param sigma_x, sigma_y: 高斯包络标准差
    :param phase_offset: 相位偏移
    :param size: 核大小
    :return: 滤波响应图
    """
    img = np.asarray(image, dtype=np.float64)

    # 确定核大小
    if size is None:
        size = int(2 * math.ceil(3 * max(sigma_x, sigma_y)) + 1)
    if size % 2 == 0:
        size += 1
    center = size // 2

    # 生成 Gabor 核
    kernel = np.zeros((size, size), dtype=np.float64)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    for i in range(size):
        for j in range(size):
            x = j - center
            y = i - center
            x_rot = x * cos_theta + y * sin_theta
            y_rot = -x * sin_theta + y * cos_theta

            # 高斯包络 × 正弦波
            gaussian = math.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))
            sinusoid = math.cos(2 * np.pi * frequency * x_rot + phase_offset)
            kernel[i, j] = gaussian * sinusoid

    # 去直流分量
    kernel = kernel - np.mean(kernel)

    # 卷积
    from scipy.ndimage import convolve as scipy_convolve
    response = scipy_convolve(img, kernel, mode='reflect')

    return response


def my_gabor_bank(image, frequencies=[0.05, 0.1, 0.2], orientations=[0, 45, 90, 135]):
    """
    Gabor 滤波器组 — 多尺度多方向纹理提取
    :return: 所有滤波响应列表
    """
    responses = []
    for f in frequencies:
        for angle_deg in orientations:
            theta = math.radians(angle_deg)
            resp = my_gabor_filter(image, theta=theta, frequency=f)
            responses.append({
                "frequency": f,
                "orientation": angle_deg,
                "response": resp,
            })
    return responses
