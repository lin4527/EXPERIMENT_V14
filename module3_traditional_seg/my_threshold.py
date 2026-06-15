"""
手动实现阈值分割算法 — 实验8
- 迭代阈值法
- Otsu 最大类间方差法
"""
import numpy as np


def my_iterative_threshold(image, initial_threshold=None, max_iterations=100, tolerance=1.0):
    """
    迭代阈值法
    原理:
    1. 选择初始阈值 T0 (通常为图像均值)
    2. 用 T 将图像分为两组 G1(≤T) 和 G2(>T)
    3. 计算两组均值 μ1, μ2
    4. 更新 T = (μ1 + μ2) / 2
    5. 重复步骤2-4直到 T 收敛 (|T_new - T_old| < tolerance)

    :param image: 灰度图像
    :param initial_threshold: 初始阈值, None则使用图像均值
    :param max_iterations: 最大迭代次数
    :param tolerance: 收敛容差
    :return: (二值图像, 最终阈值)
    """
    img = np.asarray(image, dtype=np.float64)
    img_flat = img.flatten()

    # 初始阈值
    if initial_threshold is None:
        T = np.mean(img_flat)
    else:
        T = float(initial_threshold)

    for iteration in range(max_iterations):
        # 分组
        group1 = img_flat[img_flat <= T]
        group2 = img_flat[img_flat > T]

        if len(group1) == 0 or len(group2) == 0:
            break

        # 计算各组均值
        m1 = np.mean(group1)
        m2 = np.mean(group2)

        # 更新阈值
        T_new = (m1 + m2) / 2.0

        if abs(T_new - T) < tolerance:
            T = T_new
            break

        T = T_new

    # 二值化
    binary = (img > T).astype(np.uint8) * 255
    return binary, T


def my_otsu(image):
    """
    Otsu 最大类间方差法 (大津算法)
    原理:
    1. 计算灰度直方图和概率分布
    2. 遍历所有可能的阈值 t
    3. 对每个 t，计算类间方差 σ²_B = ω0×ω1×(μ0-μ1)²
       其中 ω0, ω1 是两类出现的概率
             μ0, μ1 是两类的均值
    4. 选择使 σ²_B 最大的 t 作为最优阈值

    优化: 使用累积矩避免重复计算

    :param image: 灰度图像 (0-255)
    :return: (二值图像, 最优阈值)
    """
    img = np.asarray(image, dtype=np.uint8)
    H, W = img.shape
    total = H * W

    # Step 1: 计算直方图
    hist = np.zeros(256, dtype=np.float64)
    for i in range(H):
        for j in range(W):
            hist[img[i, j]] += 1

    # 概率分布
    prob = hist / total

    # Step 2: 累积和与累积均值 (加速计算)
    # omega[k] = Σ(i=0..k) prob[i]  (累积概率)
    # mu_cum[k] = Σ(i=0..k) i * prob[i]  (累积均值分量)
    omega = np.cumsum(prob)
    mu_cum = np.cumsum(np.arange(256) * prob)
    mu_total = mu_cum[-1]

    # Step 3: 遍历所有阈值, 计算类间方差
    sigma_b_sq = np.zeros(256)
    for t in range(256):
        w0 = omega[t]  # 背景概率
        w1 = 1.0 - w0  # 前景概率

        if w0 == 0 or w1 == 0:
            sigma_b_sq[t] = 0
            continue

        mu0 = mu_cum[t] / w0  # 背景均值
        mu1 = (mu_total - mu_cum[t]) / w1  # 前景均值

        # 类间方差 σ²_B = w0 * w1 * (μ0 - μ1)²
        sigma_b_sq[t] = w0 * w1 * (mu0 - mu1) ** 2

    # Step 4: 最优阈值 = argmax σ²_B
    best_T = np.argmax(sigma_b_sq)

    # 二值化
    binary = (img > best_T).astype(np.uint8) * 255

    return binary, int(best_T)
