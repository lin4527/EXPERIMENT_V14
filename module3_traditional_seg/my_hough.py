"""
手动实现霍夫直线检测 — 实验6扩展
用于椎体终板检测
原理: 将图像空间 (x,y) 直线映射到参数空间 (ρ,θ) 的峰值检测
"""
import numpy as np
import math


def my_hough_lines(edge_image, theta_res=1.0, rho_res=1.0, threshold=50):
    """
    霍夫直线检测 (标准霍夫变换 SHT)
    原理:
    1. 边缘图像二值化 (边缘=255, 背景=0)
    2. 对每条边缘像素 (x,y)，计算所有 θ 下的 ρ = x*cos(θ) + y*sin(θ)
    3. 在 (ρ,θ) 累加器网格上投票
    4. 寻找超过阈值的峰值 → 对应直线

    :param edge_image: 二值边缘图
    :param theta_res: θ 分辨率 (度)
    :param rho_res: ρ 分辨率 (像素)
    :param threshold: 投票阈值
    :return: [(rho, theta, votes), ...] 检测到的直线
    """
    edges = np.asarray(edge_image, dtype=np.uint8)
    H, W = edges.shape

    # 边缘像素坐标
    y_idx, x_idx = np.where(edges > 0)
    n_edges = len(x_idx)
    if n_edges == 0:
        return []

    # θ 范围: 0° ~ 180° (弧度)
    theta_vals = np.deg2rad(np.arange(0, 180, theta_res))
    n_theta = len(theta_vals)

    # ρ 范围: -D ~ D, D = sqrt(H² + W²)
    D = int(math.sqrt(H**2 + W**2))
    n_rho = int(2 * D / rho_res)

    # 累加器
    accumulator = np.zeros((n_rho, n_theta), dtype=np.int64)

    # cos/sin 预计算
    cos_vals = np.cos(theta_vals)
    sin_vals = np.sin(theta_vals)

    # 投票
    for k in range(n_edges):
        x = x_idx[k]
        y = y_idx[k]
        for t_idx in range(n_theta):
            rho_val = x * cos_vals[t_idx] + y * sin_vals[t_idx]
            rho_idx = int((rho_val + D) / rho_res)
            if 0 <= rho_idx < n_rho:
                accumulator[rho_idx, t_idx] += 1

    # 峰值检测
    lines = []
    for r_idx in range(1, n_rho - 1):
        for t_idx in range(1, n_theta - 1):
            votes = accumulator[r_idx, t_idx]
            if votes >= threshold:
                # 检查是否为局部最大值
                local_patch = accumulator[r_idx-1:r_idx+2, t_idx-1:t_idx+2]
                if votes == local_patch.max():
                    rho = r_idx * rho_res - D
                    theta = theta_vals[t_idx]
                    lines.append((float(rho), float(theta), int(votes)))

    # 按投票数降序排列
    lines.sort(key=lambda x: x[2], reverse=True)
    return lines


def my_hough_lines_probabilistic(edge_image, threshold=50, min_line_length=30, max_line_gap=10):
    """
    概率霍夫直线检测 (简化版)
    随机采样边缘点，减少计算量
    """
    edges = np.asarray(edge_image, dtype=np.uint8)
    y_idx, x_idx = np.where(edges > 0)
    n_edges = len(x_idx)
    if n_edges < threshold:
        return []

    # 随机采样
    sample_size = min(n_edges, 5000)
    indices = np.random.choice(n_edges, sample_size, replace=False)

    theta_vals = np.deg2rad(np.arange(0, 180, 1))
    D = int(math.sqrt(edges.shape[0]**2 + edges.shape[1]**2))

    accumulator = np.zeros((2 * D + 1, 180), dtype=np.int64)

    for idx in indices:
        x = x_idx[idx]
        y = y_idx[idx]
        for t_idx in range(180):
            rho = x * math.cos(theta_vals[t_idx]) + y * math.sin(theta_vals[t_idx])
            rho_idx = int(round(rho)) + D
            if 0 <= rho_idx < 2 * D + 1:
                accumulator[rho_idx, t_idx] += 1

    # 阈值过滤
    peaks = np.where(accumulator >= threshold)
    lines = []
    for r_idx, t_idx in zip(peaks[0], peaks[1]):
        rho = float(r_idx - D)
        theta = float(theta_vals[t_idx])
        lines.append((rho, theta, int(accumulator[r_idx, t_idx])))

    lines.sort(key=lambda x: x[2], reverse=True)
    return lines[:100]


def draw_hough_lines(image, lines, color=255):
    """
    在图像上绘制霍夫检测到的直线
    :param image: 原图像
    :param lines: 霍夫检测结果 [(rho, theta, votes), ...]
    :return: 绘制后的图像
    """
    import cv2
    img_arr = np.asarray(image)
    # 确保 uint8 深度 (cv2.cvtColor 不支持 float64)
    if img_arr.dtype == np.float64 or img_arr.dtype == np.float32:
        if img_arr.max() <= 1.0:
            img_arr = (img_arr * 255).astype(np.uint8)
        else:
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    if len(img_arr.shape) == 2:
        result = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
    else:
        result = img_arr.copy()
    H, W = result.shape[:2]

    for rho, theta, votes in lines[:20]:  # 仅画前20条最强的线
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return result
