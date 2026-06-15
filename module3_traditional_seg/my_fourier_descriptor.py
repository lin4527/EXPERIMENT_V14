"""
手动实现傅里叶描绘子 — 实验6扩展
用于分割轮廓的形状描述与重建
原理: 对闭合轮廓的复数坐标序列做 FFT，低频描绘子描述整体形状
"""
import numpy as np
import math


def my_fourier_descriptors(contour, num_descriptors=20):
    """
    计算傅里叶描绘子
    原理:
    1. 将轮廓表示为复数序列 z(k) = x(k) + j*y(k)
    2. 对 z(k) 做 DFT → Z(u)
    3. 取前 num_descriptors 个傅里叶系数作为形状特征
    4. 对描绘子做归一化 (平移/旋转/尺度不变性)

    :param contour: 轮廓点 (N, 2) — (x, y) 或 (row, col)
    :param num_descriptors: 保留的描绘子数量
    :return: 归一化后的大小不变描述子 (复数)
    """
    contour = np.asarray(contour, dtype=np.float64)
    if len(contour) < 3:
        return np.zeros(num_descriptors, dtype=np.complex128)

    # 确保轮廓闭合 (首尾相连)
    N = len(contour)

    # 复数表示
    z = contour[:, 0] + 1j * contour[:, 1]

    # 减去质心 (平移不变性)
    z = z - np.mean(z)

    # DFT (使用 NumPy FFT 做复数DFT — 描绘子不要求手动FFT)
    Z = np.fft.fft(z)

    # 取前 num_descriptors 个系数 (低频 → 形状轮廓)
    fd = Z[:min(num_descriptors, N)]

    # 尺度归一化: 除以第一个非零描绘子的模
    if len(fd) > 0 and abs(fd[0]) > 1e-10:
        fd = fd / abs(fd[0])
    else:
        # 找第一个非零系数
        for i, val in enumerate(fd):
            if abs(val) > 1e-10:
                fd = fd / abs(val)
                break

    # 旋转归一化: 使第一个显著描绘子的相位为零
    if len(fd) > 1:
        # 找第一个非DC的显著描绘子
        for i in range(1, len(fd)):
            if abs(fd[i]) > 0.01:
                # 旋转使得该系数的相位为零
                phase = np.angle(fd[i])
                fd = fd * np.exp(-1j * phase)
                break

    return fd


def my_fd_reconstruct(fd, num_points=256):
    """
    从傅里叶描绘子重建轮廓
    原理: 对傅里叶描绘子补零做 IDFT
    :param fd: 傅里叶描绘子 (复数)
    :param num_points: 重建的点数
    :return: 重建的轮廓点 (num_points, 2)
    """
    fd = np.asarray(fd, dtype=np.complex128)
    K = len(fd)

    # 补零到目标点数
    Z_padded = np.zeros(num_points, dtype=np.complex128)
    Z_padded[:K] = fd

    # IDFT
    z_recon = np.fft.ifft(Z_padded) * num_points

    # 转为 (x, y) 坐标
    contour = np.column_stack([np.real(z_recon), np.imag(z_recon)])
    return contour


def my_fd_similarity(fd1, fd2):
    """
    基于傅里叶描绘子的形状相似度 (欧几里得距离)
    :param fd1, fd2: 傅里叶描绘子
    :return: 距离 (越小越相似)
    """
    # 对齐长度
    min_len = min(len(fd1), len(fd2))
    d1 = fd1[:min_len]
    d2 = fd2[:min_len]
    return float(np.sqrt(np.sum(np.abs(d1 - d2)**2)))
