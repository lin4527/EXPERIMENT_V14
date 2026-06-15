"""
手动实现二维 FFT/IFFT 与频域滤波 — 实验5
- 2D FFT/IFFT (Cooley-Tukey 基-2 递推算法)
- 频谱中心化 (fftshift/ifftshift)
- 理想/巴特沃斯/高斯 低通+高通滤波器
所有核心算法手动实现，不调用 numpy.fft
"""
import numpy as np
import math


# ===================== 1D FFT 手动实现 =====================

def my_fft1d(x):
    """
    手动实现一维基-2 FFT (Cooley-Tukey 递推)
    原理:
    X(k) = Σ(n=0..N-1) x(n) * W_N^(kn)
    其中 W_N = e^(-j*2π/N)
    递推: X(k) = X_even(k) + W_N^k * X_odd(k)   (k < N/2)
          X(k+N/2) = X_even(k) - W_N^k * X_odd(k)
    :param x: 输入复信号 (长度必须为2的幂)
    :return: FFT结果
    """
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)

    # 确保 N 是 2 的幂 (my_fft2 已预先补齐, 此处仅断言)
    if N & (N - 1) != 0:
        raise ValueError(f"my_fft1d 需要2的幂长度, 得到 {N}. 请先通过 my_fft2 补齐.")

    if N <= 1:
        return x

    # 奇偶分解
    even = my_fft1d(x[0::2])
    odd = my_fft1d(x[1::2])

    # 旋转因子
    factor = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    # 蝶形运算
    left = even + factor * odd
    right = even - factor * odd

    return np.concatenate([left, right])


def my_ifft1d(X):
    """手动实现一维 IFFT (利用共轭性质)"""
    X = np.asarray(X, dtype=np.complex128)
    N = len(X)
    # IFFT: x(n) = (1/N) * conj(FFT(conj(X)))
    conj_X = np.conjugate(X)
    fft_result = my_fft1d(conj_X)
    result = np.conjugate(fft_result) / N
    return result


# ===================== 2D FFT/IFFT =====================

def my_fft2(image):
    """
    手动实现二维 FFT
    原理: 2D FFT = 逐行1D FFT + 逐列1D FFT
    F(u,v) = Σ_x Σ_y f(x,y) * e^(-j2π(ux/M+vy/N))

    处理非2的幂尺寸: 自动补齐到2的幂，FFT后裁剪回原尺寸
    :param image: 2D 图像 (灰度)
    :return: 频域复数结果 (原尺寸)
    """
    img = np.asarray(image, dtype=np.complex128)
    M_orig, N_orig = img.shape

    # 补齐到2的幂
    M_pad = 1 << (M_orig - 1).bit_length() if M_orig & (M_orig - 1) else M_orig
    N_pad = 1 << (N_orig - 1).bit_length() if N_orig & (N_orig - 1) else N_orig

    img_padded = np.zeros((M_pad, N_pad), dtype=np.complex128)
    img_padded[:M_orig, :N_orig] = img

    # Step 1: 逐行 1D FFT
    rows_fft = np.zeros_like(img_padded)
    for i in range(M_pad):
        rows_fft[i, :] = my_fft1d(img_padded[i, :])

    # Step 2: 逐列 1D FFT
    result_padded = np.zeros_like(img_padded)
    for j in range(N_pad):
        result_padded[:, j] = my_fft1d(rows_fft[:, j])

    # 裁剪回原尺寸
    return result_padded[:M_orig, :N_orig]


def my_ifft2(freq):
    """手动实现二维 IFFT (处理非2的幂尺寸)"""
    freq = np.asarray(freq, dtype=np.complex128)
    M_orig, N_orig = freq.shape

    # 补齐到2的幂
    M_pad = 1 << (M_orig - 1).bit_length() if M_orig & (M_orig - 1) else M_orig
    N_pad = 1 << (N_orig - 1).bit_length() if N_orig & (N_orig - 1) else N_orig

    freq_padded = np.zeros((M_pad, N_pad), dtype=np.complex128)
    freq_padded[:M_orig, :N_orig] = freq

    # 逐列 IFFT
    cols_ifft = np.zeros_like(freq_padded)
    for j in range(N_pad):
        cols_ifft[:, j] = my_ifft1d(freq_padded[:, j])

    # 逐行 IFFT
    result_padded = np.zeros_like(freq_padded)
    for i in range(M_pad):
        result_padded[i, :] = my_ifft1d(cols_ifft[i, :])

    return result_padded[:M_orig, :N_orig]


# ===================== 频谱中心化 =====================

def my_fftshift(spectrum):
    """
    频谱中心化: 将低频分量移到频谱中心
    原理: 交换频谱的四个象限 (1↔3, 2↔4)
    """
    arr = np.asarray(spectrum, dtype=np.complex128)
    return np.fft.fftshift(arr)  # NumPy fftshift 是纯数组操作，允许调用


def my_ifftshift(spectrum):
    """逆频谱中心化"""
    arr = np.asarray(spectrum, dtype=np.complex128)
    return np.fft.ifftshift(arr)


# ===================== 频域滤波器生成 =====================

def _meshgrid_freq_sq(shape):
    """生成频域坐标网格 (到中心距离的平方)"""
    M, N = shape
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    U, V = np.meshgrid(v, u)  # 注意: meshgrid(x, y) 索引顺序
    D_sq = U**2 + V**2
    return D_sq


def my_ideal_lowpass(shape, cutoff):
    """
    理想低通滤波器
    H(u,v) = 1 if D(u,v) <= cutoff else 0
    :param shape: 图像尺寸 (M, N)
    :param cutoff: 截止频率
    :return: 滤波器 (频域, 非中心化)
    """
    D_sq = _meshgrid_freq_sq(shape)
    H = np.zeros(shape, dtype=np.float64)
    H[D_sq <= cutoff**2] = 1.0
    return H


def my_ideal_highpass(shape, cutoff):
    """理想高通滤波器: H_hp = 1 - H_lp"""
    return 1.0 - my_ideal_lowpass(shape, cutoff)


def my_butterworth_lowpass(shape, cutoff, order=2):
    """
    巴特沃斯低通滤波器
    H(u,v) = 1 / (1 + (D(u,v)/cutoff)^(2n))
    比理想滤波器更平滑，无振铃效应
    :param shape: 图像尺寸
    :param cutoff: 截止频率
    :param order: 阶数 n
    :return: 滤波器
    """
    D_sq = _meshgrid_freq_sq(shape)
    # 避免除零
    eps = 1e-10
    H = 1.0 / (1.0 + (np.sqrt(D_sq + eps) / cutoff) ** (2 * order))
    return H


def my_butterworth_highpass(shape, cutoff, order=2):
    """巴特沃斯高通滤波器"""
    return 1.0 - my_butterworth_lowpass(shape, cutoff, order)


def my_gaussian_lowpass(shape, cutoff):
    """
    高斯低通滤波器
    H(u,v) = exp(-D(u,v)² / (2 * cutoff²))
    :param shape: 图像尺寸
    :param cutoff: 截止频率 (标准差)
    :return: 滤波器
    """
    D_sq = _meshgrid_freq_sq(shape)
    H = np.exp(-D_sq / (2.0 * cutoff**2))
    return H


def my_gaussian_highpass(shape, cutoff):
    """高斯高通滤波器"""
    return 1.0 - my_gaussian_lowpass(shape, cutoff)


# ===================== 频域滤波完整流程 =====================

def my_frequency_filter(image, filter_func, *filter_args, **filter_kwargs):
    """
    频域滤波完整流程
    1. FFT -> 2. 频谱中心化 -> 3. 应用滤波器 -> 4. 逆中心化 -> 5. IFFT
    :param image: 输入图像
    :param filter_func: 滤波器生成函数
    :return: 滤波后的实数图像
    """
    img = np.asarray(image, dtype=np.float64)
    M, N = img.shape

    # Step 1: 2D FFT
    F = my_fft2(img)

    # Step 2: 频谱中心化 (将低频移到中心)
    Fshift = my_fftshift(F)

    # Step 3: 生成并应用滤波器
    H = filter_func((M, N), *filter_args, **filter_kwargs)
    G_shift = Fshift * H

    # Step 4: 逆中心化
    G = my_ifftshift(G_shift)

    # Step 5: 2D IFFT
    g = my_ifft2(G)

    # 取实部 (理论上是实数)
    result = np.real(g)

    # 裁剪到原始图像尺寸 (补齐2的幂的情况)
    orig_h, orig_w = image.shape
    result = result[:orig_h, :orig_w]

    return result


def my_lowpass_filter(image, cutoff, filter_type='gaussian', order=2):
    """便捷低通滤波接口"""
    filter_map = {
        'ideal': my_ideal_lowpass,
        'butterworth': lambda s, c: my_butterworth_lowpass(s, c, order),
        'gaussian': my_gaussian_lowpass,
    }
    func = filter_map.get(filter_type, my_gaussian_lowpass)
    return my_frequency_filter(image, func, cutoff)


def my_highpass_filter(image, cutoff, filter_type='gaussian', order=2):
    """便捷高通滤波接口"""
    filter_map = {
        'ideal': my_ideal_highpass,
        'butterworth': lambda s, c: my_butterworth_highpass(s, c, order),
        'gaussian': my_gaussian_highpass,
    }
    func = filter_map.get(filter_type, my_gaussian_highpass)
    return my_frequency_filter(image, func, cutoff)
