"""
手动实现 2D 图像配准算法 — 实验10
- 刚性配准 (旋转 + 平移)
- 仿射配准 (缩放 + 旋转 + 剪切 + 平移)
使用 MI/NMI 作为相似性度量, 梯度下降优化
"""
import numpy as np
import math
from scipy.ndimage import map_coordinates
from .my_similarity import my_mi, my_nmi, my_mse


def _affine_transform_matrix(params):
    """
    从参数向量构建仿射变换矩阵 (2×3)
    params: [tx, ty, theta, sx, sy, shx, shy]
    刚性配准时 sx=sy=1, shx=shy=0
    """
    tx, ty = params[0], params[1]
    theta = params[2] if len(params) > 2 else 0.0
    sx = params[3] if len(params) > 3 else 1.0
    sy = params[4] if len(params) > 4 else 1.0
    shx = params[5] if len(params) > 5 else 0.0
    shy = params[6] if len(params) > 6 else 0.0

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # 仿射矩阵: A = [[a, b, tx], [c, d, ty]]
    # 其中 [a b; c d] = Scale * Shear * Rotation
    a = sx * (cos_t + shx * sin_t)
    b = sx * (shx * cos_t - sin_t)
    c = sy * (sin_t + shy * cos_t)
    d = sy * (shy * sin_t + cos_t)

    return np.array([[a, b, tx], [c, d, ty]])


def _apply_transform(image, matrix, output_shape=None):
    """
    应用仿射变换到图像 (逆向映射 + 双线性插值)
    """
    img = np.asarray(image, dtype=np.float64)
    H, W = img.shape
    if output_shape is None:
        out_h, out_w = H, W
    else:
        out_h, out_w = output_shape

    # 生成目标像素坐标网格
    y_coords, x_coords = np.mgrid[0:out_h, 0:out_w]

    # 逆向变换: 目标坐标 → 源坐标
    # 目标坐标先转为齐次坐标 [x, y, 1]^T
    # 源坐标 = inv(M) × 目标齐次坐标
    M = np.vstack([matrix, [0, 0, 1]])  # 3×3
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.zeros((out_h, out_w))

    homog_coords = np.stack([x_coords.ravel(), y_coords.ravel(),
                             np.ones(out_h * out_w)], axis=0)
    src_coords = M_inv @ homog_coords
    src_x = src_coords[0, :].reshape(out_h, out_w)
    src_y = src_coords[1, :].reshape(out_h, out_w)

    # map_coordinates 使用 (row, col) = (y, x) 顺序
    result = map_coordinates(img, [src_y, src_x], order=1, mode='constant', cval=0.0)

    return result


def _cost_function(params, fixed, moving, metric='mi', shape=None):
    """配准代价函数 (负相似度)"""
    M = _affine_transform_matrix(params)
    transformed = _apply_transform(moving, M, output_shape=fixed.shape if shape is None else shape)

    if metric == 'mi':
        sim = my_mi(fixed, transformed)
    elif metric == 'nmi':
        sim = my_nmi(fixed, transformed)
    elif metric == 'mse':
        sim = -my_mse(fixed, transformed)  # MSE越低越好 → 负值越高越好
    else:
        sim = my_nmi(fixed, transformed)

    return -sim  # 最小化负相似度 = 最大化相似度


def _numerical_gradient(params, fixed, moving, metric, eps=1e-5):
    """数值梯度 (中心差分)"""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps

        cost_plus = _cost_function(params_plus, fixed, moving, metric)
        cost_minus = _cost_function(params_minus, fixed, moving, metric)

        grad[i] = (cost_plus - cost_minus) / (2.0 * eps)

    return grad


def my_rigid_registration(fixed, moving, metric='mi', max_iters=100, lr=0.05):
    """
    2D 刚性配准 (旋转 + 平移)
    参数: [tx, ty, theta] — x平移, y平移, 旋转角度
    优化方法: 梯度下降

    :param fixed: 固定图像 (参考)
    :param moving: 浮动图像 (待配准)
    :param metric: 'mi', 'nmi', 'mse'
    :param max_iters: 最大迭代次数
    :param lr: 学习率
    :return: (配准后图像, 最终参数, 变换矩阵)
    """
    # 初始参数: [tx, ty, theta] — 零平移, 零旋转
    params = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    best_params = params.copy()
    best_cost = float('inf')

    for iteration in range(max_iters):
        cost = _cost_function(params, fixed, moving, metric)

        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()

        # 数值梯度
        grad = _numerical_gradient(params, fixed, moving, metric)

        # 梯度下降
        params = params - lr * grad

        # 角度归一化到 [-π, π]
        params[2] = (params[2] + np.pi) % (2 * np.pi) - np.pi

        if np.max(np.abs(grad)) < 1e-6:
            break

    # 生成最终变换矩阵
    transform = _affine_transform_matrix(best_params)
    registered = _apply_transform(moving, transform, output_shape=fixed.shape)

    return registered, best_params, transform


def my_affine_registration(fixed, moving, metric='mi', max_iters=150, lr=0.03):
    """
    2D 仿射配准 (缩放 + 旋转 + 剪切 + 平移)
    参数: [tx, ty, theta, sx, sy, shx, shy]
    优化: 梯度下降 + 动量

    :return: (配准后图像, 最终参数, 变换矩阵)
    """
    # 初始参数: 恒等变换
    params = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)

    best_params = params.copy()
    best_cost = float('inf')
    velocity = np.zeros_like(params)
    momentum = 0.9

    for iteration in range(max_iters):
        cost = _cost_function(params, fixed, moving, metric)

        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()

        grad = _numerical_gradient(params, fixed, moving, metric)

        # 动量更新
        velocity = momentum * velocity - lr * grad
        params = params + velocity

        # 角度归一化
        params[2] = (params[2] + np.pi) % (2 * np.pi) - np.pi
        # 缩放不小于0.1
        params[3] = max(0.1, params[3])
        params[4] = max(0.1, params[4])

        if np.max(np.abs(grad)) < 1e-6:
            break

    transform = _affine_transform_matrix(best_params)
    registered = _apply_transform(moving, transform, output_shape=fixed.shape)

    return registered, best_params, transform
