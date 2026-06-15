"""
手动实现 K-means 聚类分割 — 实验9
支持两种特征方案:
1. 仅灰度特征 (1D K-means)
2. 灰度 + 空间坐标特征 (3D K-means: (x, y, intensity))
"""
import numpy as np


def my_kmeans(data, k=2, max_iters=100, tol=1e-4, init_centers=None):
    """
    K-means 聚类核心算法 (通用实现, 支持任意维特征)
    原理:
    1. 随机初始化 k 个聚类中心
    2. 分配: 将每个样本分配到最近的聚类中心
    3. 更新: 重新计算每个聚类的均值作为新的中心
    4. 重复步骤2-3直到收敛 (中心变化 < tol) 或达到最大迭代

    :param data: 特征矩阵 (N, D) — N 个样本, D 维特征
    :param k: 聚类数目
    :param max_iters: 最大迭代次数
    :param tol: 收敛阈值
    :param init_centers: 初始聚类中心 (k, D), None则随机初始化
    :return: (labels, centers)
    """
    data = np.asarray(data, dtype=np.float64)
    N, D = data.shape

    # 初始化聚类中心
    if init_centers is not None:
        centers = np.asarray(init_centers, dtype=np.float64)
    else:
        # 随机选择 k 个样本作为初始中心
        indices = np.random.choice(N, k, replace=False)
        centers = data[indices].copy()

    labels = np.zeros(N, dtype=np.int32)

    for iteration in range(max_iters):
        # --- Step 1: 分配 ---
        # 计算每个样本到各中心的欧氏距离平方
        distances = np.zeros((N, k), dtype=np.float64)
        for c in range(k):
            diff = data - centers[c]
            distances[:, c] = np.sum(diff**2, axis=1)

        new_labels = np.argmin(distances, axis=1)

        # --- Step 2: 更新中心 ---
        new_centers = np.zeros_like(centers)
        for c in range(k):
            cluster_data = data[new_labels == c]
            if len(cluster_data) > 0:
                new_centers[c] = cluster_data.mean(axis=0)
            else:
                # 空聚类: 重新初始化为随机数据点
                new_centers[c] = data[np.random.randint(N)]

        # --- 检查收敛 ---
        center_shift = np.sum((new_centers - centers)**2)
        labels = new_labels
        centers = new_centers

        if center_shift < tol:
            break

    return labels, centers


def my_kmeans_gray(image, k=3):
    """
    基于灰度特征的 K-means 分割 (1D 特征)
    :param image: 灰度图像
    :param k: 聚类数 (椎体分割建议 k=3: 背景/软组织/骨骼)
    :return: 分割标签图 (H, W), 聚类中心灰度值
    """
    img = np.asarray(image, dtype=np.float64)
    H, W = img.shape
    pixels = img.reshape(-1, 1)  # (N, 1) — 仅灰度

    labels, centers = my_kmeans(pixels, k=k)
    label_map = labels.reshape(H, W)

    return label_map, centers.flatten()


def my_kmeans_spatial(image, k=3, spatial_weight=0.5):
    """
    基于灰度 + 空间坐标的 K-means 分割 (3D 特征)
    特征向量: [x/spatial_scale, y/spatial_scale, intensity/255]
    空间权重越大，越倾向于空间紧凑的聚类

    :param image: 灰度图像
    :param k: 聚类数
    :param spatial_weight: 空间权重 (0=纯灰度, 1=最大空间影响)
    :return: 分割标签图 (H, W)
    """
    img = np.asarray(image, dtype=np.float64)
    H, W = img.shape

    # 构建特征向量
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    x_norm = x_coords.astype(np.float64) / max(W - 1, 1) * spatial_weight
    y_norm = y_coords.astype(np.float64) / max(H - 1, 1) * spatial_weight

    # 灰度归一化
    if img.max() > img.min():
        i_norm = (img - img.min()) / (img.max() - img.min()) * (1.0 - spatial_weight)
    else:
        i_norm = np.zeros_like(img)

    features = np.stack([
        x_norm.ravel(),
        y_norm.ravel(),
        i_norm.ravel(),
    ], axis=1)

    labels, centers = my_kmeans(features, k=k)
    label_map = labels.reshape(H, W)

    return label_map


def my_kmeans_with_mask(image, mask, k=3):
    """
    带掩膜的 K-means: 仅对掩膜内的像素聚类
    适用于先粗定位ROI再细分的场景
    """
    img = np.asarray(image, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    pixels = img[mask].reshape(-1, 1)

    if len(pixels) < k:
        # 像素太少，直接返回均匀分割
        result = np.zeros_like(img, dtype=np.int32)
        return result

    labels, centers = my_kmeans(pixels, k=k)

    # 重建完整标签图
    result = np.full(img.shape, -1, dtype=np.int32)
    result[mask] = labels

    return result
