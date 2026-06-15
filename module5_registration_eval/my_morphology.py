"""
手动实现形态学操作 — 实验9后处理
- 腐蚀 (Erosion)
- 膨胀 (Dilation)
- 开运算 (Opening)
- 闭运算 (Closing)
"""
import numpy as np


def _structuring_element(shape='square', size=3):
    """生成结构元素"""
    if shape == 'square':
        return np.ones((size, size), dtype=bool)
    elif shape == 'cross':
        se = np.zeros((size, size), dtype=bool)
        center = size // 2
        se[center, :] = True
        se[:, center] = True
        return se
    elif shape == 'disk':
        se = np.zeros((size, size), dtype=bool)
        center = size // 2
        radius = size // 2
        for i in range(size):
            for j in range(size):
                if (i - center)**2 + (j - center)**2 <= radius**2:
                    se[i, j] = True
        return se
    else:
        return np.ones((size, size), dtype=bool)


def my_erosion(binary_image, kernel_size=3, kernel_shape='square', iterations=1):
    """
    腐蚀操作
    原理: 结构元素在图像上滑动，仅当SE完全被前景覆盖时中心像素才为前景
    效果: 收缩前景，消除细小噪点，断开狭窄连接
    :param binary_image: 二值图像 (0/255 或 bool)
    :param kernel_size: 核尺寸
    :param kernel_shape: 核形状 ('square', 'cross', 'disk')
    :param iterations: 腐蚀次数
    :return: 腐蚀后的二值图像
    """
    img = np.asarray(binary_image, dtype=bool)
    se = _structuring_element(kernel_shape, kernel_size)
    pad = kernel_size // 2
    H, W = img.shape

    result = img.copy()
    for _ in range(iterations):
        prev = result.copy()
        padded = np.pad(prev, pad, constant_values=False)
        result = np.zeros_like(prev)

        for i in range(H):
            for j in range(W):
                patch = padded[i:i+kernel_size, j:j+kernel_size]
                # 腐蚀: 所有SE覆盖的像素都必须为True
                result[i, j] = np.all(patch[se])

    return result


def my_dilation(binary_image, kernel_size=3, kernel_shape='square', iterations=1):
    """
    膨胀操作
    原理: 结构元素在图像上滑动，只要SE中任一元素碰到前景，中心像素就为前景
    效果: 扩张前景，填充内部空洞，连接相邻区域
    :return: 膨胀后的二值图像
    """
    img = np.asarray(binary_image, dtype=bool)
    se = _structuring_element(kernel_shape, kernel_size)
    pad = kernel_size // 2
    H, W = img.shape

    result = img.copy()
    for _ in range(iterations):
        prev = result.copy()
        padded = np.pad(prev, pad, constant_values=False)
        result = np.zeros_like(prev)

        for i in range(H):
            for j in range(W):
                patch = padded[i:i+kernel_size, j:j+kernel_size]
                # 膨胀: SE中任一像素为True则中心为True
                result[i, j] = np.any(patch & se)

    return result


def my_opening(binary_image, kernel_size=3, kernel_shape='square'):
    """
    开运算 = 先腐蚀后膨胀
    效果: 消除小噪点，平滑轮廓，断开狭窄连接
    """
    eroded = my_erosion(binary_image, kernel_size, kernel_shape)
    return my_dilation(eroded, kernel_size, kernel_shape)


def my_closing(binary_image, kernel_size=3, kernel_shape='square'):
    """
    闭运算 = 先膨胀后腐蚀
    效果: 填充小空洞，连接相邻区域，平滑轮廓
    """
    dilated = my_dilation(binary_image, kernel_size, kernel_shape)
    return my_erosion(dilated, kernel_size, kernel_shape)


def my_morphological_gradient(binary_image, kernel_size=3):
    """形态学梯度 = 膨胀 - 腐蚀 (提取边缘)"""
    dilated = my_dilation(binary_image, kernel_size)
    eroded = my_erosion(binary_image, kernel_size)
    return dilated.astype(np.int32) - eroded.astype(np.int32)


def my_connected_components(binary_image, connectivity=8):
    """
    连通域分析 (简化版 two-pass 算法)
    用于分割后处理中的区域标记
    :return: (标签图, 连通域数量)
    """
    img = np.asarray(binary_image, dtype=bool)
    H, W = img.shape
    labels = np.zeros((H, W), dtype=np.int32)

    # First pass: 初步标记
    current_label = 1
    equivalences = {}

    for i in range(H):
        for j in range(W):
            if not img[i, j]:
                continue

            # 检查已扫描的邻域
            neighbors = []
            if connectivity == 8:
                # 检查上、左上、右上、左
                if i > 0 and labels[i-1, j] > 0:
                    neighbors.append(labels[i-1, j])
                if i > 0 and j > 0 and labels[i-1, j-1] > 0:
                    neighbors.append(labels[i-1, j-1])
                if i > 0 and j < W-1 and labels[i-1, j+1] > 0:
                    neighbors.append(labels[i-1, j+1])
                if j > 0 and labels[i, j-1] > 0:
                    neighbors.append(labels[i, j-1])
            else:  # 4连通
                if i > 0 and labels[i-1, j] > 0:
                    neighbors.append(labels[i-1, j])
                if j > 0 and labels[i, j-1] > 0:
                    neighbors.append(labels[i, j-1])

            if not neighbors:
                labels[i, j] = current_label
                current_label += 1
            else:
                min_label = min(neighbors)
                labels[i, j] = min_label
                for nl in neighbors:
                    if nl != min_label:
                        equivalences[nl] = min_label

    # 解决等效标签
    for label in range(1, current_label):
        if label in equivalences:
            # 追踪到最小等价标签
            root = label
            while root in equivalences and equivalences[root] != root:
                root = equivalences[root]
            equivalences[label] = root

    # 重新编号为连续标签
    valid_labels = set(range(1, current_label))
    resolved = {}
    new_label = 1
    for label in sorted(valid_labels):
        root = label
        while root in equivalences:
            root = equivalences[root]
            if root == equivalences.get(root):
                break
        if root not in resolved:
            resolved[root] = new_label
            new_label += 1
        resolved[label] = resolved[root]

    # Second pass: 应用最终标签
    for i in range(H):
        for j in range(W):
            if labels[i, j] > 0 and labels[i, j] in resolved:
                labels[i, j] = resolved[labels[i, j]]

    n_components = new_label - 1
    return labels, n_components


def my_largest_connected_component(binary_image, connectivity=8):
    """
    保留最大连通域 (常用于去除分割噪声)
    """
    labels, n_comp = my_connected_components(binary_image, connectivity)
    if n_comp == 0:
        return np.zeros_like(binary_image, dtype=bool)

    # 统计各连通域面积
    comp_areas = {}
    for label in range(1, n_comp + 1):
        comp_areas[label] = np.sum(labels == label)

    # 找最大连通域
    largest_label = max(comp_areas, key=comp_areas.get)
    return labels == largest_label
