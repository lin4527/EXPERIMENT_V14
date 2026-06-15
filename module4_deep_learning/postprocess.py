"""
分割后处理 — 连通域分析 + 椎体节段编号
基于解剖学先验知识对分割结果进行清理和标注
"""
import numpy as np
from module5_registration_eval.my_morphology import (
    my_opening, my_closing, my_connected_components,
    my_largest_connected_component
)


def postprocess_vertebrae(segmentation, min_area=100, apply_morphology=True):
    """
    椎体分割后处理
    1. 形态学清理 (开运算去噪 + 闭运算填孔)
    2. 连通域分析
    3. 小区域过滤
    4. 按解剖位置对椎体进行节段编号 (从上到下 / 从下到上)

    :param segmentation: 分割结果 (多类别标签图)
    :param min_area: 最小椎体面积 (像素)
    :param apply_morphology: 是否应用形态学处理
    :return: 清理后的标签图 + 椎体编号映射
    """
    seg = np.asarray(segmentation, dtype=np.int32)
    labels_unique = np.unique(seg)
    labels_unique = labels_unique[labels_unique > 0]  # 排除背景

    cleaned = np.zeros_like(seg)

    for label in labels_unique:
        mask = (seg == label)

        if apply_morphology:
            # 开运算去小噪点
            mask = my_opening(mask, kernel_size=3)
            # 闭运算填小孔
            mask = my_closing(mask, kernel_size=5)

        # 保留最大连通域
        mask = my_largest_connected_component(mask, connectivity=8)

        # 面积过滤
        if mask.sum() < min_area:
            continue

        cleaned[mask] = label

    return cleaned


def assign_vertebra_levels(segmentation, spacing=None, direction='descending'):
    """
    根据空间位置对椎体进行节段编号
    原理: 沿脊柱方向 (通常是Z轴/sagittal轴) 排序椎体区域
    :param segmentation: 分割标签图或3D分割
    :param spacing: 像素间距
    :param direction: 'descending' (从上到下, 颈椎→骶椎)
    :return: 重新编号的标签图 + {新标签: 原标签} 映射
    """
    seg = np.asarray(segmentation, dtype=np.int32)

    # 对每个唯一标签，计算其质心在主轴上的位置
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels > 0]

    centroids = {}
    for label in unique_labels:
        mask = (seg == label)
        coords = np.argwhere(mask)
        if len(coords) > 0:
            centroid = coords.mean(axis=0)
            centroids[label] = centroid

    if not centroids:
        return seg, {}

    # 找出主轴 (方差最大的维度)
    all_cents = np.array(list(centroids.values()))
    axis_var = np.var(all_cents, axis=0)
    main_axis = np.argmax(axis_var)  # 0=Z, 1=Y, 2=X

    # 沿主轴排序
    sorted_labels = sorted(centroids.keys(),
                           key=lambda l: centroids[l][main_axis],
                           reverse=(direction == 'descending'))

    # 重新编号 (1, 2, 3, ...)
    mapping = {}
    relabeled = np.zeros_like(seg)
    for new_label, old_label in enumerate(sorted_labels, start=1):
        mapping[new_label] = int(old_label)
        relabeled[seg == old_label] = new_label

    return relabeled, mapping


def estimate_vertebra_count(segmentation, min_area=50):
    """估算椎体数量"""
    seg = np.asarray(segmentation, dtype=np.int32)
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels > 0]

    count = 0
    for label in unique_labels:
        area = np.sum(seg == label)
        if area >= min_area:
            count += 1

    return count
