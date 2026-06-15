"""
结果展示模式 — 差值图 / 伪彩色融合 / 分割轮廓叠加
"""
import numpy as np
import cv2


def difference_map(original, result, alpha=0.5):
    """
    差值图: 计算原图与结果的差异
    :return: 差异热力图
    """
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(result, dtype=np.float64)

    if a.shape != b.shape:
        import cv2
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    diff = np.abs(a - b)

    # 归一化并应用热力图
    if diff.max() > 0:
        diff_norm = (diff / diff.max() * 255).astype(np.uint8)
    else:
        diff_norm = np.zeros_like(diff, dtype=np.uint8)

    heat = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    return heat


def pseudocolor_fusion(original, overlay, alpha=0.4):
    """
    伪彩色融合: 将分割结果以伪彩色叠加在原图上
    """
    orig = np.asarray(original, dtype=np.float64)
    over = np.asarray(overlay, dtype=np.float64)

    if orig.ndim == 2:
        orig_rgb = cv2.cvtColor(
            np.clip((orig - orig.min()) / (orig.max() - orig.min() + 1e-10) * 255, 0, 255).astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )
    else:
        orig_rgb = orig.copy()

    if over.shape[:2] != orig_rgb.shape[:2]:
        over = cv2.resize(over, (orig_rgb.shape[1], orig_rgb.shape[0]))

    if over.ndim == 2:
        over_color = cv2.applyColorMap(
            np.clip((over - over.min()) / (over.max() - over.min() + 1e-10) * 255, 0, 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
    else:
        over_color = over

    fused = cv2.addWeighted(orig_rgb.astype(np.uint8), 1.0 - alpha,
                            over_color.astype(np.uint8), alpha, 0)
    return fused


def contour_overlay(original, segmentation, contour_color=(0, 255, 0), thickness=1):
    """
    轮廓叠加: 提取分割边界并绘制在原图上
    """
    orig = np.asarray(original, dtype=np.float64)
    seg = np.asarray(segmentation, dtype=np.float64)

    # 归一化原图
    if orig.max() > orig.min():
        orig_norm = ((orig - orig.min()) / (orig.max() - orig.min()) * 255).astype(np.uint8)
    else:
        orig_norm = np.zeros_like(orig, dtype=np.uint8)

    if orig_norm.ndim == 2:
        overlay_img = cv2.cvtColor(orig_norm, cv2.COLOR_GRAY2BGR)
    else:
        overlay_img = orig_norm.copy()

    # 确保分割为二值
    if seg.max() > 1:
        seg_bin = (seg > 0).astype(np.uint8) * 255
    else:
        seg_bin = (seg > 0.5).astype(np.uint8) * 255

    if seg_bin.shape != overlay_img.shape[:2]:
        seg_bin = cv2.resize(seg_bin, (overlay_img.shape[1], overlay_img.shape[0]))

    # 提取轮廓
    contours, _ = cv2.findContours(seg_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_img, contours, -1, contour_color, thickness)

    return overlay_img


def multi_class_overlay(original, labels, alpha=0.3):
    """
    多类别分割叠加: 不同的类别用不同颜色
    """
    orig = np.asarray(original, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    if orig.max() > orig.min():
        orig_norm = ((orig - orig.min()) / (orig.max() - orig.min()) * 255).astype(np.uint8)
    else:
        orig_norm = np.zeros_like(orig, dtype=np.uint8)

    if orig_norm.ndim == 2:
        overlay_img = cv2.cvtColor(orig_norm, cv2.COLOR_GRAY2BGR)
    else:
        overlay_img = orig_norm.copy()

    if labels.shape != overlay_img.shape[:2]:
        labels = cv2.resize(labels.astype(np.float32),
                            (overlay_img.shape[1], overlay_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(np.int32)

    # 生成伪彩色标签图
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    colors = {}
    for i, label in enumerate(unique_labels):
        # 基于HSV生成不同颜色
        hue = int(180 * i / max(len(unique_labels), 1))
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors[label] = tuple(int(c) for c in color_bgr)

    color_mask = np.zeros_like(overlay_img)
    for label, color in colors.items():
        color_mask[labels == label] = color

    fused = cv2.addWeighted(overlay_img, 1.0 - alpha, color_mask, alpha, 0)
    return fused
