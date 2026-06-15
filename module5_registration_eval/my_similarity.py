"""
手动实现相似性度量与分割评估指标 — 实验9、10
- Dice, IoU (Jaccard) 分割评估指标
- MSE 均方误差
- MI 互信息 / NMI 归一化互信息
"""
import numpy as np
import math


# ===================== 分割评估指标 =====================

def my_dice(pred, target, smooth=1e-5):
    """
    Dice 系数 (F1-score for segmentation)
    原理: Dice = 2*|A∩B| / (|A|+|B|)
    值域: [0, 1], 1=完全重合
    :param pred: 预测分割 (二值或整数标签)
    :param target: 金标准分割
    :return: Dice 系数
    """
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)

    intersection = np.sum(pred & target)
    dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

    return float(dice)


def my_iou(pred, target, smooth=1e-5):
    """
    IoU (Jaccard Index) 交并比
    原理: IoU = |A∩B| / |A∪B|
    值域: [0, 1], 1=完全重合
    :return: IoU 值
    """
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)

    intersection = np.sum(pred & target)
    union = np.sum(pred | target)

    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def my_dice_multiclass(pred, target, num_classes=None, ignore_bg=True):
    """
    多类别 Dice (逐类计算后取平均)
    :param pred: 预测标签图 (H, W)
    :param target: 金标准标签图 (H, W)
    :param num_classes: 类别数 (None则自动)
    :param ignore_bg: 是否忽略背景 (label=0)
    :return: 各类Dice列表 + 均值
    """
    pred = np.asarray(pred, dtype=np.int32)
    target = np.asarray(target, dtype=np.int32)

    if num_classes is None:
        num_classes = int(max(pred.max(), target.max())) + 1

    dices = []
    start = 1 if ignore_bg else 0
    for c in range(start, num_classes):
        p = (pred == c)
        t = (target == c)
        if p.sum() + t.sum() > 0:
            d = my_dice(p, t)
            dices.append(d)
        else:
            dices.append(1.0)  # 双方都为空的类视为完美预测

    return dices, float(np.mean(dices)) if dices else 0.0


# ===================== 通用相似性度量 =====================

def my_mse(image1, image2):
    """
    均方误差 (Mean Squared Error)
    原理: MSE = (1/N) * Σ(I1 - I2)²
    :param image1, image2: 同尺寸图像
    :return: MSE 值
    """
    a = np.asarray(image1, dtype=np.float64).flatten()
    b = np.asarray(image2, dtype=np.float64).flatten()
    return float(np.mean((a - b)**2))


def my_mi(image1, image2, bins=256):
    """
    互信息 (Mutual Information)
    原理: MI(A,B) = H(A) + H(B) - H(A,B)
          = Σ p(a,b) × log₂(p(a,b) / (p(a) × p(b)))
    值域: [0, ∞), 越大越相似

    :param image1, image2: 同尺寸图像
    :param bins: 直方图 bins 数
    :return: MI 值
    """
    a = np.asarray(image1, dtype=np.float64).flatten()
    b = np.asarray(image2, dtype=np.float64).flatten()

    # 归一化到 [0, bins-1]
    if a.max() > a.min():
        a = ((a - a.min()) / (a.max() - a.min()) * (bins - 1)).astype(np.int32)
    else:
        a = np.zeros_like(a, dtype=np.int32)
    if b.max() > b.min():
        b = ((b - b.min()) / (b.max() - b.min()) * (bins - 1)).astype(np.int32)
    else:
        b = np.zeros_like(b, dtype=np.int32)

    # 联合直方图 H(a,b)
    joint_hist = np.zeros((bins, bins), dtype=np.float64)
    for i in range(len(a)):
        joint_hist[a[i], b[i]] += 1

    # 概率分布
    joint_prob = joint_hist / len(a)

    # 边缘概率
    p_a = np.sum(joint_prob, axis=1)
    p_b = np.sum(joint_prob, axis=0)

    # 互信息计算
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint_prob[i, j] > 0:
                p_a_i = p_a[i] if p_a[i] > 0 else 1e-10
                p_b_j = p_b[j] if p_b[j] > 0 else 1e-10
                mi += joint_prob[i, j] * math.log2(joint_prob[i, j] / (p_a_i * p_b_j))

    return float(mi)


def my_nmi(image1, image2, bins=256):
    """
    归一化互信息 (Normalized Mutual Information)
    原理: NMI = 2 × MI(A,B) / (H(A) + H(B))
    值域: [0, 1], 1=完美匹配
    """
    a = np.asarray(image1, dtype=np.float64).flatten()
    b = np.asarray(image2, dtype=np.float64).flatten()

    mi = my_mi(image1, image2, bins)

    # 计算各自的熵
    def _entropy(x, bins):
        if x.max() > x.min():
            x_norm = ((x - x.min()) / (x.max() - x.min()) * (bins - 1)).astype(np.int32)
        else:
            x_norm = np.zeros_like(x, dtype=np.int32)
        hist = np.bincount(x_norm, minlength=bins).astype(np.float64)
        prob = hist / len(x)
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob)))

    h1 = _entropy(a, bins)
    h2 = _entropy(b, bins)

    if h1 + h2 > 0:
        nmi = 2.0 * mi / (h1 + h2)
    else:
        nmi = 0.0

    return float(nmi)


def my_psnr(image1, image2, max_val=255.0):
    """
    峰值信噪比 (PSNR)
    PSNR = 10 × log₁₀(MAX² / MSE)
    """
    mse = my_mse(image1, image2)
    if mse < 1e-10:
        return float('inf')
    return float(10.0 * math.log10(max_val**2 / mse))
