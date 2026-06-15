"""
数据集加载与数据增强流水线
从CT 3D体数据提取2D切片 + 对应标注
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class VertebraDataset(Dataset):
    """
    椎体分割数据集
    从3D CT体数据和标注中逐切片提取训练样本
    """

    def __init__(self, ct_volume, seg_volume, slice_axis=0,
                 image_size=256, transform=None, slices_indices=None):
        """
        :param ct_volume: CT 3D体数据 (numpy array)
        :param seg_volume: 标注3D体数据 (numpy array, 整数标签)
        :param slice_axis: 切片轴 (0=sagittal, 1=coronal, 2=axial)
        :param image_size: 统一resize到的尺寸
        :param transform: 数据增强函数
        :param slices_indices: 指定切片索引列表 (None则使用所有切片)
        """
        self.ct = np.asarray(ct_volume, dtype=np.float64)
        self.seg = np.asarray(seg_volume, dtype=np.int32)

        # 🔧 自动重映射标签: 原始标签(如16-24) → 连续标签(0,1,2,...)
        raw_labels = np.unique(self.seg)
        self.label_map = {int(raw): new for new, raw in enumerate(raw_labels)}
        self.num_classes = len(raw_labels)

        self.slice_axis = slice_axis
        self.image_size = image_size
        self.transform = transform

        total_slices = self.ct.shape[slice_axis]
        if slices_indices is None:
            self.slices = list(range(total_slices))
        else:
            self.slices = slices_indices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice_idx = self.slices[idx]

        # 提取切片
        ct_slice = np.take(self.ct, slice_idx, axis=self.slice_axis).squeeze()
        seg_slice = np.take(self.seg, slice_idx, axis=self.slice_axis).squeeze()

        # 确保2D
        if ct_slice.ndim > 2:
            ct_slice = ct_slice[..., 0] if ct_slice.shape[2] == 1 else ct_slice[:, :, 0]
        if seg_slice.ndim > 2:
            seg_slice = seg_slice[..., 0] if seg_slice.shape[2] == 1 else seg_slice[:, :, 0]

        # 归一化CT值
        if ct_slice.max() > ct_slice.min():
            ct_slice = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())
        else:
            ct_slice = np.zeros_like(ct_slice)

        # Resize 到统一尺寸
        import cv2
        ct_slice = cv2.resize(ct_slice.astype(np.float32),
                              (self.image_size, self.image_size),
                              interpolation=cv2.INTER_LINEAR)
        seg_slice = cv2.resize(seg_slice.astype(np.float32),
                               (self.image_size, self.image_size),
                               interpolation=cv2.INTER_NEAREST)

        # 数据增强
        if self.transform:
            ct_slice, seg_slice = self.transform(ct_slice, seg_slice)

        # 🔧 重映射标签: 原始(16-24) → 连续(0-9)
        seg_remapped = np.zeros_like(seg_slice, dtype=np.int32)
        for raw_lbl, new_lbl in self.label_map.items():
            seg_remapped[seg_slice == raw_lbl] = new_lbl

        # 转为 Tensor
        ct_tensor = torch.from_numpy(ct_slice).float().unsqueeze(0)  # (1, H, W)
        seg_tensor = torch.from_numpy(seg_remapped).long()           # (H, W)

        return ct_tensor, seg_tensor


# ===================== 数据增强 =====================

class VertebraTransform:
    """数据增强流水线"""

    def __init__(self, prob_aug=0.5, intensity_range=0.1, rotation_range=15):
        self.prob_aug = prob_aug
        self.intensity_range = intensity_range
        self.rotation_range = rotation_range

    def __call__(self, image, mask):
        # 灰度扰动
        if random.random() < self.prob_aug:
            noise = np.random.uniform(-self.intensity_range, self.intensity_range)
            image = np.clip(image + noise, 0, 1)

        # 水平翻转
        if random.random() < self.prob_aug:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # 垂直翻转
        if random.random() < self.prob_aug:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        return image, mask
