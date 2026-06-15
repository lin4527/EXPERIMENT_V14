"""
推理预测模块
- 单张切片分割
- 3D体数据逐片分割
"""
import numpy as np
import torch
import cv2
from .unet_model import UNet


def load_model(model_path, in_channels=1, num_classes=10, base_filters=32, device='cpu'):
    """加载训练好的模型"""
    model = UNet(in_channels=in_channels, num_classes=num_classes,
                 base_filters=base_filters)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def segment_slice(model, image, image_size=256, device='cpu'):
    """
    对单张2D切片进行分割
    :param model: UNet 模型
    :param image: 2D numpy array
    :param image_size: 模型输入尺寸
    :param device: 'cpu' / 'cuda'
    :return: 分割标签图 (原图尺寸)
    """
    img = np.asarray(image, dtype=np.float64)

    # 保存原始尺寸
    orig_h, orig_w = img.shape

    # 归一化
    if img.max() > img.min():
        img_norm = (img - img.min()) / (img.max() - img.min())
    else:
        img_norm = np.zeros_like(img)

    # Resize
    img_resized = cv2.resize(img_norm.astype(np.float32), (image_size, image_size))

    # To tensor
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    # 推理
    with torch.no_grad():
        probs, labels = model.predict(img_tensor)

    labels_np = labels.cpu().squeeze().numpy()

    # Resize回原始尺寸
    if (orig_h != image_size) or (orig_w != image_size):
        labels_np = cv2.resize(labels_np.astype(np.float32), (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)

    return labels_np.astype(np.int32)


def segment_volume(model, ct_volume, slice_axis=0, image_size=256,
                   device='cpu', stride=None, overlap=0.5):
    """
    对3D体数据进行逐片分割
    :param model: UNet 模型
    :param ct_volume: 3D numpy array
    :param slice_axis: 切片轴
    :param image_size: 模型输入尺寸
    :param device: 'cpu' / 'cuda'
    :param stride: 切片步长 (None=每片)
    :param overlap: stride模式下的重叠比例
    :return: 3D 分割结果
    """
    vol = np.asarray(ct_volume, dtype=np.float64)
    num_slices = vol.shape[slice_axis]

    if stride is None:
        slice_indices = list(range(num_slices))
        results = []
        for idx in slice_indices:
            sl = np.take(vol, idx, axis=slice_axis).squeeze()
            label = segment_slice(model, sl, image_size, device)
            results.append(label)
        seg_volume = np.stack(results, axis=slice_axis)
    else:
        # 带重叠的滑动窗口预测
        step = max(1, int(stride * (1 - overlap)))
        seg_volume = np.zeros_like(vol, dtype=np.int32)
        count_volume = np.zeros_like(vol, dtype=np.int32)

        for idx in range(0, num_slices, step):
            if idx >= num_slices:
                break
            sl = np.take(vol, idx, axis=slice_axis).squeeze()
            label = segment_slice(model, sl, image_size, device)
            # 沉积到结果体
            np.put_along_axis(seg_volume, idx, label[..., None] if label.ndim < 2 else label, axis=slice_axis)

    return seg_volume
