"""
通用图像读写与格式转换 — 实验1核心
支持 jpg/png 读写，格式转换，像素级矩阵运算
"""
import numpy as np
import cv2
import os


def read_image(filepath, grayscale=True):
    """
    读取图像文件，支持 jpg/png/bmp/tif 等格式
    :param filepath: 图像文件路径
    :param grayscale: 是否转为灰度图
    :return: numpy array (H, W) 或 (H, W, 3)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    if grayscale:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {filepath}")
    return img


def write_image(filepath, image, normalize=True):
    """
    写入图像文件，自动处理数据类型和范围
    :param filepath: 输出路径
    :param image: numpy array
    :param normalize: 是否自动归一化到 0-255
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    img = image.copy()
    if normalize and img.dtype != np.uint8:
        if img.max() > img.min():
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.jpg':
        cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(filepath, img)
    return filepath


def get_image_info(filepath):
    """获取图像基本信息"""
    if not os.path.exists(filepath):
        return {}
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return {}
    return {
        "shape": img.shape,
        "dtype": str(img.dtype),
        "min": int(img.min()),
        "max": int(img.max()),
        "mean": float(img.mean()),
        "std": float(img.std()),
        "file_size_kb": os.path.getsize(filepath) / 1024,
    }


def convert_format(input_path, output_path):
    """图像格式转换"""
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取: {input_path}")
    write_image(output_path, img)
    return output_path


def normalize_image(image):
    """归一化到 0-255"""
    if image.max() == image.min():
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)


# 像素级矩阵运算示例
def pixel_operations(image):
    """像素级矩阵运算：加、减、乘、除、取反"""
    img = image.astype(np.float64)
    return {
        "原图": img,
        "亮度+50": np.clip(img + 50, 0, 255),
        "亮度-50": np.clip(img - 50, 0, 255),
        "对比度x1.5": np.clip(img * 1.5, 0, 255),
        "取反": 255 - img,
    }
