"""
库函数对照版本 — 用于实验对比分析
调用标准库 (numpy, scipy, skimage) 实现等价功能
与手动实现版本一一对应，便于验证精度和性能差异
"""
import numpy as np
from scipy import fft
from scipy.ndimage import zoom
import cv2


# ===================== 直方图对照 =====================

def lib_histogram_equalization(image):
    """OpenCV 直方图均衡化 — 对照 my_histogram_equalization"""
    img = np.asarray(image, dtype=np.uint8)
    return cv2.equalizeHist(img)


def lib_histogram_stats(image):
    """NumPy 直方图统计 — 对照 my_histogram_stats"""
    img = np.asarray(image, dtype=np.float64).flatten()
    return {
        "mean": float(np.mean(img)),
        "std": float(np.std(img)),
        "min": float(img.min()),
        "max": float(img.max()),
    }


# ===================== FFT 对照 =====================

def lib_fft2(image):
    """NumPy FFT — 对照 my_fft2"""
    return np.fft.fft2(np.asarray(image))


def lib_ifft2(spectrum):
    """NumPy IFFT — 对照 my_ifft2"""
    return np.fft.ifft2(spectrum)


def lib_frequency_filter(image, cutoff, filter_type='gaussian'):
    """scipy 频域滤波对照"""
    from scipy import ndimage
    img = np.asarray(image, dtype=np.float64)
    if filter_type == 'gaussian':
        sigma = 1.0 / (2 * np.pi * cutoff) if cutoff > 0 else 3.0
        return ndimage.gaussian_filter(img, sigma=sigma)
    return img


# ===================== 边缘检测对照 =====================

def lib_sobel_edge(image):
    """OpenCV Sobel 边缘检测 — 对照 my_sobel"""
    img = np.asarray(image, dtype=np.uint8)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude


def lib_canny_edge(image, low=50, high=150):
    """OpenCV Canny 边缘检测 — 对照 my_canny"""
    return cv2.Canny(np.asarray(image, dtype=np.uint8), low, high)


# ===================== 插值对照 =====================

def lib_bilinear_resample(image, scale_factor):
    """scipy 双线性重采样 — 对照 my_bilinear_interpolation"""
    return zoom(np.asarray(image, dtype=np.float64), scale_factor, order=1)


def lib_nearest_resample(image, scale_factor):
    """scipy 最近邻重采样 — 对照 my_nearest_neighbor"""
    return zoom(np.asarray(image, dtype=np.float64), scale_factor, order=0)


# ===================== 阈值分割对照 =====================

def lib_otsu(image):
    """OpenCV Otsu 阈值 — 对照 my_otsu"""
    img = np.asarray(image, dtype=np.uint8)
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, ret


# ===================== 形态学对照 =====================

def lib_erosion(image, kernel_size=3):
    """OpenCV 腐蚀 — 对照 my_erosion"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(np.asarray(image, dtype=np.uint8), kernel)


def lib_dilation(image, kernel_size=3):
    """OpenCV 膨胀 — 对照 my_dilation"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(np.asarray(image, dtype=np.uint8), kernel)
