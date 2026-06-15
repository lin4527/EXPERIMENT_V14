"""
模块2 独立测试脚本
测试所有手动实现的预处理算法，并与库函数对照
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import CT_PATH, OUTPUT_DIR

from module2_preprocessing.my_grayscale_transform import *
from module2_preprocessing.my_histogram import *
from module2_preprocessing.my_frequency_filter import *
from module2_preprocessing.my_interpolation import *
from module2_preprocessing.my_color_space import *
from module2_preprocessing.lib_reference import *

# 加载测试数据
from module1_io_visualization.nifti_reader import NiftiReader
from module1_io_visualization.mpr import MPRReconstructor
from module1_io_visualization.window_level import apply_preset


def test_module2():
    print("=" * 60)
    print("模块2 测试：全算子图像预处理")
    print("=" * 60)

    # 加载 CT 切片作为测试图像
    ct_reader = NiftiReader(CT_PATH)
    mpr = MPRReconstructor(ct_reader.data)
    test_img = apply_preset(mpr.axial(), "软组织窗")
    print(f"测试图像尺寸: {test_img.shape}, 范围: [{test_img.min()}, {test_img.max()}]")

    results = {}

    # 1. 灰度变换
    print("\n[1] 测试灰度变换...")
    r1 = {
        '原图': test_img,
        '线性(k=1.5,b=10)': my_linear_transform(test_img, 1.5, 10),
        '对数(c=10)': my_log_transform(test_img, 10),
        'Gamma(γ=0.5)': my_gamma_transform(test_img, 0.5),
        'Gamma(γ=2.0)': my_gamma_transform(test_img, 2.0),
        '分段线性': my_piecewise_linear_transform(test_img, [(80, 0), (180, 255)]),
    }
    results.update(r1)
    for name, img in r1.items():
        print(f"  {name}: range=[{img.min():.1f}, {img.max():.1f}]")

    # 2. 直方图
    print("\n[2] 测试直方图处理...")
    eq_my = my_histogram_equalization(test_img)
    eq_lib = lib_histogram_equalization(test_img)
    diff = np.abs(eq_my.astype(float) - eq_lib.astype(float)).mean()
    print(f"  直方图均衡化 - my vs lib 平均差异: {diff:.2f}")

    stats = my_histogram_stats(test_img)
    print(f"  统计特征: mean={stats['mean']:.1f}, std={stats['std']:.1f}, entropy={stats['entropy']:.2f}")

    results['直方图均衡化(my)'] = eq_my
    results['直方图均衡化(lib)'] = eq_lib

    # 3. 频域滤波
    print("\n[3] 测试频域滤波...")
    # 使用小图像加速 FFT 测试
    small_img = my_image_resample(test_img, 0.25, 'bilinear')
    print(f"  降采样图像尺寸: {small_img.shape} (加速FFT测试)")

    # 验证 FFT/IFFT 可逆性
    F = my_fft2(small_img)
    recon = np.real(my_ifft2(F))
    fft_error = np.abs(small_img.astype(float) - recon).mean()
    print(f"  FFT→IFFT 重构误差: {fft_error:.6f} (应接近0)")

    # 低通滤波
    lp = my_lowpass_filter(small_img, 15, 'gaussian')
    hp = my_highpass_filter(small_img, 30, 'gaussian')
    results['高斯低通滤波'] = lp
    results['高斯高通滤波'] = hp

    # 4. 插值
    print("\n[4] 测试插值...")
    small = my_image_resample(test_img, 0.25, 'nearest')
    nn_up = my_nearest_neighbor(small, test_img.shape[:2])
    bl_up = my_bilinear_interpolation(small, test_img.shape[:2])
    print(f"  最近邻放大: shape={nn_up.shape}")
    print(f"  双线性放大: shape={bl_up.shape}")

    results['最近邻插值'] = nn_up
    results['双线性插值'] = bl_up

    q16 = my_grayscale_quantize(test_img, 16)
    q8 = my_grayscale_quantize(test_img, 8)
    results['16级量化'] = q16
    results['8级量化'] = q8

    # 5. 色彩空间
    print("\n[5] 测试色彩空间转换...")
    # 用伪彩色生成测试 RGB
    pseudo = my_pseudocolor_map(test_img, 'jet')
    hsi = my_rgb_to_hsi(pseudo)
    rgb_back = my_hsi_to_rgb(hsi)
    rgb_diff = np.abs(pseudo.astype(float) - rgb_back.astype(float)).mean()
    print(f"  RGB→HSI→RGB 重构误差: {rgb_diff:.2f} (应较小)")

    hot_map = my_pseudocolor_map(test_img, 'hot')
    bone_map = my_pseudocolor_map(test_img, 'bone')
    results['伪彩色(Jet)'] = pseudo
    results['伪彩色(Hot)'] = hot_map
    results['伪彩色(Bone)'] = bone_map

    # 6. 可视化保存
    print("\n[6] 保存可视化结果...")
    n_results = len(results)
    n_cols = 4
    n_rows = (n_results + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else (axes if n_cols > 1 else [axes])

    for idx, (name, img) in enumerate(results.items()):
        if idx < len(axes):
            axes[idx].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[idx].set_title(name, fontsize=8)
            axes[idx].axis('off')

    # 隐藏多余的子图
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "test_module2.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  已保存到: {out_path}")

    print("\n[OK] 模块2 测试通过!")
    return True


if __name__ == "__main__":
    test_module2()
