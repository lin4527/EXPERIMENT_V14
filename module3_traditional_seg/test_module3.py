"""
模块3 独立测试脚本
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import CT_PATH, SEG_PATH, OUTPUT_DIR

from module1_io_visualization.nifti_reader import NiftiReader
from module1_io_visualization.mpr import MPRReconstructor
from module1_io_visualization.window_level import apply_preset


def test_module3():
    print("=" * 60)
    print("模块3 测试：传统方法分割基线")
    print("=" * 60)

    # 加载测试数据
    ct = NiftiReader(CT_PATH)
    seg = NiftiReader(SEG_PATH)
    mpr = MPRReconstructor(ct.data)
    test_img = apply_preset(mpr.axial(), "软组织窗")

    # 同时获取对应标注切片
    seg_data = seg.data
    seg_slice = np.take(seg_data, mpr.shape[0]//2, axis=0).squeeze()
    seg_slice = (seg_slice > 0).astype(np.uint8) * 255  # 二值化

    print(f"测试图像: {test_img.shape}")

    results = {}

    # 1. 边缘检测
    print("\n[1] 边缘检测...")
    from module3_traditional_seg.my_edge_detection import (
        my_roberts, my_sobel, my_prewitt, my_canny,
        my_laplacian, my_log_edge
    )
    results.update({
        'Roberts': my_roberts(test_img),
        'Sobel': my_sobel(test_img),
        'Prewitt': my_prewitt(test_img),
        'Canny': my_canny(test_img, 40, 100),
        'Laplacian': my_laplacian(test_img),
        'LoG(σ=2)': my_log_edge(test_img, 2.0),
    })
    for name, r in results.items():
        if name in ['Roberts', 'Sobel', 'Prewitt', 'Canny', 'Laplacian', 'LoG(σ=2)']:
            print(f"  {name}: range=[{r.min():.1f}, {r.max():.1f}]")

    # 2. 霍夫直线
    print("\n[2] 霍夫直线检测...")
    from module3_traditional_seg.my_hough import my_hough_lines
    canny_edges = results['Canny']
    lines = my_hough_lines(canny_edges, threshold=80)
    print(f"  检测到 {len(lines)} 条直线")
    if lines:
        print(f"  最强直线: ρ={lines[0][0]:.1f}, θ={lines[0][1]:.2f}rad, votes={lines[0][2]}")

    # 3. GLCM 纹理
    print("\n[3] GLCM 纹理特征...")
    from module3_traditional_seg.my_texture import my_glcm, my_glcm_features, my_lbp, my_gabor_filter
    glcm_0 = my_glcm(test_img, angle=0)
    features = my_glcm_features(glcm_0)
    for name, val in features.items():
        print(f"  {name}: {val:.4f}")

    # 4. LBP
    print("\n[4] LBP 局部二值模式...")
    # 降采样以加速
    small_img = test_img[::2, ::2]
    lbp_map = my_lbp(small_img.astype(np.float64))
    results['LBP'] = lbp_map.astype(np.float64)
    print(f"  LBP特征图: shape={lbp_map.shape}, unique values={len(np.unique(lbp_map))}")

    # 5. Gabor
    print("\n[5] Gabor 小波...")
    gabor_resp = my_gabor_filter(small_img, theta=0, frequency=0.1)
    results['Gabor(0°)'] = gabor_resp
    print(f"  Gabor响应: range=[{gabor_resp.min():.2f}, {gabor_resp.max():.2f}]")

    # 6. 阈值分割
    print("\n[6] 阈值分割...")
    from module3_traditional_seg.my_threshold import my_iterative_threshold, my_otsu

    bin_iter, T_iter = my_iterative_threshold(test_img)
    bin_otsu, T_otsu = my_otsu(test_img)
    print(f"  迭代阈值法: T={T_iter:.1f}, 前景={bin_iter.sum()/bin_iter.size*100:.1f}%")
    print(f"  Otsu: T={T_otsu}, 前景={bin_otsu.sum()/bin_otsu.size*100:.1f}%")
    results['迭代阈值'] = bin_iter
    results['Otsu'] = bin_otsu

    # 7. 区域生长
    print("\n[7] 区域生长...")
    from module3_traditional_seg.my_region_growing import my_region_grow
    # 自动找种子点: 选取亮度最高的区域中心
    bright_mask = test_img > np.percentile(test_img, 90)
    if bright_mask.any():
        bright_ys, bright_xs = np.where(bright_mask)
        center_idx = len(bright_ys) // 2
        seed = (bright_ys[center_idx], bright_xs[center_idx])
    else:
        seed = (test_img.shape[0]//2, test_img.shape[1]//2)

    rg_mask = my_region_grow(test_img, seed, threshold=20)
    results['区域生长'] = rg_mask.astype(np.float64) * 255
    print(f"  种子点: {seed}, 生长区域: {rg_mask.sum()} pixels")

    # 8. K-means
    print("\n[8] K-means 聚类...")
    from module3_traditional_seg.my_kmeans import my_kmeans_gray
    km_labels, km_centers = my_kmeans_gray(test_img, k=3)
    # 将最大均值类标记为骨骼
    bone_cluster = np.argmax(km_centers)
    bone_mask = (km_labels == bone_cluster).astype(np.float64) * 255
    results['K-means(骨骼)'] = bone_mask
    print(f"  聚类中心: {km_centers.round(1)}, 骨骼类: {bone_cluster}")

    # 保存结果
    print("\n[9] 保存可视化结果...")
    plot_keys = ['Roberts', 'Sobel', 'Canny', 'LoG(σ=2)',
                 'LBP', 'Gabor(0°)', '迭代阈值', 'Otsu',
                 '区域生长', 'K-means(骨骼)']
    n_cols = 5
    n_rows = (len(plot_keys) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for idx, key in enumerate(plot_keys):
        if idx < len(axes):
            axes[idx].imshow(results[key], cmap='gray')
            axes[idx].set_title(key, fontsize=9)
            axes[idx].axis('off')

    for idx in range(len(plot_keys), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "test_module3.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  已保存到: {out_path}")

    print("\n[OK] 模块3 测试通过!")
    return True


if __name__ == "__main__":
    test_module3()
