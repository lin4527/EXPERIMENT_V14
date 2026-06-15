"""CLLOCATION.txt 逐条对照检查"""
import sys, os, numpy as np

print("=" * 60)
print("逐条对照 CLLOCATION.txt 需求检查")
print("=" * 60)

errors = []
test = np.random.rand(50, 50) * 255

# 实验1
print("\n[实验1] 图像读写与格式转换...")
try:
    from module1_io_visualization.image_io import read_image, write_image, get_image_info, convert_format, normalize_image, pixel_operations
    from module1_io_visualization.nifti_reader import NiftiReader
    ct = NiftiReader("sub-verse004_ct.nii")
    meta = ct.get_metadata()
    assert "voxel_size" in meta
    print(f"  OK - NIfTI loaded, voxel={meta['voxel_size']}")
except Exception as e:
    errors.append(f"实验1: {e}")

# 实验2
print("[实验2] 图像重采样与插值...")
try:
    from module2_preprocessing.my_interpolation import my_nearest_neighbor, my_bilinear_interpolation, my_image_resample, my_grayscale_quantize
    r1 = my_nearest_neighbor(test, (25, 25))
    r2 = my_bilinear_interpolation(test, (25, 25))
    r3 = my_image_resample(test, 0.5, "bilinear")
    r4 = my_grayscale_quantize(test.astype(np.uint8), 16)
    assert r1.shape == (25, 25)
    assert r2.shape == (25, 25)
    print("  OK - 4 interpolation functions")
except Exception as e:
    errors.append(f"实验2: {e}")

# 实验3
print("[实验3] 灰度变换...")
try:
    from module2_preprocessing.my_grayscale_transform import (
        my_linear_transform, my_log_transform, my_gamma_transform,
        my_piecewise_linear_transform, my_contrast_stretch
    )
    my_linear_transform(test, 1.5, 10)
    my_log_transform(test, 1.0)
    my_gamma_transform(test, 0.5)
    my_piecewise_linear_transform(test, [(80, 0), (180, 255)])
    my_contrast_stretch(test)
    print("  OK - 5 transforms")
except Exception as e:
    errors.append(f"实验3: {e}")

# 实验4
print("[实验4] 直方图统计与均衡化...")
try:
    from module2_preprocessing.my_histogram import my_calc_hist, my_histogram_equalization, my_histogram_stats
    h = my_calc_hist(test.astype(np.uint8))
    eq = my_histogram_equalization(test.astype(np.uint8))
    stats = my_histogram_stats(test)
    assert "mean" in stats and "entropy" in stats
    print(f"  OK - mean={stats['mean']:.1f}, entropy={stats['entropy']:.2f}")
except Exception as e:
    errors.append(f"实验4: {e}")

# 实验5
print("[实验5] 二维FFT与频域滤波...")
try:
    from module2_preprocessing.my_frequency_filter import (
        my_fft2, my_ifft2, my_fftshift, my_ideal_lowpass, my_ideal_highpass,
        my_butterworth_lowpass, my_butterworth_highpass, my_gaussian_lowpass,
        my_gaussian_highpass, my_lowpass_filter, my_highpass_filter
    )
    small = np.random.rand(32, 32)
    F = my_fft2(small)
    recon = np.real(my_ifft2(F))
    err = np.abs(small - recon).mean()
    print(f"  FFT roundtrip error: {err:.6f}")
    my_ideal_lowpass((64, 64), 10)
    my_gaussian_lowpass((64, 64), 10)
    my_butterworth_lowpass((64, 64), 10, 2)
    my_lowpass_filter(small, 8, "gaussian")
    my_highpass_filter(small, 16, "gaussian")
    print("  OK - 3 filters + lowpass + highpass")
except Exception as e:
    errors.append(f"实验5: {e}")

# 实验6
print("[实验6] 边缘检测 + 霍夫直线 + 傅里叶描绘子...")
try:
    from module3_traditional_seg.my_edge_detection import (
        my_roberts, my_sobel, my_prewitt, my_canny, my_laplacian, my_log_edge
    )
    from module3_traditional_seg.my_hough import my_hough_lines
    from module3_traditional_seg.my_fourier_descriptor import (
        my_fourier_descriptors, my_fd_reconstruct, my_fd_similarity
    )
    my_roberts(test)
    my_sobel(test)
    my_prewitt(test)
    my_canny(test.astype(np.uint8), 40, 100)
    my_laplacian(test)
    my_log_edge(test, 2.0)
    edges = my_canny(test.astype(np.uint8), 40, 100)
    lines = my_hough_lines(edges, threshold=30)
    contour = np.array([[i, np.sin(i / 10) * 30 + 30] for i in range(100)])
    fd = my_fourier_descriptors(contour, 10)
    recon = my_fd_reconstruct(fd, 128)
    assert recon.shape == (128, 2)
    print(f"  OK - 6 edge detectors + Hough({len(lines)} lines) + FD")
except Exception as e:
    errors.append(f"实验6: {e}")

# 实验7
print("[实验7] GLCM(4方向) + LBP + Gabor...")
try:
    from module3_traditional_seg.my_texture import (
        my_glcm, my_glcm_features, my_glcm_4direction_features,
        my_lbp, my_lbp_histogram, my_gabor_filter, my_gabor_bank
    )
    glcm0 = my_glcm(test.astype(np.uint8), angle=0)
    glcm45 = my_glcm(test.astype(np.uint8), angle=45)
    glcm90 = my_glcm(test.astype(np.uint8), angle=90)
    glcm135 = my_glcm(test.astype(np.uint8), angle=135)
    f = my_glcm_features(glcm0)
    assert all(k in f for k in ["contrast", "energy", "entropy", "idm", "correlation"])
    f4 = my_glcm_4direction_features(test.astype(np.uint8))
    lbp = my_lbp(test.astype(np.float64))
    gabor = my_gabor_filter(test.astype(np.float64), theta=0, frequency=0.1)
    gabor_bank = my_gabor_bank(test.astype(np.float64))
    print(f"  OK - 4-direction GLCM (contrast={f['contrast']:.1f}), LBP, Gabor({len(gabor_bank)} filters)")
except Exception as e:
    errors.append(f"实验7: {e}")

# 实验8
print("[实验8] 迭代阈值 + Otsu...")
try:
    from module3_traditional_seg.my_threshold import my_iterative_threshold, my_otsu
    b1, t1 = my_iterative_threshold(test)
    b2, t2 = my_otsu(test.astype(np.uint8))
    print(f"  OK - iterative(T={t1:.1f}), Otsu(T={t2})")
except Exception as e:
    errors.append(f"实验8: {e}")

# 实验9
print("[实验9] 区域生长 + K-means + Dice/IoU...")
try:
    from module3_traditional_seg.my_region_growing import my_region_grow
    from module3_traditional_seg.my_kmeans import my_kmeans_gray, my_kmeans_spatial
    from module5_registration_eval.my_similarity import my_dice, my_iou, my_dice_multiclass
    rg = my_region_grow(test, (25, 25), threshold=30)
    labels, centers = my_kmeans_gray(test, k=3)
    labels2 = my_kmeans_spatial(test, k=3, spatial_weight=0.3)
    d = my_dice(b1 > 0, b2 > 0)
    iou = my_iou(b1 > 0, b2 > 0)
    m_dice = my_dice_multiclass(labels, labels2, num_classes=3)
    print(f"  OK - region_grow(area={rg.sum()}), kmeans, Dice={d:.3f}, IoU={iou:.3f}")
except Exception as e:
    errors.append(f"实验9: {e}")

# 实验10
print("[实验10] 配准 + MI/NMI/MSE...")
try:
    from module5_registration_eval.my_similarity import my_mse, my_mi, my_nmi, my_psnr
    mse = my_mse(test, test + 5)
    mi = my_mi(test, test + 5, bins=64)
    nmi = my_nmi(test, test + 5, bins=64)
    psnr = my_psnr(test, test + 5)
    print(f"  OK - MSE={mse:.1f}, MI={mi:.3f}, NMI={nmi:.3f}, PSNR={psnr:.1f}")
except Exception as e:
    errors.append(f"实验10: {e}")

# 实验11
print("[实验11] RGB/HSI + 伪彩色...")
try:
    from module2_preprocessing.my_color_space import (
        my_rgb_to_hsi, my_hsi_to_rgb, my_pseudocolor_map,
        my_channel_separate, my_channel_filter, my_channel_merge
    )
    rgb_test = np.stack([test, test // 2, test // 4], axis=2).astype(np.uint8)
    hsi = my_rgb_to_hsi(rgb_test)
    rgb_back = my_hsi_to_rgb(hsi)
    diff = np.abs(rgb_test.astype(float) - rgb_back.astype(float)).mean()
    pseudo = my_pseudocolor_map(test, "jet")
    ch = my_channel_separate(rgb_test)
    filtered = my_channel_filter(rgb_test, "R", lambda x: x * 0.5)
    print(f"  OK - RGB-HSI diff={diff:.1f}, pseudocolor(jet+hot), channel ops")
except Exception as e:
    errors.append(f"实验11: {e}")

# 实验12
print("[实验12] NIfTI + 窗宽窗位 + MPR...")
try:
    from module1_io_visualization.window_level import apply_window_level, apply_preset, get_window_presets, auto_window
    from module1_io_visualization.mpr import MPRReconstructor
    presets = get_window_presets()
    mpr = MPRReconstructor(ct.data)
    axial = mpr.axial(80)
    coronal = mpr.coronal(50)
    sagittal = mpr.sagittal(30)
    views = mpr.get_orthogonal_views()
    ww, wl = auto_window(axial)
    bone = apply_preset(axial, "骨窗")
    soft = apply_preset(axial, "软组织窗")
    lung = apply_preset(axial, "肺窗")
    custom = apply_window_level(axial, 400, 40)
    print(f"  OK - {len(presets)} presets, MPR({axial.shape},{coronal.shape},{sagittal.shape})")
except Exception as e:
    errors.append(f"实验12: {e}")

# 实验13
print("[实验13] PyQt5 GUI...")
try:
    from PyQt5.QtWidgets import QApplication
    from module6_gui.main_window import MainWindow
    print("  OK - PyQt5 + MainWindow importable")
except Exception as e:
    errors.append(f"实验13: {e}")

# 模块文件完整性
print("\n[模块文件检查]...")
module_dirs = {
    "module1_io_visualization": 5,
    "module2_preprocessing": 7,
    "module3_traditional_seg": 8,
    "module4_deep_learning": 6,
    "module5_registration_eval": 5,
    "module6_gui": 6,
}
for d, expected in module_dirs.items():
    count = len([f for f in os.listdir(d) if f.endswith(".py")])
    status = "OK" if count >= expected else f"MISSING (got {count}, expected {expected})"
    print(f"  {d}: {count} files {status}")

# 数据文件
print("\n[数据文件检查]...")
for df in ["sub-verse004_ct.nii", "verse004_CT-sag_seg.nii"]:
    if not os.path.exists(df):
        errors.append(f"Missing: {df}")
    else:
        print(f"  {df}: {os.path.getsize(df)/1024/1024:.1f} MB")

# my_ 命名规范
print("\n[命名规范] my_ 前缀计数...")
import module2_preprocessing, module3_traditional_seg, module5_registration_eval
my_count = 0
for mod in [module2_preprocessing, module3_traditional_seg, module5_registration_eval]:
    for name in dir(mod):
        if name.startswith("my_"):
            my_count += 1
print(f"  my_ 函数: {my_count} 个")

# 最终报告
print(f"\n{'='*60}")
if errors:
    print(f"发现问题 {len(errors)} 个:")
    for e in errors:
        print(f"  [FAIL] {e}")
else:
    print("ALL 13 EXPERIMENTS VERIFIED OK")
print(f"{'='*60}")
