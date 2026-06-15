"""
模块1 独立测试脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from config import CT_PATH, SEG_PATH

# 测试 NIfTI 读取
from module1_io_visualization.nifti_reader import NiftiReader
from module1_io_visualization.window_level import apply_window_level, apply_preset, auto_window
from module1_io_visualization.mpr import MPRReconstructor
from module1_io_visualization.image_io import read_image, write_image, get_image_info


def test_module1():
    print("=" * 60)
    print("模块1 测试：医学影像数据读取与可视化")
    print("=" * 60)

    # 1. NIfTI 读取
    print("\n[1] 加载 CT 数据...")
    ct_reader = NiftiReader(CT_PATH)
    print(f"  数据维度: {ct_reader.shape}")
    print(f"  元数据: {ct_reader.get_metadata()}")

    print("\n[2] 加载标注数据...")
    seg_reader = NiftiReader(SEG_PATH)
    print(f"  标注维度: {seg_reader.shape}")
    print(f"  标签值: {np.unique(seg_reader.data).astype(int)[:15]}...")  # 只显示前15个

    # 2. MPR 重建
    print("\n[3] MPR 多平面重建...")
    mpr = MPRReconstructor(ct_reader.data)
    views = mpr.get_orthogonal_views()
    for name, slice_data in views.items():
        print(f"  {name}: shape={slice_data.shape}, range=[{slice_data.min():.0f}, {slice_data.max():.0f}]")

    # 3. 窗宽窗位
    print("\n[4] 窗宽窗位调节...")
    axial_slice = mpr.axial()
    bone_win = apply_preset(axial_slice, "骨窗")
    soft_win = apply_preset(axial_slice, "软组织窗")
    lung_win = apply_preset(axial_slice, "肺窗")
    auto_ww, auto_wl = auto_window(axial_slice)
    print(f"  骨窗: range=[{bone_win.min()}, {bone_win.max()}]")
    print(f"  软组织窗: range=[{soft_win.min()}, {soft_win.max()}]")
    print(f"  肺窗: range=[{lung_win.min()}, {lung_win.max()}]")
    print(f"  自动窗宽窗位: WW={auto_ww}, WL={auto_wl}")

    # 4. 可视化
    print("\n[5] 保存可视化结果...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    titles = ['Axial (骨窗)', 'Coronal (骨窗)', 'Sagittal (骨窗)',
              'Axial (软组织窗)', 'Axial (肺窗)', 'Axial (自动窗)',
              'Seg Axial', 'Seg Sagittal']
    images = [
        apply_preset(mpr.axial(), "骨窗"),
        apply_preset(mpr.coronal(), "骨窗"),
        apply_preset(mpr.sagittal(), "骨窗"),
        soft_win,
        lung_win,
        apply_window_level(axial_slice, auto_ww, auto_wl),
        seg_reader.get_center_slice('axial'),
        seg_reader.get_center_slice('sagittal'),
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "test_module1.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  已保存到: {out_path}")

    print("\n[OK] 模块1 测试通过!")
    return True


if __name__ == "__main__":
    test_module1()
