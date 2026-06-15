# 医学图像处理综合实验 — 脊椎椎体逐节分割

基于 Python 实现的医学图像处理系统，对脊椎 CT 数据进行逐节椎体分割。手动实现了图像读写、插值、灰度变换、直方图均衡化、FFT 频域滤波、边缘检测、纹理分析、阈值分割、区域生长、K-means 聚类、图像配准、Dice/IoU 评估等 50 多个核心算法，最后用 2D U-Net 做了深度学习分割。

## 数据

- `sub-verse004_ct.nii` — CT 3D 体数据 (161×338×61, HU 值)
- `verse004_CT-sag_seg.nii` — 椎体标注 (9 节椎体，标签 16-24)

## 运行

```bash
pip install -r requirements.txt
python main.py           # 启动 GUI
python main.py --test     # 运行测试
python view_raw_data.py   # 查看原始数据
```

## 目录结构

```
module1_io_visualization/      NIfTI 读写、窗宽窗位、MPR 多平面重建
module2_preprocessing/         灰度变换、直方图、FFT 频域滤波、插值、色彩空间
module3_traditional_seg/       边缘检测、霍夫直线、GLCM/LBP/Gabor、阈值、区域生长、K-means
module4_deep_learning/         2D U-Net 训练与推理
module5_registration_eval/     配准、相似性度量、形态学处理、评估器
module6_gui/                   PyQt5 交互界面
```

## 实验覆盖

| 实验 | 内容 | 关键函数 |
|------|------|---------|
| 1 | 图像读写/格式转换 | `read_image()`, `write_image()`, `NiftiReader` |
| 2 | 插值/重采样/量化 | `my_bilinear_interpolation()`, `my_grayscale_quantize()` |
| 3 | 灰度变换 | `my_gamma_transform()`, `my_log_transform()` |
| 4 | 直方图均衡化 | `my_histogram_equalization()` |
| 5 | 2D FFT 频域滤波 | `my_fft2()`, `my_gaussian_lowpass()` |
| 6 | 边缘检测/霍夫直线 | `my_canny()`, `my_hough_lines()`, `my_fourier_descriptors()` |
| 7 | GLCM/LBP/Gabor 纹理 | `my_glcm()`, `my_lbp()`, `my_gabor_filter()` |
| 8 | 阈值分割 | `my_otsu()`, `my_iterative_threshold()` |
| 9 | 区域生长/K-means/Dice | `my_region_grow()`, `my_kmeans()`, `my_dice()` |
| 10 | 图像配准/MI/NMI | `my_rigid_registration()`, `my_mi()` |
| 11 | RGB-HSI/伪彩色 | `my_rgb_to_hsi()`, `my_pseudocolor_map()` |
| 12 | NIfTI/窗宽窗位/MPR | `apply_window_level()`, `MPRReconstructor` |
| 13 | PyQt5 GUI | `MainWindow` |

## U-Net 训练

- 网络：2D U-Net，~7.8M 参数
- 输入：256×256 矢状面切片
- 损失函数：CrossEntropy + Dice Loss
- 硬件：RTX 3050 8GB，训练约 90 秒
- 验证 Dice：0.73

## 环境

- Python 3.10
- PyTorch 2.7.1+cu118
- NumPy, OpenCV, SciPy, Matplotlib
- NiBabel, SimpleITK
- PyQt5
