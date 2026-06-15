# 医学图像处理综合实验 14 — 脊椎椎体逐节分割系统

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1+cu118-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.1-green)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/GPU-RTX%203050%208GB-orange)]()

基于 Python 的完整医学图像处理教学实验系统，覆盖课程前 **13 个实验的全部核心技术点**，所有基础算法以 `my_` 前缀手动编码实现原理，最终通过 **2D U-Net 深度学习** 实现脊椎椎体自动逐节分割。

> 🏥 数据来源：verse004 公开数据集 | 📐 74 个手动实现算法 | 🧠 Dice=0.73 @RTX3050

---

## 📸 效果预览

```
原始 CT 矢状面  →  U-Net 分割  →  金标准标注
    🦴🦴🦴           🟥🟧🟨🟩🟦        🔴🔴🔴
    一节节椎体        逐节分离          专业标注
```

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 GUI
python main.py

# 3. 运行测试
python main.py --test

# 4. 查看原始数据
python view_raw_data.py
```

---

## 📁 工程结构

```
EXPERIMENT_V14/
├── main.py                          # 主入口 (GUI)
├── config.py                        # 全局配置
├── view_raw_data.py                 # 原始数据展示工具
├── check_all.py                     # 13实验需求逐条验证
├── requirements.txt                 # 依赖清单
├── .gitignore
│
├── module1_io_visualization/        # 模块1: 数据读取与可视化
│   ├── nifti_reader.py              #   NIfTI 3D体数据读写
│   ├── image_io.py                  #   通用图像读写/格式转换
│   ├── window_level.py              #   窗宽窗位调节 (骨窗/软组织窗/肺窗)
│   ├── mpr.py                       #   MPR 多平面重建 (轴/冠/矢)
│   └── test_module1.py
│
├── module2_preprocessing/           # 模块2: 全算子图像预处理
│   ├── my_grayscale_transform.py    #   ⭐ 线性/对数/Gamma/分段线性变换
│   ├── my_histogram.py              #   ⭐ 直方图统计 + 均衡化
│   ├── my_frequency_filter.py       #   ⭐ 2D FFT/IFFT + 理想/巴特沃斯/高斯滤波
│   ├── my_interpolation.py          #   ⭐ 最近邻/双线性插值 + 灰度量化
│   ├── my_color_space.py            #   ⭐ RGB↔HSI + 伪彩色映射
│   ├── lib_reference.py             #   库函数对照版本
│   └── test_module2.py
│
├── module3_traditional_seg/         # 模块3: 传统方法分割基线
│   ├── my_edge_detection.py         #   ⭐ 6种边缘检测 + Canny完整流程
│   ├── my_hough.py                  #   ⭐ 霍夫直线检测
│   ├── my_fourier_descriptor.py     #   ⭐ 傅里叶描绘子
│   ├── my_texture.py                #   ⭐ 4方向GLCM + LBP + Gabor小波
│   ├── my_threshold.py              #   ⭐ 迭代阈值 + Otsu
│   ├── my_region_growing.py         #   ⭐ 区域生长 (BFS八邻域)
│   ├── my_kmeans.py                 #   ⭐ K-means聚类 (灰度/灰度+空间)
│   └── test_module3.py
│
├── module4_deep_learning/           # 模块4: 深度学习椎体分割
│   ├── unet_model.py                #   2D U-Net 网络定义
│   ├── dataset.py                   #   数据集加载 + 数据增强
│   ├── train.py                     #   训练循环 (Dice+CE联合损失)
│   ├── inference.py                 #   推理预测
│   └── postprocess.py               #   后处理 + 椎体编号
│
├── module5_registration_eval/       # 模块5: 配准后处理与量化评估
│   ├── my_registration.py           #   ⭐ 2D刚性/仿射配准
│   ├── my_similarity.py             #   ⭐ Dice/IoU/MI/NMI/MSE/PSNR
│   ├── my_morphology.py             #   ⭐ 腐蚀/膨胀/开运算/闭运算
│   └── evaluator.py                 #   综合评估器 + 自动对比报告
│
├── module6_gui/                     # 模块6: GUI交互集成系统
│   ├── main_window.py               #   主窗口 (PyQt5)
│   ├── canvas_widgets.py            #   MPR三视图 + 双画布
│   ├── control_panels.py            #   5组功能面板
│   ├── worker_threads.py            #   后台计算线程 (QThread)
│   └── result_display.py            #   显示模式 (差值图/伪彩色/轮廓)
│
├── 使用教程.md                       #   📖 完整操作教程
├── 实验目的.md                       #   📝 实验目标说明
├── 实验环境与工具链.md               #   🔧 AI辅助开发工具链
├── 实验报告.md                       #   📄 完整实验报告
│
└── output/                          #   运行时输出 (自动创建)
```

---

## 🔬 技术覆盖总览

| 实验 | 核心技术 | 实现文件 | 关键函数 |
|------|---------|---------|---------|
| 1 | 图像读写/格式转换/矩阵运算 | `image_io.py`, `nifti_reader.py` | `read_image()`, `write_image()` |
| 2 | 最近邻/双线性插值/重采样/量化 | `my_interpolation.py` | `my_bilinear_interpolation()` |
| 3 | 线性/对数/Gamma/分段线性变换 | `my_grayscale_transform.py` | `my_gamma_transform()` |
| 4 | 直方图统计 + 均衡化 | `my_histogram.py` | `my_histogram_equalization()` |
| 5 | 2D FFT/IFFT/低通/高通滤波 | `my_frequency_filter.py` | `my_fft2()`, `my_gaussian_lowpass()` |
| 6 | 6种边缘检测 + 霍夫直线 + 傅里叶描绘子 | `my_edge_detection.py` | `my_canny()`, `my_hough_lines()` |
| 7 | 4方向GLCM + LBP + Gabor小波 | `my_texture.py` | `my_glcm()`, `my_lbp()` |
| 8 | 迭代阈值 + Otsu | `my_threshold.py` | `my_otsu()` |
| 9 | 区域生长 + K-means + Dice/IoU | `my_region_growing.py` | `my_region_grow()` |
| 10 | 2D配准 + MI/NMI/MSE | `my_registration.py` | `my_rigid_registration()` |
| 11 | RGB↔HSI + 伪彩色 + 通道滤波 | `my_color_space.py` | `my_rgb_to_hsi()` |
| 12 | NIfTI读写 + 窗宽窗位 + MPR | `window_level.py`, `mpr.py` | `apply_window_level()` |
| 13 | PyQt5 GUI | `module6_gui/` | `MainWindow` |

---

## 🧠 深度学习分割

### 网络架构
```
输入 (1,256,256)
    │
[编码器]  C1(64) → C2(128) → C3(256) → C4(512)
           │          │          │          │
           │ 跳跃连接  │          │          │
           ↓          ↓          ↓          ↓
[解码器]  C1'(64)← C2'(128)← C3'(256)← C4'(512)  ← [瓶颈 C5(1024)]
    │
输出 (10,256,256)  ← 10类逐像素概率
```

### 训练结果
| Epoch | Train Loss | Val Dice | 备注 |
|-------|-----------|----------|------|
| 1 | 2.892 | 0.005 | 随机初始化 |
| 4 | 2.210 | 0.026 | 开始学习 |
| 8 | 1.782 | 0.522 | 🔓 "开窍" |
| **12** | **1.577** | **0.729** | 🏆 **最佳** |
| 17 | 1.467 | 0.682 | 过拟合，早停 |

- **硬件**：NVIDIA RTX 3050 Laptop GPU (8GB)
- **耗时**：96 秒 / 20 epochs
- **提升**：较传统 K-means (Dice=0.61) 提升 **19.5%**

---

## 📊 多方法对比

| 方法 | Dice ↑ | 类型 | 自动化 |
|------|--------|------|--------|
| K-means (k=3) | 0.610 | 传统聚类 | ✅ |
| Otsu 阈值 | 0.512 | 传统阈值 | ✅ |
| 迭代阈值 | 0.498 | 传统阈值 | ✅ |
| 区域生长 | 0.423 | 传统交互 | ❌ 需种子点 |
| **U-Net (ours)** | **0.729** | **深度学习** | ✅ |

---

## 🛠 AI 辅助开发工具链

本实验采用 **Claude Code + DeepSeek-v4-pro [1M]** 双模型协同开发架构：

| 环节 | AI 参与 | 产出 |
|------|---------|------|
| 架构设计 | Plan Agent 生成 6 模块结构 | 43 文件目录树 |
| 算法实现 | 主推理引擎编码 74 个 `my_` 函数 | 5900 行代码 |
| 缺陷检测 | Multi-Agent 交叉审查 | 发现并修复 9 个关键 bug |
| GPU 训练 | 参数自动调优 | 96s/20epochs, Dice=0.73 |
| 全局审查 | 全工程代码扫描 | 补全 `__init__.py`、统一命名 |
| 文档生成 | 教程/报告/目的/工具链 | 4 份 Markdown 文档 |

详见 [实验环境与工具链.md](实验环境与工具链.md)

---

## 📄 文档

| 文档 | 说明 |
|------|------|
| [使用教程.md](使用教程.md) | 完整 GUI 操作教程，每个按钮功能详解 |
| [实验目的.md](实验目的.md) | 实验目标与技术路线 |
| [实验环境与工具链.md](实验环境与工具链.md) | AI 辅助开发工具链说明 |
| [实验报告.md](实验报告.md) | 完整实验报告（含公式、原理、结果分析） |

---

## 📝 运行环境

- Python 3.8+
- NumPy, OpenCV, Matplotlib, SciPy
- NiBabel, SimpleITK
- PyTorch 2.0+ (GPU 推荐)
- PyQt5

```bash
pip install -r requirements.txt
```

---

## 🤝 AI 协作说明

本项目由 **林** + **Claude Code (Anthropic)** + **DeepSeek-v4-pro** 协同完成。

```
Co-Authored-By: Claude Code <noreply@anthropic.com>
Co-Authored-By: DeepSeek-v4-pro <noreply@deepseek.com>
```

---

## 📜 License

MIT License — 仅供学习与教学使用。
