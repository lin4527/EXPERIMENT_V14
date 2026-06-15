"""
全局配置文件 — 医学图像处理综合实验 14
"""
import os

# 工程根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据文件路径
CT_PATH = os.path.join(ROOT_DIR, "sub-verse004_ct.nii")
SEG_PATH = os.path.join(ROOT_DIR, "verse004_CT-sag_seg.nii")

# 输出目录
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 窗宽窗位预设 (WW/WL, HU单位)
WINDOW_PRESETS = {
    "骨窗":     (1500, 300),
    "软组织窗":  (400, 40),
    "肺窗":     (1500, -500),
    "脑窗":     (80, 40),
    "纵隔窗":   (400, 40),
    "自定义":   (400, 40),
}

# MPR 默认切片索引比例
MPR_DEFAULT_RATIO = 0.5  # 中间切片

# 图像处理默认参数
DEFAULT_GAMMA = 1.0
DEFAULT_LOG_C = 1.0
DEFAULT_BRIGHTNESS = 0
DEFAULT_CONTRAST = 1.0

# 频域滤波默认参数
DEFAULT_CUTOFF_RATIO = 0.1  # 截止频率比例
DEFAULT_BUTTERWORTH_ORDER = 2

# GLCM 参数
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, 45, 135, 90]  # 对应 0°/45°/90°/135°

# 深度学习参数
DL_BATCH_SIZE = 2
DL_EPOCHS = 10
DL_LEARNING_RATE = 1e-4
DL_IMAGE_SIZE = 256
DL_NUM_CLASSES = 10  # 包含背景

# GUI 参数
GUI_DEFAULT_WIDTH = 1280
GUI_DEFAULT_HEIGHT = 768
GUI_CANVAS_SIZE = 400
