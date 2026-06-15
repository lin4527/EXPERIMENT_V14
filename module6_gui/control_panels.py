"""
功能面板组件 — 左侧面板 (5组: 文件/预处理/分割/深度学习/评估)
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QComboBox, QSlider, QLabel,
                             QSpinBox, QDoubleSpinBox, QTextEdit,
                             QFileDialog, QScrollArea, QCheckBox, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal


class FilePanel(QGroupBox):
    """文件导入面板"""
    load_ct = pyqtSignal(str)
    load_seg = pyqtSignal(str)
    save_result = pyqtSignal(str)

    def __init__(self):
        super().__init__("📁 文件导入")
        layout = QVBoxLayout()

        self.btn_load_ct = QPushButton("加载 CT 数据")
        self.btn_load_seg = QPushButton("加载标注数据")
        self.btn_save = QPushButton("保存当前结果")
        self.lbl_ct_path = QLabel("未加载")
        self.lbl_seg_path = QLabel("未加载")

        self.lbl_ct_path.setWordWrap(True)
        self.lbl_seg_path.setWordWrap(True)

        layout.addWidget(self.btn_load_ct)
        layout.addWidget(self.lbl_ct_path)
        layout.addWidget(self.btn_load_seg)
        layout.addWidget(self.lbl_seg_path)
        layout.addWidget(self.btn_save)

        self.btn_load_ct.clicked.connect(
            lambda: self._open_file("CT", self.load_ct))
        self.btn_load_seg.clicked.connect(
            lambda: self._open_file("标注", self.load_seg))
        self.btn_save.clicked.connect(
            lambda: self._save_file())

        self.setLayout(layout)

    def _open_file(self, title, signal):
        path, _ = QFileDialog.getOpenFileName(
            self, f"选择{title}文件", "",
            "Medical Images (*.nii *.nii.gz *.dcm *.png *.jpg);;All Files (*)"
        )
        if path:
            signal.emit(path)

    def _save_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "PNG (*.png);;JPG (*.jpg);;NIfTI (*.nii)"
        )
        if path:
            self.save_result.emit(path)


class PreprocessPanel(QGroupBox):
    """预处理面板"""
    apply_transform = pyqtSignal(str, dict)  # (transform_name, params)
    apply_filter = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__("🔧 预处理")
        layout = QVBoxLayout()

        # 窗宽窗位
        layout.addWidget(QLabel("窗宽窗位预设:"))
        self.cmb_window = QComboBox()
        self.cmb_window.addItems(["骨窗", "软组织窗", "肺窗", "脑窗", "纵隔窗"])
        layout.addWidget(self.cmb_window)

        self.btn_apply_window = QPushButton("应用窗宽窗位")
        layout.addWidget(self.btn_apply_window)

        # 灰度变换
        layout.addWidget(QLabel("灰度变换:"))
        self.cmb_gray = QComboBox()
        self.cmb_gray.addItems(["线性变换", "对数变换", "Gamma变换", "分段线性", "对比度拉伸"])
        layout.addWidget(self.cmb_gray)

        # 参数
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Gamma/斜率:"))
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.1, 5.0)
        self.spin_gamma.setValue(1.0)
        self.spin_gamma.setSingleStep(0.1)
        hl.addWidget(self.spin_gamma)
        layout.addLayout(hl)

        self.btn_apply_gray = QPushButton("应用灰度变换")
        layout.addWidget(self.btn_apply_gray)

        # 直方图
        self.btn_hist_eq = QPushButton("直方图均衡化")
        layout.addWidget(self.btn_hist_eq)

        # 频域滤波
        layout.addWidget(QLabel("频域滤波:"))
        fl = QHBoxLayout()
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems(["高斯低通", "高斯高通", "理想低通", "巴特沃斯低通"])
        fl.addWidget(self.cmb_filter)
        self.spin_cutoff = QSpinBox()
        self.spin_cutoff.setRange(1, 100)
        self.spin_cutoff.setValue(10)
        fl.addWidget(QLabel("截止:"))
        fl.addWidget(self.spin_cutoff)
        layout.addLayout(fl)
        self.btn_apply_filter = QPushButton("应用频域滤波")
        layout.addWidget(self.btn_apply_filter)

        # 插值
        layout.addWidget(QLabel("重采样:"))
        rl = QHBoxLayout()
        self.cmb_interp = QComboBox()
        self.cmb_interp.addItems(["最近邻", "双线性"])
        rl.addWidget(self.cmb_interp)
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.1, 4.0)
        self.spin_scale.setValue(1.0)
        self.spin_scale.setSingleStep(0.1)
        rl.addWidget(QLabel("倍率:"))
        rl.addWidget(self.spin_scale)
        layout.addLayout(rl)
        self.btn_resample = QPushButton("重采样")
        layout.addWidget(self.btn_resample)

        # 色阶量化
        ql = QHBoxLayout()
        ql.addWidget(QLabel("灰度量化:"))
        self.spin_levels = QSpinBox()
        self.spin_levels.setRange(2, 256)
        self.spin_levels.setValue(256)
        ql.addWidget(self.spin_levels)
        layout.addLayout(ql)
        self.btn_quantize = QPushButton("灰度量化")
        layout.addWidget(self.btn_quantize)

        self.setLayout(layout)

    def get_transform_params(self):
        return {
            "gamma": self.spin_gamma.value(),
            "transform_type": self.cmb_gray.currentText(),
            "window": self.cmb_window.currentText(),
        }

    def get_filter_params(self):
        return {
            "filter_type": self.cmb_filter.currentText(),
            "cutoff": self.spin_cutoff.value(),
        }


class SegmentationPanel(QGroupBox):
    """传统分割方法面板"""
    run_otsu = pyqtSignal()
    run_iterative = pyqtSignal()
    run_region_grow = pyqtSignal(int, int, int)  # row, col, threshold
    run_kmeans = pyqtSignal(int, bool)  # k, use_spatial
    run_edge = pyqtSignal(str)  # edge method
    run_texture = pyqtSignal(str)  # texture method

    def __init__(self):
        super().__init__("🔪 传统分割")
        layout = QVBoxLayout()

        # 边缘检测
        layout.addWidget(QLabel("边缘检测:"))
        self.cmb_edge = QComboBox()
        self.cmb_edge.addItems(["Sobel", "Canny", "Roberts", "Prewitt", "Laplacian", "LoG"])
        layout.addWidget(self.cmb_edge)
        self.btn_edge = QPushButton("执行边缘检测")
        layout.addWidget(self.btn_edge)

        # 阈值分割
        layout.addWidget(QLabel("阈值分割:"))
        bl = QHBoxLayout()
        self.btn_otsu = QPushButton("Otsu")
        self.btn_iterative = QPushButton("迭代阈值")
        bl.addWidget(self.btn_otsu)
        bl.addWidget(self.btn_iterative)
        layout.addLayout(bl)

        # 区域生长
        layout.addWidget(QLabel("区域生长 (点击图像选种子点):"))
        rgl = QHBoxLayout()
        rgl.addWidget(QLabel("阈值:"))
        self.spin_rg_thresh = QSpinBox()
        self.spin_rg_thresh.setRange(1, 100)
        self.spin_rg_thresh.setValue(20)
        rgl.addWidget(self.spin_rg_thresh)
        layout.addLayout(rgl)
        self.lbl_seed = QLabel("种子点: (点击图像选取)")
        layout.addWidget(self.lbl_seed)
        self.btn_region_grow = QPushButton("区域生长")
        layout.addWidget(self.btn_region_grow)

        # K-means
        layout.addWidget(QLabel("K-means 聚类:"))
        kml = QHBoxLayout()
        kml.addWidget(QLabel("K:"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 10)
        self.spin_k.setValue(3)
        kml.addWidget(self.spin_k)
        self.chk_spatial = QCheckBox("灰度+空间")
        kml.addWidget(self.chk_spatial)
        layout.addLayout(kml)
        self.btn_kmeans = QPushButton("K-means 分割")
        layout.addWidget(self.btn_kmeans)

        # 霍夫直线
        self.btn_hough = QPushButton("霍夫直线检测")
        layout.addWidget(self.btn_hough)

        # 纹理分析
        layout.addWidget(QLabel("纹理分析:"))
        self.cmb_texture = QComboBox()
        self.cmb_texture.addItems(["GLCM", "LBP", "Gabor"])
        layout.addWidget(self.cmb_texture)
        self.btn_texture = QPushButton("纹理特征提取")
        layout.addWidget(self.btn_texture)

        layout.addStretch()
        self.setLayout(layout)


class DeepLearningPanel(QGroupBox):
    """深度学习面板"""
    train_signal = pyqtSignal(int)   # epochs
    predict_signal = pyqtSignal()    # run inference
    load_model_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__("🧠 深度学习 (U-Net)")
        layout = QVBoxLayout()

        layout.addWidget(QLabel("网络: 2D U-Net (轻量版)"))
        self.lbl_params = QLabel("参数量: ~1M | CPU可训练")
        layout.addWidget(self.lbl_params)

        layout.addWidget(QLabel("训练配置:"))
        tl = QHBoxLayout()
        tl.addWidget(QLabel("Epochs:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 50)
        self.spin_epochs.setValue(5)
        tl.addWidget(self.spin_epochs)
        layout.addLayout(tl)

        tl2 = QHBoxLayout()
        tl2.addWidget(QLabel("Batch:"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 8)
        self.spin_batch.setValue(2)
        tl2.addWidget(self.spin_batch)
        layout.addLayout(tl2)

        tl3 = QHBoxLayout()
        tl3.addWidget(QLabel("LR:"))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(1e-5, 1e-2)
        self.spin_lr.setValue(1e-4)
        self.spin_lr.setSingleStep(1e-4)
        self.spin_lr.setDecimals(6)
        tl3.addWidget(self.spin_lr)
        layout.addLayout(tl3)

        self.btn_train = QPushButton("🚀 开始训练")
        self.btn_train.setStyleSheet("background-color: #2d5a27; color: white; font-weight: bold;")
        layout.addWidget(self.btn_train)

        self.btn_load_model = QPushButton("📂 加载预训练模型")
        layout.addWidget(self.btn_load_model)

        self.btn_predict = QPushButton("🔮 推理预测 (当前切片)")
        layout.addWidget(self.btn_predict)

        self.progress_label = QLabel("就绪")
        layout.addWidget(self.progress_label)

        self.btn_postprocess = QPushButton("后处理 (连通域+编号)")
        layout.addWidget(self.btn_postprocess)

        layout.addStretch()
        self.setLayout(layout)


class EvalPanel(QGroupBox):
    """评估面板"""
    run_eval = pyqtSignal()
    compare_signal = pyqtSignal()

    def __init__(self):
        super().__init__("📊 量  化评估")
        layout = QVBoxLayout()

        self.btn_compare = QPushButton("对比四种方法 (Otsu/区域生长/K-means/DL)")
        self.btn_compare.setStyleSheet("background-color: #1a4a6e; color: white;")
        layout.addWidget(self.btn_compare)

        self.btn_morph_clean = QPushButton("形态学后处理")
        layout.addWidget(self.btn_morph_clean)

        layout.addWidget(QLabel("评估结果:"))
        self.text_report = QTextEdit()
        self.text_report.setReadOnly(True)
        self.text_report.setMaximumHeight(200)
        self.text_report.setPlaceholderText("点击「对比四种方法」生成评估报告...")
        layout.addWidget(self.text_report)

        layout.addStretch()
        self.setLayout(layout)


class DisplayControls(QGroupBox):
    """显示模式控制"""
    mode_changed = pyqtSignal(str)  # display mode

    def __init__(self):
        super().__init__("🖼️ 显示模式")
        layout = QVBoxLayout()

        self.cmb_display = QComboBox()
        self.cmb_display.addItems(["原图", "差值图", "伪彩色融合", "轮廓叠加", "窗宽窗位"])
        layout.addWidget(self.cmb_display)

        self.btn_apply_mode = QPushButton("应用显示模式")
        layout.addWidget(self.btn_apply_mode)

        self.mpr_label = QLabel("MPR 视图:")
        layout.addWidget(self.mpr_label)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("切片位置(%):"))
        self.slider_slice = QSlider(Qt.Horizontal)
        self.slider_slice.setRange(0, 100)
        self.slider_slice.setValue(50)
        hl.addWidget(self.slider_slice)
        layout.addLayout(hl)

        self.setLayout(layout)
