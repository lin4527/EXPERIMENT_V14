"""
主窗口 — 左侧功能面板 + 右侧双画布 + 底部参数/指标区
集成所有6个模块
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QTabWidget, QScrollArea, QStatusBar,
                             QMessageBox, QLabel, QTextEdit, QTableWidget,
                             QTableWidgetItem, QPushButton, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

from config import (CT_PATH, SEG_PATH, GUI_DEFAULT_WIDTH, GUI_DEFAULT_HEIGHT,
                    WINDOW_PRESETS, OUTPUT_DIR)
from module1_io_visualization import NiftiReader, MPRReconstructor, apply_window_level, apply_preset
from module2_preprocessing import *
from module3_traditional_seg import *
from module5_registration_eval import *
from module6_gui.canvas_widgets import ImageCanvas, MPRViewer, DualCanvas
from module6_gui.control_panels import (FilePanel, PreprocessPanel, SegmentationPanel,
                                        DeepLearningPanel, EvalPanel, DisplayControls)
from module6_gui.worker_threads import (ThresholdWorker, RegionGrowWorker,
                                        KMeansWorker, TrainWorker, EvalWorker, BaseWorker)


class MainWindow(QMainWindow):
    """医学图像处理综合实验系统 — 主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("医学图像处理综合实验 14 — 脊椎椎体逐节分割系统")
        self.setMinimumSize(GUI_DEFAULT_WIDTH, GUI_DEFAULT_HEIGHT)

        # --- 数据状态 ---
        self.ct_reader = None
        self.seg_reader = None
        self.mpr = None
        self.current_slice = None      # 当前显示的2D切片
        self.current_result = None     # 当前分割/处理结果
        self.gt_slice = None           # 金标准切片
        self.seed_point = None         # 区域生长种子点
        self.dl_model = None           # 深度学习模型
        self.current_slice_idx = 0.5   # 切片位置比例

        # --- 初始化UI ---
        self._init_ui()
        self._connect_signals()

        # 自动加载数据
        QTimer.singleShot(500, self._auto_load_data)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # === 左侧面板 (可滚动，确保小屏幕上所有面板都可见) ===
        left_scroll = QScrollArea()
        left_scroll.setMinimumWidth(280)
        left_scroll.setMaximumWidth(340)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(4)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # 文件面板
        self.file_panel = FilePanel()
        left_layout.addWidget(self.file_panel)

        # 预处理面板
        self.preprocess_panel = PreprocessPanel()
        left_layout.addWidget(self.preprocess_panel)

        # 分割面板
        self.seg_panel = SegmentationPanel()
        left_layout.addWidget(self.seg_panel)

        # 深度学习面板
        self.dl_panel = DeepLearningPanel()
        left_layout.addWidget(self.dl_panel)

        # 评估面板
        self.eval_panel = EvalPanel()
        left_layout.addWidget(self.eval_panel)

        # 显示控制面板
        self.display_panel = DisplayControls()
        left_layout.addWidget(self.display_panel)

        # ⚠️ 不加 stretch，让内容自然排列，超出时自动出现滚动条
        left_scroll.setWidget(left_widget)

        # === 右侧显示区 ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # MPR 三视图
        mpr_group = QGroupBox("MPR 多平面重建 (轴状面 / 冠状面 / 矢状面)")
        mpr_layout = QVBoxLayout()
        self.mpr_viewer = MPRViewer()
        mpr_layout.addWidget(self.mpr_viewer)
        mpr_group.setLayout(mpr_layout)
        right_layout.addWidget(mpr_group)

        # 原图+结果双画布
        dual_group = QGroupBox("处理结果对照 (左:原图 | 右:结果)")
        dual_layout = QVBoxLayout()
        self.dual_canvas = DualCanvas("原图 (Original)", "结果 (Result)")
        dual_layout.addWidget(self.dual_canvas)
        dual_group.setLayout(dual_layout)
        right_layout.addWidget(dual_group)

        # === 底部状态栏 ===
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_info = QLabel("就绪 | 等待加载数据...")
        self.status_bar.addWidget(self.lbl_info)

        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        main_layout.addWidget(splitter)

    def _connect_signals(self):
        """连接所有信号和槽"""
        # 文件面板
        self.file_panel.load_ct.connect(self._load_ct)
        self.file_panel.load_seg.connect(self._load_seg)
        self.file_panel.save_result.connect(self._save_result)

        # 预处理面板
        self.preprocess_panel.btn_apply_window.clicked.connect(self._apply_window)
        self.preprocess_panel.btn_apply_gray.clicked.connect(self._apply_gray_transform)
        self.preprocess_panel.btn_hist_eq.clicked.connect(self._apply_hist_eq)
        self.preprocess_panel.btn_apply_filter.clicked.connect(self._apply_freq_filter)
        self.preprocess_panel.btn_resample.clicked.connect(self._apply_resample)
        self.preprocess_panel.btn_quantize.clicked.connect(self._apply_quantize)

        # 分割面板
        self.seg_panel.btn_edge.clicked.connect(self._run_edge)
        self.seg_panel.btn_otsu.clicked.connect(self._run_otsu)
        self.seg_panel.btn_iterative.clicked.connect(self._run_iterative)
        self.seg_panel.btn_region_grow.clicked.connect(self._run_region_grow)
        self.seg_panel.btn_kmeans.clicked.connect(self._run_kmeans)
        self.seg_panel.btn_hough.clicked.connect(self._run_hough)
        self.seg_panel.btn_texture.clicked.connect(self._run_texture)

        # 深度学习面板
        self.dl_panel.btn_train.clicked.connect(self._train_unet)
        self.dl_panel.btn_load_model.clicked.connect(self._load_dl_model)
        self.dl_panel.btn_predict.clicked.connect(self._predict_dl)
        self.dl_panel.btn_postprocess.clicked.connect(self._dl_postprocess)

        # 评估面板
        self.eval_panel.btn_compare.clicked.connect(self._compare_methods)
        self.eval_panel.btn_morph_clean.clicked.connect(self._morph_clean)

        # 显示控制
        self.display_panel.btn_apply_mode.clicked.connect(self._change_display_mode)
        self.display_panel.slider_slice.valueChanged.connect(self._on_slice_slider)

        # 画布鼠标点击 → 选取种子点
        self.dual_canvas.canvas1.mouse_clicked.connect(self._on_seed_clicked)

    # ===================== 数据加载 =====================

    def _auto_load_data(self):
        """自动加载默认数据"""
        if os.path.exists(CT_PATH):
            self._load_ct(CT_PATH)
        if os.path.exists(SEG_PATH):
            self._load_seg(SEG_PATH)

    def _load_ct(self, path):
        try:
            self.ct_reader = NiftiReader(path)
            self.mpr = MPRReconstructor(self.ct_reader.data)

            # 🔧 自动计算最优窗宽窗位
            self._auto_window_level()

            self._update_all_views()
            self._update_current_slice()
            self.file_panel.lbl_ct_path.setText(f"CT: {os.path.basename(path)}")
            meta = self.ct_reader.get_metadata()
            info = f"CT加载成功: {self.ct_reader.shape} | HU[{meta['value_range'][0]:.0f},{meta['value_range'][1]:.0f}] | {path}"
            self.status_bar.showMessage(info)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载CT失败: {e}")

    def _auto_window_level(self):
        """根据CT体数据自动计算并设置最优窗宽窗位"""
        if self.mpr is None:
            return
        from module1_io_visualization.window_level import auto_window, WINDOW_PRESETS
        # 取中间轴状面切片做统计
        center_slice = self.mpr.axial()
        self._auto_ww, self._auto_wl = auto_window(center_slice)
        # 把自动计算的结果加入预设列表并选中
        WINDOW_PRESETS["自动窗"] = (self._auto_ww, self._auto_wl)
        idx = self.preprocess_panel.cmb_window.findText("自动窗")
        if idx < 0:
            self.preprocess_panel.cmb_window.addItem("自动窗")
        self.preprocess_panel.cmb_window.setCurrentText("自动窗")
        print(f"[自动窗宽窗位] WW={self._auto_ww}, WL={self._auto_wl}  (基于CT值1%-99%分位数)")

    def _load_seg(self, path):
        try:
            self.seg_reader = NiftiReader(path)
            self.file_panel.lbl_seg_path.setText(f"标注: {os.path.basename(path)}")
            self.status_bar.showMessage(f"标注加载成功: {self.seg_reader.shape}")
            self._update_gt_slice()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载标注失败: {e}")

    def _save_result(self, path):
        if self.current_result is not None:
            try:
                from module1_io_visualization.image_io import write_image
                write_image(path, self.current_result)
                self.status_bar.showMessage(f"已保存: {path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    # ===================== 视图更新 =====================

    def _get_slice_index(self, axis=0):
        """根据滑块值和轴计算当前切片索引"""
        ratio = self.display_panel.slider_slice.value() / 100.0
        if self.mpr:
            idx = int(ratio * (self.mpr.shape[axis] - 1))
            return max(0, min(idx, self.mpr.shape[axis] - 1))
        return 0

    def _update_all_views(self):
        if self.mpr is None:
            return
        # 每个切面用自己的轴索引
        idx_ax = self._get_slice_index(0)   # 轴状面沿 Z(axis0)
        idx_co = self._get_slice_index(1)   # 冠状面沿 Y(axis1)
        idx_sa = self._get_slice_index(2)   # 矢状面沿 X(axis2)
        axial = self.mpr.axial(idx_ax)
        coronal = self.mpr.coronal(idx_co)
        sagittal = self.mpr.sagittal(idx_sa)

        # 应用窗宽窗位
        ww_name = self.preprocess_panel.cmb_window.currentText()
        axial_win = apply_preset(axial, ww_name)
        coronal_win = apply_preset(coronal, ww_name)
        sagittal_win = apply_preset(sagittal, ww_name)

        self.mpr_viewer.update_views(axial_win, coronal_win, sagittal_win)

    def _update_current_slice(self):
        """更新当前切片显示 — 默认显示矢状面(完整脊柱纵向视图)"""
        if self.mpr is None:
            return
        idx = self._get_slice_index(2)  # 矢状面
        ww_name = self.preprocess_panel.cmb_window.currentText()
        self.current_slice = apply_preset(self.mpr.sagittal(idx), ww_name)
        self.dual_canvas.set_original(self.current_slice)

        # 清空结果
        self.current_result = None

    def _update_gt_slice(self):
        """更新金标准切片"""
        if self.seg_reader is None or self.mpr is None:
            return
        idx = self._get_slice_index()
        # 标注数据的切片方向可能与CT不同，尝试匹配
        try:
            self.gt_slice = self.seg_reader.get_center_slice('axial')
        except:
            self.gt_slice = None

    def _on_slice_slider(self):
        self._update_all_views()
        self._update_current_slice()
        self._update_gt_slice()
        self.lbl_info.setText(f"切片位置: {self.display_panel.slider_slice.value()}%")

    # ===================== 预处理 =====================

    def _apply_window(self):
        self._update_all_views()
        self._update_current_slice()

    def _apply_gray_transform(self):
        if self.current_slice is None:
            return
        t_type = self.preprocess_panel.cmb_gray.currentText()
        gamma = self.preprocess_panel.spin_gamma.value()
        img = self.current_slice.astype(np.float64)

        if t_type == "线性变换":
            result = my_linear_transform(img, gamma, 0)
        elif t_type == "对数变换":
            result = my_log_transform(img, gamma)
        elif t_type == "Gamma变换":
            result = my_gamma_transform(img, gamma)
        elif t_type == "对比度拉伸":
            result = my_contrast_stretch(img)
        else:
            result = my_piecewise_linear_transform(img, [(80, 0), (180, 255)])

        self.current_result = np.clip(result, 0, 255).astype(np.uint8)
        self.dual_canvas.set_result(self.current_result)
        self.status_bar.showMessage(f"灰度变换完成: {t_type}")

    def _apply_hist_eq(self):
        if self.current_slice is None:
            return
        self.current_result = my_histogram_equalization(self.current_slice)
        self.dual_canvas.set_result(self.current_result)
        self.status_bar.showMessage("直方图均衡化完成")

    def _apply_freq_filter(self):
        if self.current_slice is None:
            return
        # 使用小图像加速
        from module2_preprocessing.my_interpolation import my_image_resample
        small = my_image_resample(self.current_slice, 0.5, 'bilinear')

        f_type = self.preprocess_panel.cmb_filter.currentText()
        cutoff = self.preprocess_panel.spin_cutoff.value()

        filter_map = {
            "高斯低通": ('gaussian', 'lowpass'),
            "高斯高通": ('gaussian', 'highpass'),
            "理想低通": ('ideal', 'lowpass'),
            "巴特沃斯低通": ('butterworth', 'lowpass'),
        }
        ftype, mode = filter_map.get(f_type, ('gaussian', 'lowpass'))

        if mode == 'lowpass':
            result = my_lowpass_filter(small, cutoff, ftype)
        else:
            result = my_highpass_filter(small, cutoff, ftype)

        # 恢复到原图大小
        self.current_result = my_bilinear_interpolation(result, self.current_slice.shape[:2])
        self.dual_canvas.set_result(self.current_result)
        self.status_bar.showMessage(f"频域滤波完成: {f_type}")

    def _apply_resample(self):
        if self.current_slice is None:
            return
        method = 'nearest' if self.preprocess_panel.cmb_interp.currentText() == "最近邻" else 'bilinear'
        scale = self.preprocess_panel.spin_scale.value()

        if method == 'nearest':
            result = my_nearest_neighbor(self.current_slice,
                                         (int(self.current_slice.shape[0]*scale),
                                          int(self.current_slice.shape[1]*scale)))
        else:
            result = my_bilinear_interpolation(self.current_slice,
                                               (int(self.current_slice.shape[0]*scale),
                                                int(self.current_slice.shape[1]*scale)))

        self.current_result = result
        self.dual_canvas.set_result(result)
        self.status_bar.showMessage(f"重采样完成: {method} x{scale:.1f}")

    def _apply_quantize(self):
        if self.current_slice is None:
            return
        levels = self.preprocess_panel.spin_levels.value()
        self.current_result = my_grayscale_quantize(self.current_slice, levels)
        self.dual_canvas.set_result(self.current_result)
        self.status_bar.showMessage(f"灰度量化完成: {levels}级")

    # ===================== 传统分割 =====================

    def _run_edge(self):
        if self.current_slice is None:
            return
        method = self.seg_panel.cmb_edge.currentText()
        edge_map = {
            "Sobel": my_sobel,
            "Canny": lambda x: my_canny(x, 40, 100),
            "Roberts": my_roberts,
            "Prewitt": my_prewitt,
            "Laplacian": my_laplacian,
            "LoG": lambda x: my_log_edge(x, 2.0),
        }
        func = edge_map.get(method, my_sobel)
        self.current_result = np.clip(func(self.current_slice.astype(np.float64)), 0, 255).astype(np.uint8)
        self.dual_canvas.set_result(self.current_result)
        self.status_bar.showMessage(f"边缘检测完成: {method}")

    def _run_otsu(self):
        if self.current_slice is None:
            return
        self.worker = ThresholdWorker(my_otsu, self.current_slice)
        self.worker.finished.connect(lambda b, t: (
            setattr(self, 'current_result', b),
            self.dual_canvas.set_result(b),
            self.status_bar.showMessage(f"Otsu完成: 阈值={t}")
        ))
        self.worker.error.connect(lambda e: QMessageBox.warning(self, "错误", str(e)))
        self.worker.start()

    def _run_iterative(self):
        if self.current_slice is None:
            return
        self.worker = ThresholdWorker(my_iterative_threshold, self.current_slice)
        self.worker.finished.connect(lambda b, t: (
            setattr(self, 'current_result', b),
            self.dual_canvas.set_result(b),
            self.status_bar.showMessage(f"迭代阈值完成: T={t:.1f}")
        ))
        self.worker.error.connect(lambda e: QMessageBox.warning(self, "错误", str(e)))
        self.worker.start()

    def _on_seed_clicked(self, row, col):
        self.seed_point = (row, col)
        self.seg_panel.lbl_seed.setText(f"种子点: ({row}, {col})")

    def _run_region_grow(self):
        if self.current_slice is None or self.seed_point is None:
            QMessageBox.information(self, "提示", "请先在左侧原图上点击选取种子点")
            return
        threshold = self.seg_panel.spin_rg_thresh.value()
        self.worker = RegionGrowWorker(my_region_grow, self.current_slice.astype(np.float64),
                                       self.seed_point, threshold=threshold)
        self.worker.finished.connect(lambda mask: (
            setattr(self, 'current_result', (mask.astype(np.uint8) * 255)),
            self.dual_canvas.set_result(mask.astype(np.uint8) * 255),
            self.status_bar.showMessage(f"区域生长完成: 面积={mask.sum()}px")
        ))
        self.worker.error.connect(lambda e: QMessageBox.warning(self, "错误", str(e)))
        self.worker.start()

    def _run_kmeans(self):
        if self.current_slice is None:
            return
        k = self.seg_panel.spin_k.value()
        use_spatial = self.seg_panel.chk_spatial.isChecked()

        if use_spatial:
            def do_spatial_kmeans(img):
                labels = my_kmeans_spatial(img, k=k, spatial_weight=0.3)
                # 可视化：不同类用不同灰度
                viz = (labels.astype(np.float64) / (k - 1) * 255).astype(np.uint8) if k > 1 else labels.astype(np.uint8)
                return viz, None
            self.worker = BaseWorker(do_spatial_kmeans, self.current_slice)
        else:
            self.worker = KMeansWorker(my_kmeans_gray, self.current_slice, k=k)

        self.worker.finished.connect(self._on_kmeans_done)
        self.worker.error.connect(lambda e: QMessageBox.warning(self, "错误", str(e)))
        self.worker.start()

    def _on_kmeans_done(self, result, centers=None):
        if centers is not None:
            labels = result
            k = len(centers)
            viz = (labels.astype(np.float64) / max(k - 1, 1) * 255).astype(np.uint8)
        else:
            viz = result
        self.current_result = viz
        self.dual_canvas.set_result(viz)
        self.status_bar.showMessage(f"K-means完成: {len(np.unique(viz))}类")

    def _run_hough(self):
        if self.current_slice is None:
            return
        from module3_traditional_seg.my_hough import my_hough_lines, draw_hough_lines
        from module3_traditional_seg.my_edge_detection import my_canny
        edges = my_canny(self.current_slice, 40, 100)
        lines = my_hough_lines(edges, threshold=60)
        result = draw_hough_lines(self.current_slice, lines)
        self.current_result = result
        self.dual_canvas.set_result(result)
        self.status_bar.showMessage(f"霍夫直线检测: {len(lines)}条")

    def _run_texture(self):
        if self.current_slice is None:
            return
        t_type = self.seg_panel.cmb_texture.currentText()
        small = self.current_slice[::2, ::2].astype(np.float64)

        if t_type == "GLCM":
            from module3_traditional_seg.my_texture import my_glcm, my_glcm_features
            glcm = my_glcm(small, angle=0)
            feat = my_glcm_features(glcm)
            # 显示 GLCM (对数增强)
            glcm_vis = np.log1p(glcm * 1000)
            self.current_result = glcm_vis
            msg = f"GLCM: contrast={feat['contrast']:.2f}, energy={feat['energy']:.4f}"
        elif t_type == "LBP":
            from module3_traditional_seg.my_texture import my_lbp
            lbp = my_lbp(small)
            self.current_result = lbp
            msg = f"LBP完成: unique patterns={len(np.unique(lbp))}"
        else:  # Gabor
            from module3_traditional_seg.my_texture import my_gabor_filter
            resp = my_gabor_filter(small, theta=0, frequency=0.1)
            self.current_result = resp
            msg = f"Gabor完成: range=[{resp.min():.2f}, {resp.max():.2f}]"

        self.dual_canvas.set_result(self.current_result)
        self.status_bar.showMessage(msg)

    # ===================== 深度学习 =====================

    def _train_unet(self):
        if self.ct_reader is None or self.seg_reader is None:
            QMessageBox.warning(self, "警告", "请先加载CT和标注数据")
            return

        from module4_deep_learning.unet_model import UNet
        from module4_deep_learning.dataset import VertebraDataset, VertebraTransform
        from module4_deep_learning.train import train_unet

        epochs = self.dl_panel.spin_epochs.value()
        batch = self.dl_panel.spin_batch.value()
        lr = self.dl_panel.spin_lr.value()

        # 过滤有标注的切片（跳过纯背景切片，加速训练）
        good_indices = []
        for i in range(self.seg_reader.shape[0]):
            if (self.seg_reader.data[i] > 0).sum() > 500:
                good_indices.append(i)
        if len(good_indices) < 10:
            good_indices = list(range(self.seg_reader.shape[0]))
        np.random.seed(42)
        np.random.shuffle(good_indices)
        split = int(len(good_indices) * 0.8)

        # 获取类别数
        temp_ds = VertebraDataset(self.ct_reader.data, self.seg_reader.data,
                                  slice_axis=0, image_size=256,
                                  transform=VertebraTransform(),
                                  slices_indices=good_indices)
        num_classes = temp_ds.num_classes

        self.dl_panel.progress_label.setText(f"{len(good_indices)}切片 x{num_classes}类 GPU...")
        save_path = os.path.join(OUTPUT_DIR, "unet_vertebra.pth")

        def train_wrapper():
            _train_ds = VertebraDataset(self.ct_reader.data, self.seg_reader.data,
                                        slice_axis=0, image_size=256,
                                        transform=VertebraTransform(),
                                        slices_indices=good_indices[:split])
            _val_ds = VertebraDataset(self.ct_reader.data, self.seg_reader.data,
                                      slice_axis=0, image_size=256,
                                      slices_indices=good_indices[split:])
            _model = UNet(in_channels=1, num_classes=num_classes, base_filters=32)
            self.dl_model = _model
            return train_unet(_model, _train_ds, _val_ds,
                              epochs=epochs, batch_size=batch, lr=lr,
                              save_path=save_path)

        self.worker = BaseWorker(train_wrapper)
        self.worker.finished.connect(lambda h: (
            self.dl_panel.progress_label.setText(f"训练完成! ValDice: {h['val_dice'][-1]:.3f}"),
            self.status_bar.showMessage("U-Net训练完成!")
        ))
        self.worker.error.connect(lambda e: (
            QMessageBox.warning(self, "训练错误", str(e)),
            self.dl_panel.progress_label.setText("训练失败")
        ))
        self.worker.start()

    def _load_dl_model(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "加载模型", "", "PyTorch Model (*.pth)")
        if path:
            try:
                from module4_deep_learning.unet_model import UNet
                import torch
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                # 从checkpoint推断num_classes
                out_w = ckpt['model_state_dict']['outc.weight']
                num_classes = out_w.shape[0]
                self.dl_model = UNet(in_channels=1, num_classes=num_classes, base_filters=32)
                self.dl_model.load_state_dict(ckpt['model_state_dict'])
                self.dl_panel.progress_label.setText(f"模型: {os.path.basename(path)} ({num_classes}类)")
                self.status_bar.showMessage(f"模型加载成功: {path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型失败: {e}")

    def _predict_dl(self):
        if self.dl_model is None:
            QMessageBox.warning(self, "警告", "请先训练或加载模型")
            return
        if self.current_slice is None:
            return

        # 获取模型所在设备
        device = next(self.dl_model.parameters()).device
        from module4_deep_learning.inference import segment_slice
        labels = segment_slice(self.dl_model, self.current_slice,
                               image_size=256, device=str(device))
        # 反向映射标签: 模型输出(0-9) → 原始标签(0,16-24)
        viz = (labels.astype(np.float64) / max(labels.max(), 1) * 255).astype(np.uint8)
        self.current_result = viz
        self.dual_canvas.set_result(viz)
        n_regions = len(np.unique(labels))
        self.status_bar.showMessage(f"DL推理完成: {n_regions}个区域 (GPU: {device}")

    def _dl_postprocess(self):
        if self.current_result is None:
            return
        from module4_deep_learning.postprocess import postprocess_vertebrae
        cleaned = postprocess_vertebrae(self.current_result, min_area=50)
        self.current_result = cleaned
        self.dual_canvas.set_result(cleaned)
        self.status_bar.showMessage("后处理完成")

    # ===================== 评估 =====================

    def _compare_methods(self):
        if self.current_slice is None:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return

        # 用当前切片生成各方法的预测
        img = self.current_slice.astype(np.float64)
        bin_otsu, _ = my_otsu(img)
        bin_iter, _ = my_iterative_threshold(img)

        # 区域生长 (中心种子)
        center = (img.shape[0] // 2, img.shape[1] // 2)
        rg_mask = my_region_grow(img, center, threshold=20)

        # K-means
        km_labels, km_centers = my_kmeans_gray(img, k=3)
        bone_cluster = np.argmax(km_centers)
        bone_mask = ((km_labels == bone_cluster).astype(np.uint8) * 255)

        predictions = {
            "Otsu": bin_otsu,
            "迭代阈值": bin_iter,
            "区域生长": (rg_mask.astype(np.uint8) * 255),
            "K-means": bone_mask,
        }

        # 使用标注作为金标准 (如果有)
        if self.gt_slice is not None:
            gt = self.gt_slice
        else:
            # 无标注时用Otsu作为参考
            gt = bin_otsu

        evaluator = SegmentationEvaluator(gt)
        report, table_data = evaluator.generate_report(predictions)

        self.eval_panel.text_report.setPlainText(report)
        self.status_bar.showMessage("评估完成")

    def _morph_clean(self):
        if self.current_result is None:
            return
        cleaned = my_opening(my_closing(self.current_result > 0, 5), 3)
        self.current_result = (cleaned.astype(np.uint8) * 255)
        self.dual_canvas.set_result(self.current_result)
        self.status_bar.showMessage("形态学后处理完成 (开运算+闭运算)")

    # ===================== 显示模式 =====================

    def _change_display_mode(self):
        mode = self.display_panel.cmb_display.currentText()
        if self.current_slice is None:
            return

        if mode == "原图":
            self.dual_canvas.set_original(self.current_slice)
        elif mode == "差值图" and self.current_result is not None:
            diff = np.abs(self.current_slice.astype(float) - self.current_result.astype(float))
            self.dual_canvas.set_result(diff, colormap='hot')
        elif mode == "伪彩色融合":
            from module2_preprocessing.my_color_space import my_pseudocolor_map
            pseudo = my_pseudocolor_map(self.current_slice, 'jet')
            self.dual_canvas.set_original(pseudo)
        elif mode == "轮廓叠加" and self.current_result is not None:
            # 在结果上叠加轮廓
            import cv2
            edges = cv2.Canny(self.current_result.astype(np.uint8), 50, 150)
            overlay = cv2.cvtColor(self.current_slice, cv2.COLOR_GRAY2BGR)
            overlay[edges > 0] = [0, 255, 0]
            self.dual_canvas.set_result(overlay)

        self.status_bar.showMessage(f"显示模式: {mode}")
