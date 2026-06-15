"""
原始数据展示工具 — 展示 CT 和标注数据最原始的样子
"""
import sys, os, numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QGroupBox)
from PyQt5.QtCore import Qt
import nibabel as nib
from config import CT_PATH, SEG_PATH


class RawDataViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("原始数据展示 — CT 体数据 + 椎体标注")
        self.setMinimumSize(1100, 750)

        # 加载数据
        self.ct = nib.load(CT_PATH).get_fdata()
        self.seg = nib.load(SEG_PATH).get_fdata()

        # 当前切片索引
        self.idx_axial = self.ct.shape[0] // 2
        self.idx_coronal = self.ct.shape[1] // 2
        self.idx_sagittal = self.ct.shape[2] // 2

        self._init_ui()
        self._update_all()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # === 数据信息 ===
        info = QLabel(
            f"CT 体数据: {self.ct.shape}  |  HU 范围: [{self.ct.min():.0f}, {self.ct.max():.0f}]  |  "
            f"标注: {self.seg.shape}  |  椎体标签: {np.unique(self.seg)[1:].astype(int).tolist()}  |  "
            f"共 {len(np.unique(self.seg))-1} 节椎体"
        )
        info.setStyleSheet("font-size: 13px; padding: 8px; background: #2a2a2a; color: #ddd;")
        layout.addWidget(info)

        # === Matplotlib 画布 ===
        self.fig = Figure(figsize=(14, 10))
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)

        # === 底部三根滑块 ===
        slider_group = QGroupBox("切片位置 (拖动切换)")
        slider_layout = QVBoxLayout()

        for axis_name, axis_range, attr, update_fn in [
            ("轴状面 Axial (沿Z轴)", self.ct.shape[0], "idx_axial", self._update_all),
            ("冠状面 Coronal (沿Y轴)", self.ct.shape[1], "idx_coronal", self._update_all),
            ("矢状面 Sagittal (沿X轴)", self.ct.shape[2], "idx_sagittal", self._update_all),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(f"{axis_name} [{axis_range} slices]")
            lbl.setMinimumWidth(280)
            row.addWidget(lbl)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, axis_range - 1)
            slider.setValue(getattr(self, attr))
            slider.valueChanged.connect(lambda v, a=attr: setattr(self, a, v) or self._update_all())
            row.addWidget(slider)
            val_label = QLabel(str(getattr(self, attr)))
            val_label.setMinimumWidth(40)
            slider.valueChanged.connect(lambda v, l=val_label: l.setText(str(v)))
            row.addWidget(val_label)
            slider_layout.addLayout(row)

        slider_group.setLayout(slider_layout)
        layout.addWidget(slider_group)

    def _update_all(self):
        self.fig.clear()

        # 三行两列: 左=CT原始HU, 右=标注叠加
        axes = self.fig.subplots(3, 2)

        slices_ct = [
            ("Axial 轴状面", self.ct[self.idx_axial, :, :]),
            ("Coronal 冠状面", self.ct[:, self.idx_coronal, :]),
            ("Sagittal 矢状面", self.ct[:, :, self.idx_sagittal]),
        ]
        slices_seg = [
            self.seg[self.idx_axial, :, :],
            self.seg[:, self.idx_coronal, :],
            self.seg[:, :, self.idx_sagittal],
        ]

        for row, ((title, ct_slice), seg_slice) in enumerate(zip(slices_ct, slices_seg)):
            # 左列：CT 原始 HU 值
            ax_ct = axes[row, 0]
            im_ct = ax_ct.imshow(ct_slice.T, cmap="gray", origin="lower")
            ax_ct.set_title(f"{title} — 原始 CT (HU)", fontsize=11)
            self.fig.colorbar(im_ct, ax=ax_ct, fraction=0.046, pad=0.04)
            ax_ct.axis("off")

            # 右列：标注轮廓叠加在 CT 上
            ax_seg = axes[row, 1]
            ax_seg.imshow(ct_slice.T, cmap="gray", origin="lower")

            seg_labels = np.unique(seg_slice)
            seg_labels = seg_labels[seg_labels > 0]
            import cv2
            for label in seg_labels:
                mask = (seg_slice == label).T.astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    ax_seg.plot(cnt[:, 0, 0], cnt[:, 0, 1],
                                "r-", linewidth=0.8)

            ax_seg.set_title(
                f"{title} — 标注叠加 ({len(seg_labels)} vertebrae)",
                fontsize=11
            )
            ax_seg.axis("off")

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = RawDataViewer()
    viewer.show()
    print("原始数据展示已启动!")
    print(f"  CT: {viewer.ct.shape}, HU=[{viewer.ct.min():.0f}, {viewer.ct.max():.0f}]")
    print(f"  标注: {viewer.seg.shape}, 椎体: {np.unique(viewer.seg)[1:].astype(int).tolist()}")
    print(f"  拖动底部三根滑块切换切面位置")
    sys.exit(app.exec_())
