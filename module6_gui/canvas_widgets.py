"""
画布显示组件 — MPR三视图 / 原图+结果双画布
"""
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QScrollArea, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, pyqtSignal


def numpy_to_qpixmap(image, colormap='gray'):
    """将 numpy array 转换为 QPixmap 用于显示"""
    img = np.asarray(image, dtype=np.float64)

    if img.ndim == 3 and img.shape[2] == 3:
        # RGB 图像
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-10) * 255, 0, 255).astype(np.uint8)
        h, w, c = img.shape
        qimg = QImage(img.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    else:
        # 灰度图像
        if img.max() > 1 and img.dtype != np.uint8:
            img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-10) * 255, 0, 255).astype(np.uint8)
        elif img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        h, w = img.shape
        qimg = QImage(img.tobytes(), w, h, w, QImage.Format_Grayscale8)

    return QPixmap.fromImage(qimg)


class ImageCanvas(QLabel):
    """
    可交互的图像显示画布
    支持鼠标点击获取种子点 (用于区域生长)
    """
    mouse_clicked = pyqtSignal(int, int)  # (row, col)
    mouse_moved = pyqtSignal(int, int)

    def __init__(self, title="", fixed_size=400):
        super().__init__()
        self.setMinimumSize(fixed_size, fixed_size)
        self.setMaximumSize(fixed_size + 100, fixed_size + 100)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #555; background-color: #1a1a1a;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.title = title
        self.original_pixmap = None
        self.current_pixmap = None
        self.seed_points = []
        self._scale = 1.0
        self._image_shape = None

    def set_image(self, image, colormap='gray'):
        """设置显示图像"""
        self.current_pixmap = numpy_to_qpixmap(image, colormap)
        self.original_pixmap = self.current_pixmap
        if hasattr(image, 'shape'):
            self._image_shape = image.shape[:2]
        self._fit_to_canvas()
        self.update()

    def _fit_to_canvas(self):
        """缩放图像以适应画布"""
        if self.original_pixmap is None:
            return
        cw = self.width() - 4
        ch = self.height() - 4
        pw = self.original_pixmap.width()
        ph = self.original_pixmap.height()
        if pw > 0 and ph > 0:
            scale = min(cw / pw, ch / ph)
            self._scale = scale
            self.current_pixmap = self.original_pixmap.scaled(
                int(pw * scale), int(ph * scale),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        self.setPixmap(self.current_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_to_canvas()

    def mousePressEvent(self, event):
        """鼠标点击 → 发送图像坐标"""
        if self._image_shape is None:
            return
        # 将 widget 坐标转换为图像坐标
        x = event.pos().x()
        y = event.pos().y()

        # 考虑图像缩放和居中偏移
        pix_rect = self.pixmap().rect() if self.pixmap() else None
        if pix_rect is None:
            return
        offset_x = (self.width() - pix_rect.width()) // 2
        offset_y = (self.height() - pix_rect.height()) // 2

        img_x = x - offset_x
        img_y = y - offset_y

        if self._scale > 0:
            row = int(img_y / self._scale)
            col = int(img_x / self._scale)
        else:
            row = img_y
            col = img_x

        # 边界检查
        H, W = self._image_shape
        row = max(0, min(row, H - 1))
        col = max(0, min(col, W - 1))

        self.seed_points.append((row, col))
        self.mouse_clicked.emit(row, col)

        # 重绘显示种子点
        self._draw_seed_marker(row, col)

    def _draw_seed_marker(self, row, col):
        """在图像上绘制种子点标记"""
        if self.current_pixmap is None:
            return
        pix = self.current_pixmap.copy()
        painter = QPainter(pix)
        painter.setPen(QPen(QColor(255, 0, 0), 3))
        painter.setBrush(QColor(255, 0, 0, 100))
        x = int(col * self._scale)
        y = int(row * self._scale)
        painter.drawEllipse(QPoint(x, y), 5, 5)
        painter.end()
        self.setPixmap(pix)

    def clear_seeds(self):
        self.seed_points = []
        self._fit_to_canvas()


class MPRViewer(QWidget):
    """MPR 三平面视图组件"""

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()

        self.axial_view = ImageCanvas("轴状面 (Axial)", 280)
        self.coronal_view = ImageCanvas("冠状面 (Coronal)", 280)
        self.sagittal_view = ImageCanvas("矢状面 (Sagittal)", 280)

        layout.addWidget(self.axial_view)
        layout.addWidget(self.coronal_view)
        layout.addWidget(self.sagittal_view)

        self.setLayout(layout)

    def update_views(self, axial, coronal, sagittal, colormap='gray'):
        self.axial_view.set_image(axial, colormap)
        self.coronal_view.set_image(coronal, colormap)
        self.sagittal_view.set_image(sagittal, colormap)

    def clear(self):
        self.axial_view.clear()
        self.coronal_view.clear()
        self.sagittal_view.clear()


class DualCanvas(QWidget):
    """双画布组件: 原图 + 结果对照"""

    def __init__(self, title1="原图", title2="结果"):
        super().__init__()
        layout = QHBoxLayout()

        self.canvas1 = ImageCanvas(title1, 380)
        self.canvas2 = ImageCanvas(title2, 380)

        layout.addWidget(self.canvas1)
        layout.addWidget(self.canvas2)
        self.setLayout(layout)

    def set_original(self, image, colormap='gray'):
        self.canvas1.set_image(image, colormap)

    def set_result(self, image, colormap='gray'):
        self.canvas2.set_image(image, colormap)
