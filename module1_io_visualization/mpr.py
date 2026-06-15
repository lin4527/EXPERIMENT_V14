"""
MPR 多平面重建 — 实验12核心
支持轴状面(Axial)、冠状面(Coronal)、矢状面(Sagittal)正交切片提取
"""
import numpy as np
from config import MPR_DEFAULT_RATIO


class MPRReconstructor:
    """MPR 多平面重建器"""

    def __init__(self, volume_data):
        """
        :param volume_data: 3D numpy array (sagittal, coronal, axial)
                            即 shape 为 (Z, Y, X)
        """
        self.volume = np.asarray(volume_data)
        self.shape = self.volume.shape

    def axial(self, index=None):
        """
        轴状面 (横断面) — 沿 Z 轴切片
        :param index: 切片索引, 默认中间层
        :return: 2D numpy array (Y, X)
        """
        if index is None:
            index = int(self.shape[0] * MPR_DEFAULT_RATIO)
        index = np.clip(index, 0, self.shape[0] - 1)
        return self.volume[index, :, :]

    def coronal(self, index=None):
        """
        冠状面 — 沿 Y 轴切片
        :param index: 切片索引, 默认中间层
        :return: 2D numpy array (Z, X)
        """
        if index is None:
            index = int(self.shape[1] * MPR_DEFAULT_RATIO)
        index = np.clip(index, 0, self.shape[1] - 1)
        return self.volume[:, index, :]

    def sagittal(self, index=None):
        """
        矢状面 — 沿 X 轴切片
        :param index: 切片索引, 默认中间层
        :return: 2D numpy array (Z, Y)
        """
        if index is None:
            index = int(self.shape[2] * MPR_DEFAULT_RATIO)
        index = np.clip(index, 0, self.shape[2] - 1)
        return self.volume[:, :, index]

    def get_orthogonal_views(self, indices=None):
        """
        获取三个正交面的切片
        :param indices: (axial_idx, coronal_idx, sagittal_idx)
        :return: dict with keys 'axial', 'coronal', 'sagittal'
        """
        if indices is None:
            indices = (None, None, None)
        return {
            'axial': self.axial(indices[0]),
            'coronal': self.coronal(indices[1]),
            'sagittal': self.sagittal(indices[2]),
        }

    def get_slice_by_world(self, axis, world_position):
        """
        根据世界坐标(近似)获取切片
        :param axis: 'axial'/'coronal'/'sagittal'
        :param world_position: 归一化位置 [0, 1]
        :return: 2D numpy array
        """
        axis_map = {'axial': 0, 'coronal': 1, 'sagittal': 2}
        ax = axis_map.get(axis, 0)
        index = int(world_position * (self.shape[ax] - 1))
        index = np.clip(index, 0, self.shape[ax] - 1)
        return getattr(self, axis)(index)
