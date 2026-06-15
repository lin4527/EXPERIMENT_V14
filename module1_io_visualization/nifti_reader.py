"""
NIfTI 格式医学影像读写与元数据解析
"""
import numpy as np
import nibabel as nib
import os


class NiftiReader:
    """NIfTI 格式 3D 体数据读取器"""

    def __init__(self, filepath=None):
        self.filepath = filepath
        self._data = None
        self._header = None
        self._affine = None
        if filepath and os.path.exists(filepath):
            self.load(filepath)

    def load(self, filepath):
        """加载 NIfTI 文件"""
        self.filepath = filepath
        img = nib.load(filepath)
        self._data = img.get_fdata()
        self._header = img.header
        self._affine = img.affine
        return self

    @property
    def data(self):
        """返回 3D 体数据 (numpy array)"""
        return self._data

    @property
    def shape(self):
        """返回数据维度"""
        return self._data.shape if self._data is not None else None

    @property
    def header(self):
        return self._header

    @property
    def affine(self):
        return self._affine

    def get_metadata(self):
        """解析并返回关键元数据字典"""
        if self._header is None:
            return {}
        h = self._header
        pixdim = h.get('pixdim', [1, 1, 1, 1])
        return {
            "dimensions": list(self._data.shape) if self._data is not None else [],
            "voxel_size": [float(pixdim[1]), float(pixdim[2]), float(pixdim[3])],
            "slice_thickness": float(pixdim[3]) if len(pixdim) > 3 else float(pixdim[1]),
            "datatype": str(self._data.dtype) if self._data is not None else "unknown",
            "intent": h.get('intent_name', 'N/A').decode() if hasattr(h.get('intent_name', b''), 'decode') else str(h.get('intent_name', 'N/A')),
            "description": h.get('descrip', b'')[:80].decode(errors='replace') if hasattr(h.get('descrip', b''), 'decode') else '',
            "affine_matrix": self._affine.tolist() if self._affine is not None else [],
            "value_range": [float(self._data.min()), float(self._data.max())] if self._data is not None else [],
        }

    def get_slice(self, axis, index):
        """
        提取指定方向的切片
        :param axis: 'axial'(0), 'coronal'(1), 'sagittal'(2)
        :param index: 切片索引
        :return: 2D numpy array
        """
        if self._data is None:
            return None
        axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
        ax = axis_map.get(axis, 2)
        return np.take(self._data, index, axis=ax).squeeze()

    def get_center_slice(self, axis):
        """获取中心切片"""
        if self._data is None:
            return None
        axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
        ax = axis_map.get(axis, 2)
        center = self._data.shape[ax] // 2
        return self.get_slice(axis, center)

    def save_nifti(self, data, filepath, reference_obj=None):
        """保存数据为 NIfTI 格式"""
        if reference_obj is not None:
            # 使用参考对象的 affine
            affine = reference_obj.affine if hasattr(reference_obj, 'affine') else reference_obj
        else:
            affine = self._affine
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(img, filepath)
        return filepath
