"""
后台计算线程 (QThread)
所有耗时计算在后台线程执行，避免阻塞 GUI
"""
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import traceback


class BaseWorker(QThread):
    """基础工作线程"""
    finished = pyqtSignal(object)  # 结果
    error = pyqtSignal(str)        # 错误信息
    progress = pyqtSignal(int)     # 进度 (0-100)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}")


class ThresholdWorker(QThread):
    """阈值分割后台计算"""
    finished = pyqtSignal(object, object)  # (binary, threshold_value)
    error = pyqtSignal(str)

    def __init__(self, func, image, **params):
        super().__init__()
        self.func = func
        self.image = image
        self.params = params

    def run(self):
        try:
            result = self.func(self.image, **self.params)
            self.finished.emit(result[0], result[1])
        except Exception as e:
            self.error.emit(str(e))


class RegionGrowWorker(QThread):
    """区域生长后台计算"""
    finished = pyqtSignal(object)  # mask
    error = pyqtSignal(str)

    def __init__(self, func, image, seed, **params):
        super().__init__()
        self.func = func
        self.image = image
        self.seed = seed
        self.params = params

    def run(self):
        try:
            result = self.func(self.image, self.seed, **self.params)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class KMeansWorker(QThread):
    """K-means 后台计算"""
    finished = pyqtSignal(object, object)  # (labels, centers)
    error = pyqtSignal(str)

    def __init__(self, func, image, **params):
        super().__init__()
        self.func = func
        self.image = image
        self.params = params

    def run(self):
        try:
            labels, centers = self.func(self.image, **self.params)
            self.finished.emit(labels, centers)
        except Exception as e:
            self.error.emit(str(e))


class TrainWorker(QThread):
    """深度学习训练后台线程"""
    finished = pyqtSignal(object)  # model
    progress = pyqtSignal(int, float, float)  # (epoch, train_loss, val_dice)
    error = pyqtSignal(str)
    log_signal = pyqtSignal(str)  # 日志消息

    def __init__(self, train_func, model, dataset, val_dataset, **params):
        super().__init__()
        self.train_func = train_func
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.params = params

    def run(self):
        try:
            history = self.train_func(
                self.model, self.dataset, self.val_dataset, **self.params
            )
            self.finished.emit(self.model)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}")


class EvalWorker(QThread):
    """评估后台计算"""
    finished = pyqtSignal(str, list)  # (report_text, table_data)
    error = pyqtSignal(str)

    def __init__(self, evaluator, predictions_dict):
        super().__init__()
        self.evaluator = evaluator
        self.predictions_dict = predictions_dict

    def run(self):
        try:
            report, table_data = self.evaluator.generate_report(self.predictions_dict)
            self.finished.emit(report, table_data['rows'])
        except Exception as e:
            self.error.emit(str(e))
