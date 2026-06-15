"""
综合评估器 — 自动对比四种分割方法的指标
生成对比表格 (Dice/IoU/MSE/MI/NMI)
"""
import numpy as np
from .my_similarity import my_dice, my_iou, my_mse, my_mi, my_nmi


class SegmentationEvaluator:
    """多方法分割评估对比系统"""

    def __init__(self, ground_truth):
        """
        :param ground_truth: 金标准分割 (二值或整数标签)
        """
        self.ground_truth = np.asarray(ground_truth)

    def evaluate_method(self, prediction, method_name):
        """
        评估单个分割方法
        :param prediction: 预测分割结果
        :param method_name: 方法名称
        :return: 指标字典
        """
        pred = np.asarray(prediction)
        gt = self.ground_truth

        # 统一尺寸
        if pred.shape != gt.shape:
            # 重采样到GT尺寸
            from module2_preprocessing.my_interpolation import my_bilinear_interpolation
            pred = my_bilinear_interpolation(pred, gt.shape[:2])

        metrics = {
            "method": method_name,
            "Dice": my_dice(pred > 0, gt > 0),
            "IoU": my_iou(pred > 0, gt > 0),
        }

        # MSE 针对二值掩膜
        mse_val = my_mse((pred > 0).astype(np.float64), (gt > 0).astype(np.float64))
        metrics["MSE"] = mse_val

        # MI/NMI (使用二值掩膜)
        metrics["MI"] = my_mi((pred > 0).astype(np.float64) * 255,
                              (gt > 0).astype(np.float64) * 255, bins=64)
        metrics["NMI"] = my_nmi((pred > 0).astype(np.float64) * 255,
                                (gt > 0).astype(np.float64) * 255, bins=64)

        return metrics

    def evaluate_all(self, predictions_dict):
        """
        评估所有方法
        :param predictions_dict: {"方法名": 预测结果, ...}
        :return: 评估指标列表和对比表格文本
        """
        results = []
        for name, pred in predictions_dict.items():
            metrics = self.evaluate_method(pred, name)
            results.append(metrics)

        return results

    def generate_report(self, predictions_dict):
        """生成Markdown格式的评估报告"""
        results = self.evaluate_all(predictions_dict)

        # 表头
        lines = []
        lines.append("# 分割方法评估对比报告")
        lines.append("")
        lines.append("| 方法 | Dice ↑ | IoU ↑ | MSE ↓ | MI ↑ | NMI ↑ |")
        lines.append("|------|--------|-------|-------|------|-------|")

        for r in results:
            lines.append(
                f"| {r['method']} | {r['Dice']:.4f} | {r['IoU']:.4f} | "
                f"{r['MSE']:.4f} | {r['MI']:.4f} | {r['NMI']:.4f} |"
            )

        lines.append("")

        # 最佳方法
        best_dice = max(results, key=lambda r: r['Dice'])
        best_iou = max(results, key=lambda r: r['IoU'])
        lines.append(f"- 🏆 最高 Dice: **{best_dice['method']}** ({best_dice['Dice']:.4f})")
        lines.append(f"- 🏆 最高 IoU: **{best_iou['method']}** ({best_iou['IoU']:.4f})")

        report = "\n".join(lines)

        # 同时返回结构化数据
        table_data = {
            "headers": ["方法", "Dice ↑", "IoU ↑", "MSE ↓", "MI ↑", "NMI ↑"],
            "rows": [[r['method'], f"{r['Dice']:.4f}", f"{r['IoU']:.4f}",
                      f"{r['MSE']:.4f}", f"{r['MI']:.4f}", f"{r['NMI']:.4f}"]
                     for r in results],
        }

        return report, table_data

    def compare_with_gt(self, prediction):
        """生成预测与金标准的差异可视化数据"""
        pred = np.asarray(prediction) > 0
        gt = self.ground_truth > 0

        return {
            "TP": (pred & gt).astype(np.int32),
            "FP": (pred & ~gt).astype(np.int32),
            "FN": (~pred & gt).astype(np.int32),
            "TN": (~pred & ~gt).astype(np.int32),
        }
