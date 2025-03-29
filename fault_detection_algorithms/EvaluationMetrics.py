import numpy as np
from sklearn.metrics import roc_curve, auc


class BasicErrorMetricsCalculator:
    def __init__(self, predicted, expected, fault_ids=None):
        self.TP = np.sum(predicted & expected)
        self.FP = np.sum(predicted & ~expected)
        self.FN = np.sum(~predicted & expected)
        self.TN = np.sum(~predicted & ~expected)

        self.fault_detection_rate = self.TP/(self.TP + self.FN)
        self.false_detection_rate = self.FP/(self.TP + self.FP)
        self.false_alarm_rate = self.FP/(self.TN + self.FP)
        self.by_fault_ids = dict()
        if fault_ids is not None:
            for fid in np.unique(fault_ids):
                selector = fault_ids == fid
                self.by_fault_ids[fid] = BasicErrorMetricsCalculator(
                    predicted[selector], expected[selector])


class AdvanceErrorMetricsCalculator:
    def __init__(self, score, expected, fault_ids=None):
        fpr, tpr, thresholds = roc_curve(expected, score)
        self.fault_detection_rate = tpr
        self.false_alarm_rate = fpr
        self.thresholds = thresholds
        self.roc_auc = auc(fpr, tpr)

        self.by_fault_ids = dict()
        if fault_ids is not None:
            for fid in np.unique(fault_ids):
                selector = fault_ids == fid
                self.by_fault_ids[fid] = AdvanceErrorMetricsCalculator(
                    score[selector], expected[selector])
