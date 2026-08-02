"""Calcula Recall, Precision y Specificity por clase [REQ-EVL-01]."""
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class ClassMetrics:
    class_name: str
    recall: float
    precision: float
    specificity: float


def compute_metrics_per_class(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, ClassMetrics]:
    """
    Recall = RP / (RP + FN)
    Precision = RP / (RP + FP)
    Specificity = RN / (RN + FP)
    """
    metrics: Dict[str, ClassMetrics] = {}
    n = len(y_true)

    for idx, class_name in enumerate(class_names):
        true_positive = int(np.sum((y_true == idx) & (y_pred == idx)))
        false_negative = int(np.sum((y_true == idx) & (y_pred != idx)))
        false_positive = int(np.sum((y_true != idx) & (y_pred == idx)))
        true_negative = n - true_positive - false_negative - false_positive

        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0.0

        metrics[class_name] = ClassMetrics(
            class_name=class_name, recall=recall, precision=precision, specificity=specificity
        )

    return metrics