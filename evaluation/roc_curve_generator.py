"""Genera curvas ROC (TPR vs FPR) para seleccionar el umbral óptimo T que minimiza falsos aceptados [REQ-EVL-01]."""
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_curve, auc


@dataclass
class ROCResult:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc_score: float
    optimal_threshold: float


def generate_roc_curve(y_true_binary: np.ndarray, y_scores: np.ndarray) -> ROCResult:
    """
    y_true_binary: 1 si la predicción es correcta (aceptación válida), 0 si es un falso aceptado.
    y_scores: probabilidades/confianzas del SVM tras la conversión logística.
    """
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
    auc_score = auc(fpr, tpr)

    # Umbral óptimo: minimiza FPR mientras maximiza TPR (distancia de Youden).
    youden_index = tpr - fpr
    optimal_idx = int(np.argmax(youden_index))
    optimal_threshold = float(thresholds[optimal_idx])

    return ROCResult(fpr=fpr, tpr=tpr, thresholds=thresholds, auc_score=auc_score,
                      optimal_threshold=optimal_threshold)