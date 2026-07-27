"""Calcula los pesos C_j por clase para class_weight='balanced' [REQ-ENT-01]."""
from typing import Dict
import numpy as np


def compute_class_weights(labels: np.ndarray, base_c: float = 1.0) -> Dict[int, float]:
    """
    C_j = C * [ N_total / (N_clases * N_j) ]
    Devuelve un diccionario {clase: peso_C_j} equivalente a sklearn's class_weight='balanced'.
    """
    unique_classes, counts = np.unique(labels, return_counts=True)
    n_total = len(labels)
    n_classes = len(unique_classes)

    weights: Dict[int, float] = {}
    for cls, n_j in zip(unique_classes, counts):
        weights[int(cls)] = base_c * (n_total / (n_classes * n_j))

    return weights