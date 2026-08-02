"""Construye la matriz de confusión de 16x16 más la clase 'Desconocido' [REQ-EVL-01]."""
from typing import List
import numpy as np
from sklearn.metrics import confusion_matrix

UNKNOWN_CLASS_NAME = "Desconocido"


def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> np.ndarray:
    """Devuelve la matriz de confusión (N+1)x(N+1) incluyendo la clase Desconocido al final."""
    all_labels = list(range(len(class_names))) + [len(class_names)]  # último índice = Desconocido
    matrix = confusion_matrix(y_true, y_pred, labels=all_labels)
    return matrix


def get_extended_class_names(class_names: List[str]) -> List[str]:
    return list(class_names) + [UNKNOWN_CLASS_NAME]