"""Calcula la varianza del Laplaciano sobre la ROI y descarta fotogramas borrosos [REQ-CAP-03]."""
import cv2
import numpy as np

DEFAULT_SHARPNESS_THRESHOLD = 100.0


def compute_laplacian_variance(roi: np.ndarray) -> float:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def passes_sharpness_filter(roi: np.ndarray, threshold: float = DEFAULT_SHARPNESS_THRESHOLD) -> bool:
    """Devuelve True si la ROI no está borrosa (varianza del Laplaciano >= umbral)."""
    if roi is None or roi.size == 0:
        return False
    return compute_laplacian_variance(roi) >= threshold