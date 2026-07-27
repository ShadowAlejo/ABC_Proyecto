"""Descarta ROIs que no superen dimensiones mínimas en píxeles [REQ-CAP: filtro de resolución]."""
import numpy as np

DEFAULT_MIN_WIDTH = 40
DEFAULT_MIN_HEIGHT = 80


def passes_resolution_filter(roi: np.ndarray, min_width: int = DEFAULT_MIN_WIDTH,
                              min_height: int = DEFAULT_MIN_HEIGHT) -> bool:
    """Verifica que la ROI supere las dimensiones mínimas para evitar degradación de detalle LBP."""
    if roi is None or roi.size == 0:
        return False
    h, w = roi.shape[:2]
    return w >= min_width and h >= min_height