"""Divide la ROI en rejilla 16x16 paso 8, calcula histogramas LBP-U multi-escala locales y concatena con normalización L2 global [REQ-RID-05]."""
import numpy as np
from feature_extraction.body.lbp_uniform_extractor import (
    compute_multiscale_lbp_maps,
    compute_multiscale_block_histogram,
)

BLOCK_SIZE = 16
BLOCK_STRIDE = 8


def extract_spatial_grid_lbp(gray_roi: np.ndarray, weight_map: np.ndarray | None = None) -> np.ndarray:
    """
    Extrae el vector LBP-U multi-escala concatenado sobre una rejilla 16x16 paso 8 [REQ-RID-05].
    Si se provee weight_map, pondera cada histograma de bloque. Aplica normalización L2 global al final.
    """
    if gray_roi is None or gray_roi.ndim != 2:
        raise ValueError("Se requiere una imagen en escala de grises (2D).")

    map_r1, map_r2 = compute_multiscale_lbp_maps(gray_roi)
    h, w = map_r1.shape

    block_histograms = []
    for y in range(0, h - BLOCK_SIZE + 1, BLOCK_STRIDE):
        for x in range(0, w - BLOCK_SIZE + 1, BLOCK_STRIDE):
            block_r1 = map_r1[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            block_r2 = map_r2[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            hist = compute_multiscale_block_histogram(block_r1, block_r2)

            if weight_map is not None:
                block_weight = float(np.mean(weight_map[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]))
                hist = hist * block_weight

            block_histograms.append(hist)

    if not block_histograms:
        raise ValueError("La ROI es demasiado pequeña para el tamaño de bloque/paso configurado.")

    vector = np.concatenate(block_histograms, axis=0).astype(np.float32)

    # Normalización L2 global (crítico para SVM lineal con pesos zonales)
    norm = np.linalg.norm(vector)
    if norm > 1e-6:
        vector = vector / norm

    return vector