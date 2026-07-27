"""Divide la ROI en rejilla 16x16 paso 8, calcula histogramas LBP-U locales y concatena el vector global [REQ-RID-05]."""
import numpy as np
from feature_extraction.body.lbp_uniform_extractor import compute_lbp_map, compute_uniform_histogram

BLOCK_SIZE = 16
BLOCK_STRIDE = 8
HIST_BINS = 59


def extract_spatial_grid_lbp(gray_roi: np.ndarray, weight_map: np.ndarray | None = None) -> np.ndarray:
    """
    Extrae el vector LBP-U concatenado sobre una rejilla de bloques 16x16 con paso 8 [REQ-RID-05].
    Si se provee weight_map (de stable_zone_masker), pondera cada histograma de bloque.
    """
    if gray_roi is None or gray_roi.ndim != 2:
        raise ValueError("Se requiere una imagen en escala de grises (2D).")

    lbp_map = compute_lbp_map(gray_roi)
    h, w = lbp_map.shape

    block_histograms = []
    for y in range(0, h - BLOCK_SIZE + 1, BLOCK_STRIDE):
        for x in range(0, w - BLOCK_SIZE + 1, BLOCK_STRIDE):
            block = lbp_map[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            hist = compute_uniform_histogram(block, n_bins=HIST_BINS)

            if weight_map is not None:
                block_weight = float(np.mean(weight_map[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]))
                hist = hist * block_weight

            block_histograms.append(hist)

    if not block_histograms:
        raise ValueError("La ROI es demasiado pequeña para el tamaño de bloque/paso configurado.")

    return np.concatenate(block_histograms, axis=0).astype(np.float32)