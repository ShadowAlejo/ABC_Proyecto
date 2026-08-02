"""Divide la ROI del cuerpo completo en rejilla 4x8, extrae LBP-U con normalización Hellinger [REQ-RID-05]."""
import numpy as np
from feature_extraction.body.lbp_uniform_extractor import (
    compute_lbp_map,
    compute_uniform_histogram,
)

BLOCK_SIZE = 32
BLOCK_STRIDE = 32

def extract_spatial_grid_lbp(gray_roi: np.ndarray) -> np.ndarray:
    """
    Extrae el vector LBP-U sobre una rejilla de 4x8 bloques (1888 dimensiones).
    """
    if gray_roi is None or gray_roi.ndim != 2:
        raise ValueError("Se requiere una imagen en escala de grises (2D).")

    map_r = compute_lbp_map(gray_roi)
    h, w = map_r.shape  # Debería ser 256x128

    block_hists = []

    # Iterar por filas y columnas sin solapamiento
    for y in range(0, h - BLOCK_SIZE + 1, BLOCK_STRIDE):
        for x in range(0, w - BLOCK_SIZE + 1, BLOCK_STRIDE):
            block_r = map_r[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            hist = compute_uniform_histogram(block_r)
            block_hists.append(hist)

    if not block_hists:
        raise ValueError("La ROI es demasiado pequeña para el tamaño de bloque configurado.")

    # Vector concatenado de 32 bloques * 59 = 1888 dimensiones
    vector = np.concatenate(block_hists, axis=0).astype(np.float32)

    return vector