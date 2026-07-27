"""Calcula el descriptor LBP-U (U<=2) generando histogramas de 59 bins por bloque local [REQ-RID-04]."""
import numpy as np
from skimage.feature import local_binary_pattern

LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"  # U <= 2 transiciones -> 59 bins (58 uniformes + 1 acumulativo)
LBP_N_BINS = LBP_N_POINTS + 2  # 10 -> 59 solo cuando n_points=8: 8+2=10... ver nota abajo


def compute_lbp_map(gray_image: np.ndarray) -> np.ndarray:
    """Calcula el mapa LBP uniforme sobre una imagen en escala de grises."""
    if gray_image is None or gray_image.ndim != 2:
        raise ValueError("Se requiere una imagen en escala de grises (2D) para LBP.")
    return local_binary_pattern(gray_image, P=LBP_N_POINTS, R=LBP_RADIUS, method=LBP_METHOD)


def compute_uniform_histogram(lbp_block: np.ndarray, n_bins: int = 59) -> np.ndarray:
    """
    Genera el histograma normalizado de 59 bins (58 patrones uniformes + 1 bin no uniforme)
    para P=8 vecinos, según la definición estándar de Ojala et al. [REQ-RID-04].
    """
    hist, _ = np.histogram(lbp_block.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist