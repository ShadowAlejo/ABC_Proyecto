"""Calcula descriptores LBP-U (R=1, P=8) con normalización Hellinger (L1-sqrt) [REQ-RID-04]."""
import numpy as np
from skimage.feature import local_binary_pattern

# Escala única: Fina (R=1, P=8 -> 59 bins)
LBP_R = 1
LBP_P = 8
LBP_METHOD = "uniform"
BINS = LBP_P * (LBP_P - 1) + 3  # 59

def compute_lbp_map(gray_image: np.ndarray) -> np.ndarray:
    """Calcula el mapa LBP uniforme de escala fina (R=1, P=8)."""
    if gray_image is None or gray_image.ndim != 2:
        raise ValueError("Se requiere una imagen en escala de grises (2D) para LBP.")
    return local_binary_pattern(gray_image, P=LBP_P, R=LBP_R, method=LBP_METHOD)

def compute_uniform_histogram(lbp_block: np.ndarray) -> np.ndarray:
    """Genera el histograma Hellinger-normalizado (L1-sqrt) para un bloque LBP."""
    hist, _ = np.histogram(lbp_block.ravel(), bins=BINS, range=(0, BINS))
    hist = hist.astype(np.float32)
    
    # Normalización Hellinger (Block-wise L1-sqrt)
    eps = 1e-7
    hist_l1 = hist / (hist.sum() + eps)
    hist_hellinger = np.sqrt(hist_l1)
    
    return hist_hellinger