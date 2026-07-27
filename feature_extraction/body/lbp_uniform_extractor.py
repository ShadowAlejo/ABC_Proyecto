"""Calcula descriptores LBP-U (U<=2) multi-escala para captura de patrones locales y estructurales [REQ-RID-04]."""
import numpy as np
from skimage.feature import local_binary_pattern

# Escala 1: Fina (R=1, P=8 -> 59 bins)
LBP_R1 = 1
LBP_P1 = 8

# Escala 2: Gruesa / Estructural (R=2, P=16 -> 243 bins)
LBP_R2 = 2
LBP_P2 = 16

LBP_METHOD = "uniform"

BINS_R1 = LBP_P1 * (LBP_P1 - 1) + 3  # 59
BINS_R2 = LBP_P2 * (LBP_P2 - 1) + 3  # 243


def compute_lbp_map(gray_image: np.ndarray) -> np.ndarray:
    """Calcula el mapa LBP uniforme de escala fina (R=1, P=8)."""
    if gray_image is None or gray_image.ndim != 2:
        raise ValueError("Se requiere una imagen en escala de grises (2D) para LBP.")
    return local_binary_pattern(gray_image, P=LBP_P1, R=LBP_R1, method=LBP_METHOD)


def compute_multiscale_lbp_maps(gray_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula los mapas LBP uniformes a dos escalas: R=1 (fina) y R=2 (gruesa)."""
    if gray_image is None or gray_image.ndim != 2:
        raise ValueError("Se requiere una imagen en escala de grises (2D) para LBP.")

    map_r1 = local_binary_pattern(gray_image, P=LBP_P1, R=LBP_R1, method=LBP_METHOD)
    map_r2 = local_binary_pattern(gray_image, P=LBP_P2, R=LBP_R2, method=LBP_METHOD)
    return map_r1, map_r2


def compute_uniform_histogram(lbp_block: np.ndarray, n_bins: int = BINS_R1) -> np.ndarray:
    """Genera el histograma normalizado para un bloque LBP dado y número de bins."""
    hist, _ = np.histogram(lbp_block.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def compute_multiscale_block_histogram(block_r1: np.ndarray, block_r2: np.ndarray) -> np.ndarray:
    """Calcula y concatena los histogramas normalizados de ambas escalas (59 + 243 = 302 bins)."""
    h1 = compute_uniform_histogram(block_r1, n_bins=BINS_R1)
    h2 = compute_uniform_histogram(block_r2, n_bins=BINS_R2)
    return np.concatenate([h1, h2], axis=0)