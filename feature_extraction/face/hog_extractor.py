"""Extrae el descriptor facial HOG optimizado en canal único de Intensidad con filtro Tan & Triggs canónico.

Arquitectura de Subespacios Geométricos con Solapamiento Anatómico:
  1. Filtro de Iluminación Tan & Triggs Canónico en 4 etapas (Gamma + DoG + Ecualización L_alpha + Truncamiento e Hiperbólica).
  2. Gradientes espaciales calculados con diferencias centrales 1D [-1, 0, 1].
  3. HOG Coarse Global (96x96): 1,200 dimensiones.
  4. HOG Región Superior Ojos/Cejas (Y: 16-64, X: 16-80, 48x64 px): 1,680 dimensiones.
  5. HOG Región Inferior Nariz/Boca (Y: 48-88, X: 24-72, 40x48 px): 960 dimensiones.
     (Solapamiento de 16 px en Y: 48-64 para continuidad anatómica del puente y punta nasal).
  6. Vector total concatenado y normalizado L2 por subespacio: 3,840 dimensiones.
"""
import cv2
import numpy as np


def _apply_tan_triggs(gray: np.ndarray, alpha: float = 0.1, tau: float = 10.0,
                      gamma: float = 0.2, sigma0: float = 1.0, sigma1: float = 2.0) -> np.ndarray:
    """Filtro de Iluminación Tan & Triggs canónico en 4 etapas."""
    # 1. Corrección Gamma (compresión de rango dinámico)
    img = np.power(gray.astype(np.float32) / 255.0, gamma)

    # 2. Difference of Gaussians (DoG)
    g1 = cv2.GaussianBlur(img, (0, 0), sigma0)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma1)
    dog = g1 - g2

    # 3. Primera etapa de ecualización de contraste (norma L_alpha)
    mean_abs = float(np.mean(np.abs(dog) ** alpha)) ** (1.0 / alpha)
    norm1 = dog / (mean_abs + 1e-5)

    # 4. Segunda etapa de ecualización y truncamiento no lineal
    mean_min = float(np.mean(np.minimum(tau, np.abs(norm1)) ** alpha)) ** (1.0 / alpha)
    norm2 = norm1 / (mean_min + 1e-5)
    compressed = tau * np.tanh(norm2 / tau)

    # Mapeo a rango [0, 255] uint8
    return cv2.normalize(compressed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def _compute_hog_cells(image: np.ndarray, cell_h: int, cell_w: int, nbins: int = 12) -> np.ndarray:
    """Calcula histogramas de gradientes no firmados por celda usando diferencias centrales 1D."""
    h, w = image.shape[:2]
    img_f = image.astype(np.float32)

    # Operadores 1D de diferencia central [-1, 0, 1]
    kernel_x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    kernel_y = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
    gx = cv2.filter2D(img_f, cv2.CV_32F, kernel_x, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.filter2D(img_f, cv2.CV_32F, kernel_y, borderType=cv2.BORDER_REFLECT_101)

    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180.0

    n_cells_y = h // cell_h
    n_cells_x = w // cell_w
    bin_width = 180.0 / nbins
    bin_idx = np.clip((ang / bin_width).astype(np.int32), 0, nbins - 1)

    cell_hist = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)
    for b in range(nbins):
        mask = (bin_idx == b).astype(np.float32) * mag
        grid = mask[:n_cells_y * cell_h, :n_cells_x * cell_w].reshape(n_cells_y, cell_h, n_cells_x, cell_w)
        cell_hist[:, :, b] = grid.sum(axis=(1, 3))

    return cell_hist


def _compute_hog_descriptor(image: np.ndarray, cell_size: tuple[int, int],
                           block_cells: tuple[int, int] = (2, 2), nbins: int = 12) -> np.ndarray:
    """Calcula descriptor HOG con normalización por bloques y L2-Hys clipping."""
    cell_w, cell_h = cell_size
    bx, by = block_cells
    cell_hist = _compute_hog_cells(image, cell_h, cell_w, nbins)

    n_cells_y, n_cells_x, _ = cell_hist.shape
    n_blocks_y = n_cells_y - by + 1
    n_blocks_x = n_cells_x - bx + 1

    blocks = []
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            blk = cell_hist[i:i+by, j:j+bx].flatten()
            norm = np.linalg.norm(blk) + 1e-5
            blk = blk / norm
            blk = np.clip(blk, 0, 0.2)
            blk = blk / (np.linalg.norm(blk) + 1e-5)
            blocks.append(blk)

    return np.concatenate(blocks).astype(np.float32) if blocks else np.array([], dtype=np.float32)


def _compute_channel_hog(channel_8u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula HOG Global + Región Superior (Ojos/Cejas) + Región Inferior (Nariz/Boca) con solapamiento."""
    # 1. Global Coarse (96x96, celdas 16x16, bloques 2x2, 12 bins -> 1200 dims)
    global_hog = _compute_hog_descriptor(channel_8u, cell_size=(16, 16), block_cells=(2, 2), nbins=12)
    norm_g = np.linalg.norm(global_hog)
    if norm_g > 1e-6:
        global_hog = global_hog / norm_g

    # 2. Región Superior Ojos/Cejas: Y: 16-64 (alto 48 px), X: 16-80 (ancho 64 px) -> 1680 dims
    crop_upper = channel_8u[16:64, 16:80]
    feat_upper = _compute_hog_descriptor(crop_upper, cell_size=(8, 8), block_cells=(2, 2), nbins=12)
    norm_u = np.linalg.norm(feat_upper)
    if norm_u > 1e-6:
        feat_upper = feat_upper / norm_u

    # 3. Región Inferior Nariz/Boca: Y: 48-88 (alto 40 px), X: 24-72 (ancho 48 px) -> 960 dims
    crop_lower = channel_8u[48:88, 24:72]
    feat_lower = _compute_hog_descriptor(crop_lower, cell_size=(8, 8), block_cells=(2, 2), nbins=12)
    norm_l = np.linalg.norm(feat_lower)
    if norm_l > 1e-6:
        feat_lower = feat_lower / norm_l

    return global_hog, feat_upper, feat_lower


def extract_hog_features(face_bgr_96x96: np.ndarray, landmarks: list = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrae HOG usando el canal de Intensidad con filtro Tan & Triggs canónico y subespacios anatómicos.

    Args:
        face_bgr_96x96: Imagen BGR de 96x96 alineada afinalmente.
        landmarks: Lista de puntos faciales (mantenido para compatibilidad de firma).

    Returns:
        Tupla de 3 vectores resultantes: (Global [1200 dims], Superior [1680 dims], Inferior [960 dims]).
    """
    if face_bgr_96x96 is None or face_bgr_96x96.shape[:2] != (96, 96):
        raise ValueError("Se requiere imagen facial alineada de 96x96 BGR.")

    # Canal Único: Intensidad (Grayscale) con Tan & Triggs canónico
    gray = cv2.cvtColor(face_bgr_96x96, cv2.COLOR_BGR2GRAY)
    channel_intensity = _apply_tan_triggs(gray)

    return _compute_channel_hog(channel_intensity)