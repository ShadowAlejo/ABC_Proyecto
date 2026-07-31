"""Extrae el descriptor facial HOG maximizado (Piramidal + Componentes locales + Opponent-HOG).

Para maximizar la precision, esta implementacion aplica:
  1. Filtro Tan & Triggs (Gamma + DoG + Truncado) a la Intensidad.
  2. Gradientes con signo (0-360) (signedGradient=True).
  3. HOG Piramidal (escalas 4x4, 8x8, 16x16).
  4. HOG Basado en Componentes Locales (6 parches: ojos, nariz, comisuras, entrecejo).
  5. Opponent-HOG: extraccion en canales de Intensidad, O1 (R-G) y O2 (R+G-2B).
"""
import cv2
import numpy as np

# ── HOG Piramidal Global (96x96) con 12 orientaciones y SIN signo ──────────────────
# Atributos posicionales de OpenCV HOGDescriptor:
# winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient

_HOG_COARSE = cv2.HOGDescriptor(
    (96, 96), (32, 32), (16, 16), (16, 16), 12, 1, -1.0, 0, 0.2, False, 64, False
)

# ── HOG de Regiones Locales (Superior e Inferior) SIN signo ────────────
_HOG_UPPER = cv2.HOGDescriptor(
    (64, 40), (16, 16), (8, 8), (8, 8), 12, 1, -1.0, 0, 0.2, False, 64, False
)
_HOG_LOWER = cv2.HOGDescriptor(
    (48, 32), (16, 16), (8, 8), (8, 8), 12, 1, -1.0, 0, 0.2, False, 64, False
)


def _apply_tan_triggs(gray: np.ndarray, alpha=0.1, tau=10.0, gamma=0.2, sigma0=1, sigma1=2) -> np.ndarray:
    """Filtro de Iluminacion Tan & Triggs: Gamma -> DoG -> Truncado de contraste."""
    # 1. Gamma Correction
    img = np.power(gray.astype(np.float32) / 255.0, gamma)

    # 2. Difference of Gaussians (DoG)
    g1 = cv2.GaussianBlur(img, (0, 0), sigma0)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma1)
    dog = g1 - g2

    # 3. Contrast Truncation
    dog = dog / (np.mean(np.abs(dog)) ** alpha)
    dog = dog / (np.mean(np.minimum(np.abs(dog), tau)) ** alpha)
    dog = tau * np.tanh(dog / tau)
    
    return cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def _compute_channel_hog(channel_8u: np.ndarray) -> np.ndarray:
    """Calcula HOG Piramidal + Regiones para un solo canal de 8 bits."""
    # Piramidal Global (Solo Coarse para balancear dimensiones ~ 1200 vars)
    global_hog = _HOG_COARSE.compute(channel_8u).flatten().astype(np.float32)

    norm_g = np.linalg.norm(global_hog)
    if norm_g > 1e-6:
        global_hog = global_hog / norm_g

    # Región Superior (Ojos/Cejas) - Y: 16-56, X: 16-80
    crop_upper = channel_8u[16:56, 16:80]
    feat_upper = _HOG_UPPER.compute(crop_upper).flatten().astype(np.float32)
    norm_u = np.linalg.norm(feat_upper)
    if norm_u > 1e-6:
        feat_upper = feat_upper / norm_u

    # Región Inferior (Nariz/Boca) - Y: 56-88, X: 24-72
    crop_lower = channel_8u[56:88, 24:72]
    feat_lower = _HOG_LOWER.compute(crop_lower).flatten().astype(np.float32)
    norm_l = np.linalg.norm(feat_lower)
    if norm_l > 1e-6:
        feat_lower = feat_lower / norm_l

    return np.concatenate([global_hog, feat_upper, feat_lower], axis=0)


def extract_hog_features(face_bgr_96x96: np.ndarray, landmarks: list = None) -> np.ndarray:
    """Extrae HOG usando únicamente el canal de Intensidad con filtro Tan & Triggs.

    Args:
        face_bgr_96x96: Imagen BGR de 96x96 alineada afinalmente.
        landmarks: Lista de 5 puntos (x, y) de la cara alineada (No usado, mantenido).

    Returns:
        Vector resultante concatenado de 3264 dimensiones.
    """
    if face_bgr_96x96 is None or face_bgr_96x96.shape[:2] != (96, 96):
        raise ValueError("Se requiere imagen facial alineada de 96x96 BGR.")

    # Canal Único: Intensidad (Grayscale) con Tan & Triggs
    gray = cv2.cvtColor(face_bgr_96x96, cv2.COLOR_BGR2GRAY)
    channel_intensity = _apply_tan_triggs(gray)

    return _compute_channel_hog(channel_intensity)