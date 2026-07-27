"""Calcula el descriptor HoG sobre la imagen facial normalizada [REQ-FAC-03]."""
import cv2
import numpy as np

_HOG = cv2.HOGDescriptor(
    _winSize=(64, 64),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)


def extract_hog_features(face_gray_64x64: np.ndarray) -> np.ndarray:
    """Extrae el vector de características HoG de una imagen facial en escala de grises 64x64."""
    if face_gray_64x64 is None or face_gray_64x64.shape[:2] != (64, 64):
        raise ValueError("Se requiere una imagen facial normalizada de 64x64 en escala de grises.")

    features = _HOG.compute(face_gray_64x64)
    return features.flatten().astype(np.float32)