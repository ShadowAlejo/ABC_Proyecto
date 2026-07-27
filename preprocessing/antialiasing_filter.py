"""Suavizado gaussiano previo al redimensionamiento de ROIs para evitar artefactos de aliasing."""
import cv2
import numpy as np


def apply_antialiasing(image: np.ndarray, kernel_size: tuple[int, int] = (3, 3), sigma: float = 0.8) -> np.ndarray:
    """Aplica un filtro gaussiano suave previo al resize, según [REQ-INV-03]."""
    if image is None or image.size == 0:
        raise ValueError("Imagen vacía o nula recibida en antialiasing_filter.")
    return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)