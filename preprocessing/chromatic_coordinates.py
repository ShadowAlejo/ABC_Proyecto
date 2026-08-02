"""Proyección sobre el plano R+G+B=1 para cancelar el factor de escala de intensidad lumínica."""
import numpy as np


def to_chromatic_coordinates(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convierte una imagen BGR a coordenadas cromáticas invariantes (r, g, b) con r+g+b=1."""
    if image is None or image.size == 0:
        raise ValueError("Imagen vacía o nula recibida en chromatic_coordinates.")

    img_float = image.astype(np.float32)
    b, g, r = img_float[..., 0], img_float[..., 1], img_float[..., 2]
    total = r + g + b + eps

    chromatic = np.stack([b / total, g / total, r / total], axis=-1)
    return chromatic.astype(np.float32)