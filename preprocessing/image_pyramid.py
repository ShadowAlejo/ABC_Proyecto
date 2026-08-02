"""Genera múltiples niveles de escala (s^i) del fotograma para invariancia a la distancia a la cámara."""
from typing import List
import cv2
import numpy as np


def build_image_pyramid(image: np.ndarray, scale_factor: float = 0.75, levels: int = 4) -> List[np.ndarray]:
    """Genera una pirámide de imágenes con factor de escala s^i, i = 0..levels-1 [REQ-INV-04]."""
    if image is None or image.size == 0:
        raise ValueError("Imagen vacía o nula recibida en image_pyramid.")
    if not (0.0 < scale_factor < 1.0):
        raise ValueError("scale_factor debe estar en el rango (0, 1).")

    pyramid = [image]
    current = image
    for i in range(1, levels):
        new_w = max(1, int(current.shape[1] * scale_factor))
        new_h = max(1, int(current.shape[0] * scale_factor))
        current = cv2.resize(current, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pyramid.append(current)
    return pyramid