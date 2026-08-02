"""Recorta y normaliza la región facial/cabeza (96x96 BGR).

Estrategia:
  - Si is_body_roi=True y h > 1.5 * w (detección de persona completa), extrae la región superior de la cabeza (28-32% superior).
  - Si is_body_roi=False (foto de rostro de raw_images), preserva la cara completa sin recortarla.
  - Normaliza la resolución a 96x96 BGR con interpolación bilineal.
"""
import cv2
import numpy as np


def normalize_face(roi: np.ndarray, is_body_roi: bool = True) -> np.ndarray | None:
    """Devuelve la imagen facial/cabeza normalizada (96x96 BGR).

    Args:
        roi: Imagen BGR de entrada (cuerpo detectado o foto de rostro).
        is_body_roi: Si es True, asume que es una caja de cuerpo entero y extrae la cabeza.
                     Si es False, asume que ya es una foto de rostro/cabeza y usa la imagen completa.
    """
    if roi is None or roi.size == 0 or roi.ndim != 3:
        return None

    h, w = roi.shape[:2]
    if h <= 0 or w <= 0:
        return None

    if is_body_roi and (h > int(w * 1.5)):
        # Cuerpo entero detectado por YOLO: extraer el 30% superior (cabeza completa con ojos, nariz y boca)
        head_h = max(16, int(h * 0.30))
        face_crop = roi[0:head_h, 0:w]
    else:
        # Foto de rostro/cabeza (de raw_images): usar la imagen completa
        face_crop = roi

    if face_crop.size == 0:
        return None

    # Resize estandarizado a 96x96 BGR
    normalized_96x96 = cv2.resize(face_crop, (96, 96), interpolation=cv2.INTER_LINEAR)
    return normalized_96x96