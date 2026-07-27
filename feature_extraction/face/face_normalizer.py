"""Recorta la región facial, convierte a escala de grises y redimensiona a 64x64 [REQ-FAC-02].

Modo entrenamiento (enhance_for_training=True):
  Aplica CLAHE + unsharp masking al crop gris antes del resize para producir un
  descriptor HOG más robusto ante variaciones de iluminación y nitidez en el dataset.
"""
import cv2
import numpy as np
from preprocessing.roi_resizer import resize_face_roi
from feature_extraction.face.yunet_face_detector import FaceDetectionResult

PADDING_RATIO = 0.15  # margen adicional alrededor del bbox facial detectado

# Parámetros de enhancement para entrenamiento
_CLAHE_CLIP_LIMIT  = 3.0
_CLAHE_TILE_GRID   = (8, 8)
_UNSHARP_SIGMA     = 1.0
_UNSHARP_ALPHA     = 1.4   # ponderación de la máscara (realce moderado, no agresivo)


def normalize_face(roi: np.ndarray,
                   face_result: FaceDetectionResult,
                   enhance_for_training: bool = False) -> np.ndarray | None:
    """Devuelve la imagen facial en escala de grises, tamaño canónico 64x64, o None si no hay rostro.

    Args:
        roi: Imagen BGR original de la que recortar el rostro.
        face_result: Resultado de YuNetFaceDetector.detect() o detect_training().
        enhance_for_training: Si True, aplica CLAHE y unsharp masking al crop gris
                              para normalizar iluminación y realzar bordes antes de
                              calcular HOG. Solo usar durante construcción del dataset;
                              no activar en inferencia RT.
    """
    if not face_result.detected or face_result.bbox is None:
        return None

    x, y, w, h = [float(v) for v in face_result.bbox]
    roi_h, roi_w = roi.shape[:2]

    pad_x, pad_y = w * PADDING_RATIO, h * PADDING_RATIO
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(roi_w, int(x + w + pad_x))
    y2 = min(roi_h, int(y + h + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None

    face_crop = roi[y1:y2, x1:x2]
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    if enhance_for_training:
        gray = _enhance_gray(gray)

    return resize_face_roi(gray)


def _enhance_gray(gray: np.ndarray) -> np.ndarray:
    """Aplica CLAHE + unsharp masking a un crop facial en escala de grises.

    - CLAHE normaliza la iluminación local, haciendo el HOG invariante a
      condiciones de iluminación desuniformes o globalmente oscuras/claras.
    - Unsharp masking realza gradientes de bordes faciales (ojos, nariz, boca),
      mejorando la discriminación del descriptor en imágenes ligeramente borrosas.
    """
    # 1. CLAHE — normalización de iluminación local
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=_CLAHE_TILE_GRID)
    gray_eq = clahe.apply(gray)

    # 2. Unsharp masking — realce moderado de bordes para HOG
    blurred = cv2.GaussianBlur(gray_eq, (0, 0), _UNSHARP_SIGMA)
    sharpened = cv2.addWeighted(gray_eq, _UNSHARP_ALPHA, blurred, -(_UNSHARP_ALPHA - 1.0), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)