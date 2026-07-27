"""Recorta la región facial, convierte a escala de grises y redimensiona a 64x64 [REQ-FAC-02]."""
import cv2
import numpy as np
from preprocessing.roi_resizer import resize_face_roi
from feature_extraction.face.yunet_face_detector import FaceDetectionResult

PADDING_RATIO = 0.15  # margen adicional alrededor del bbox facial detectado


def normalize_face(roi: np.ndarray, face_result: FaceDetectionResult) -> np.ndarray | None:
    """Devuelve la imagen facial en escala de grises, tamaño canónico 64x64, o None si no hay rostro."""
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
    return resize_face_roi(gray)