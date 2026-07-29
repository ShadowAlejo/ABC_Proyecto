"""Recorta la region facial, alinea por landmarks, y redimensiona a 64x64 [REQ-FAC-02].

Pipeline de normalizacion (Similarity Transform):
  1. Recortar bbox con padding.
  2. Transformacion Afin usando 5 landmarks mapeados a coordenadas canonicas 64x64.
  3. Mantiene el espacio de color BGR para el Opponent-HOG.
"""
import cv2
import numpy as np
from feature_extraction.face.yunet_face_detector import FaceDetectionResult

PADDING_RATIO = 0.15

# Coordenadas canonicas para una imagen 64x64
CANONICAL_LANDMARKS = np.array([
    [20.0, 24.0],  # Left Eye
    [44.0, 24.0],  # Right Eye
    [32.0, 38.0],  # Nose
    [22.0, 50.0],  # Mouth Left
    [42.0, 50.0]   # Mouth Right
], dtype=np.float32)

def normalize_face(roi: np.ndarray,
                   face_result: FaceDetectionResult,
                   enhance_for_training: bool = True) -> tuple[np.ndarray, list] | tuple[None, None]:
    """Devuelve la imagen facial alineada afinalmente (64x64 BGR) y sus landmarks."""
    if not face_result.detected or face_result.bbox is None:
        return None, None

    x, y, w, h = [float(v) for v in face_result.bbox]
    roi_h, roi_w = roi.shape[:2]

    if not all(np.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None, None

    pad_x, pad_y = w * PADDING_RATIO, h * PADDING_RATIO
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(roi_w, int(x + w + pad_x))
    y2 = min(roi_h, int(y + h + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None, None

    face_crop = roi[y1:y2, x1:x2]

    # Transformacion afin de la imagen recortada
    aligned_bgr, final_landmarks = _affine_alignment(face_crop, face_result.landmarks, x1, y1)
    
    return aligned_bgr, final_landmarks

def _affine_alignment(crop_bgr: np.ndarray,
                      landmarks: np.ndarray,
                      crop_x1: int,
                      crop_y1: int) -> tuple[np.ndarray, list]:
    """Alinea usando estimateAffinePartial2D a un lienzo de 64x64."""
    if landmarks is None or len(landmarks) < 5:
        # Fallback a resize directo
        resized = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_LINEAR)
        return resized, []

    # Ajustar landmarks al crop actual
    crop_landmarks = np.array([
        [lx - crop_x1, ly - crop_y1] for lx, ly in landmarks
    ], dtype=np.float32)

    # Transformacion afin (Similarity Transform: rotacion + escala + traslacion)
    M, _ = cv2.estimateAffinePartial2D(crop_landmarks, CANONICAL_LANDMARKS)

    if M is None:
        resized = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_LINEAR)
        return resized, []

    aligned_bgr = cv2.warpAffine(
        crop_bgr, M, (64, 64),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return aligned_bgr, CANONICAL_LANDMARKS.tolist()