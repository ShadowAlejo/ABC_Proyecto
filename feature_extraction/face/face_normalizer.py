"""Recorta la región facial, alinea usando exclusivamente landmarks rígidos (ojos) y proyecta a 96x96 [REQ-FAC-02].

Pipeline de normalización (Similarity Transform Rígido):
  1. Recortar bbox con padding del 15%.
  2. Transformación Afín de Similitud (rotación, escala y traslación) calculada sobre los centros oculares fijos.
  3. Excluye comisuras bucales para evitar distorsiones por gestos, habla o expresiones faciales.
"""
import cv2
import numpy as np
from feature_extraction.face.yoloface_detector import FaceDetectionResult

PADDING_RATIO = 0.15

# Coordenadas canónicas rígidas para una imagen 96x96 (centros oculares fijos)
CANONICAL_EYES = np.array([
    [30.0, 36.0],  # Left Eye
    [66.0, 36.0],  # Right Eye
], dtype=np.float32)

CANONICAL_LANDMARKS_FULL = np.array([
    [30.0, 36.0],  # Left Eye
    [66.0, 36.0],  # Right Eye
    [48.0, 57.0],  # Nose
    [33.0, 75.0],  # Mouth Left
    [63.0, 75.0]   # Mouth Right
], dtype=np.float32)


def normalize_face(roi: np.ndarray,
                   face_result: FaceDetectionResult,
                   enhance_for_training: bool = True,
                   custom_landmarks: np.ndarray | list | None = None) -> tuple[np.ndarray, list] | tuple[None, None]:
    """Devuelve la imagen facial alineada afinalmente (96x96 BGR) y sus landmarks."""
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

    # Usar landmarks perturbados si se proveen (Jittering previo), o los originales
    lms = custom_landmarks if custom_landmarks is not None else face_result.landmarks

    # Transformación afín rígida basada en los ojos
    aligned_bgr, final_landmarks = _affine_alignment(face_crop, lms, x1, y1)
    
    return aligned_bgr, final_landmarks


def _affine_alignment(crop_bgr: np.ndarray,
                      landmarks: np.ndarray,
                      crop_x1: int,
                      crop_y1: int) -> tuple[np.ndarray, list]:
    """Alinea usando estimateAffinePartial2D sobre los 2 puntos oculares rígidos."""
    if landmarks is None or len(landmarks) < 2:
        # Fallback a resize directo
        resized = cv2.resize(crop_bgr, (96, 96), interpolation=cv2.INTER_LINEAR)
        return resized, []

    # Ajustar landmarks de ojos al crop actual (landmarks[0] = ojo_izq, landmarks[1] = ojo_der)
    crop_eyes = np.array([
        [landmarks[0][0] - crop_x1, landmarks[0][1] - crop_y1],
        [landmarks[1][0] - crop_x1, landmarks[1][1] - crop_y1]
    ], dtype=np.float32)

    # Transformación de similitud exacta (2 puntos = rotación + escala + traslación sin deformación por boca)
    M, _ = cv2.estimateAffinePartial2D(crop_eyes, CANONICAL_EYES)

    if M is None:
        resized = cv2.resize(crop_bgr, (96, 96), interpolation=cv2.INTER_LINEAR)
        return resized, []

    aligned_bgr = cv2.warpAffine(
        crop_bgr, M, (96, 96),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return aligned_bgr, CANONICAL_LANDMARKS_FULL.tolist()