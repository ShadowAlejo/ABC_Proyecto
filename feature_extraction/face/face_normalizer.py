"""Recorta la region facial, alinea por landmarks, convierte a escala de grises y redimensiona a 64x64 [REQ-FAC-02].

Pipeline de normalizacion (mismo orden en entrenamiento e inferencia):
  1. Recortar bbox con padding del 15%
  2. Convertir a escala de grises
  3. Alinear por landmarks: rotar para dejar los ojos horizontales (invarianza a roll de cabeza)
  4. Resize a 64x64
  5. Enhancement (CLAHE + unsharp masking) — SIEMPRE activo para garantizar consistencia
     entre entrenamiento e inferencia.

Nota sobre enhance_for_training:
  El parametro se mantiene por compatibilidad pero el enhancement se aplica siempre,
  ya que el modelo fue entrenado con el y debe recibir el mismo preprocesado en inferencia.
"""
import cv2
import numpy as np
from preprocessing.roi_resizer import resize_face_roi
from feature_extraction.face.yunet_face_detector import FaceDetectionResult

PADDING_RATIO = 0.15  # margen adicional alrededor del bbox facial detectado

# Parametros de enhancement
_CLAHE_CLIP_LIMIT  = 3.0
_CLAHE_TILE_GRID   = (8, 8)
_UNSHARP_SIGMA     = 1.0
_UNSHARP_ALPHA     = 1.4

# Limite maximo de angulo de alineacion por landmarks (grados)
# Inclinaciones mayores a esto suelen indicar perfil extremo (yaw) no roll.
_MAX_ALIGN_ANGLE_DEG = 20.0


def normalize_face(roi: np.ndarray,
                   face_result: FaceDetectionResult,
                   enhance_for_training: bool = True) -> tuple[np.ndarray, list] | tuple[None, None]:
    """Devuelve la imagen facial normalizada (64x64) y sus landmarks transformados.

    Pipeline identico para entrenamiento e inferencia:
      1. Crop con padding
      2. Escala de grises
      3. Alineacion por landmarks (roll correction) y transformacion de puntos
      4. CLAHE + Unsharp masking
      5. Resize 64x64 y escalado de landmarks

    Returns:
        tuple (face_gray_64x64, landmarks_64x64) o (None, None)
    """
    if not face_result.detected or face_result.bbox is None:
        return None, None

    x, y, w, h = [float(v) for v in face_result.bbox]
    roi_h, roi_w = roi.shape[:2]

    # Guardia contra coordenadas invalidas (inf/nan/degeneradas)
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
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # -- Alineacion por landmarks y transformacion de coordenadas --
    gray, aligned_landmarks = _align_by_landmarks(gray, face_result.landmarks, x1, y1)

    # -- Enhancement: CLAHE + unsharp masking --
    gray = _enhance_gray(gray)

    # -- Resize y escalado final de landmarks a 64x64 --
    final_gray = resize_face_roi(gray)
    
    final_landmarks = []
    if aligned_landmarks and final_gray is not None:
        scale_x = 64.0 / gray.shape[1]
        scale_y = 64.0 / gray.shape[0]
        for (lx, ly) in aligned_landmarks:
            final_landmarks.append((lx * scale_x, ly * scale_y))

    return final_gray, final_landmarks


def _align_by_landmarks(gray_crop: np.ndarray,
                         landmarks,
                         crop_x1: int,
                         crop_y1: int) -> tuple[np.ndarray, list]:
    """Rota el crop facial para correccion de roll y transforma los landmarks.

    Returns:
        tuple (imagen_rotada, landmarks_transformados)
    """
    if landmarks is None or len(landmarks) < 5:
        return gray_crop, []

    # 1. Transformar landmarks del espacio ROI al espacio del crop
    crop_landmarks = []
    for (lx, ly) in landmarks:
        crop_landmarks.append((float(lx) - crop_x1, float(ly) - crop_y1))

    left_eye  = crop_landmarks[0]
    right_eye = crop_landmarks[1]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return gray_crop, crop_landmarks

    angle = float(np.degrees(np.arctan2(dy, dx)))

    # No corregir si el angulo es insignificante o demasiado grande (yaw, no roll)
    if abs(angle) < 1.0 or abs(angle) > _MAX_ALIGN_ANGLE_DEG:
        return gray_crop, crop_landmarks

    # Rotar imagen
    h, w = gray_crop.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(
        gray_crop, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    
    # 2. Rotar landmarks con la misma matriz M
    aligned_landmarks = []
    for (lx, ly) in crop_landmarks:
        # M es 2x3: [ [a, b, tx], [c, d, ty] ]
        # x' = a*x + b*y + tx
        # y' = c*x + d*y + ty
        nx = M[0, 0] * lx + M[0, 1] * ly + M[0, 2]
        ny = M[1, 0] * lx + M[1, 1] * ly + M[1, 2]
        aligned_landmarks.append((nx, ny))

    return aligned, aligned_landmarks


def _enhance_gray(gray: np.ndarray) -> np.ndarray:
    """Aplica CLAHE + unsharp masking a un crop facial en escala de grises.

    - CLAHE: normaliza iluminacion local (invarianza a condiciones de luz).
    - Unsharp masking: realza gradientes de bordes (ojos, nariz, boca)
      para mejorar la discriminacion del descriptor HOG+LBP.
    """
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=_CLAHE_TILE_GRID)
    gray_eq = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray_eq, (0, 0), _UNSHARP_SIGMA)
    sharpened = cv2.addWeighted(gray_eq, _UNSHARP_ALPHA, blurred, -(_UNSHARP_ALPHA - 1.0), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)