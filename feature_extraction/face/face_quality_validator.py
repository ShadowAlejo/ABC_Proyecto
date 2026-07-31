"""Valida la calidad de un rostro detectado antes de usarlo para entrenamiento o inferencia.
Reduce falsos positivos 'técnicamente válidos mas' inútiles (borrosos, diminutos, muy rotados).

Modos:
  - training_mode=False (defecto): umbrales estrictos para inferencia RT.
  - training_mode=True : umbrales relajados para maximizar muestras del dataset de entrenamiento.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from feature_extraction.face.yunet_face_detector import FaceDetectionResult

# ── Umbrales modo INFERENCIA (tiempo real) ──────────────────────────────────
INF_MIN_FACE_WIDTH_PX      = 32
INF_MIN_FACE_HEIGHT_PX     = 32
INF_MIN_SHARPNESS_VARIANCE = 25.0   # Laplaciano; rostros muy borrosos degradan el HoG.
INF_MAX_YAW_ASYMMETRY      = 0.60   # Asimetría horizontal ojo-nariz (proxy de perfil extremo).

# ── Umbrales modo ENTRENAMIENTO (más permisivos para maximizar muestras) ────
TRN_MIN_FACE_WIDTH_PX      = 32
TRN_MIN_FACE_HEIGHT_PX     = 32
TRN_MIN_SHARPNESS_VARIANCE = 25.0    # Acepta imágenes con cierto desenfoque.
TRN_MAX_YAW_ASYMMETRY      = 0.60   # Tolera hasta ~45° de perfil lateral.


@dataclass
class FaceQualityReport:
    is_valid: bool
    reasons: list[str]
    sharpness: float
    size_ok: bool
    pose_ok: bool


def validate_face_quality(roi: np.ndarray,
                           face_result: FaceDetectionResult,
                           training_mode: bool = False) -> FaceQualityReport:
    """Aplica una batería de chequeos de calidad sobre el rostro detectado.

    Args:
        roi: Imagen BGR de la que se extrajo la detección facial.
        face_result: Resultado de YuNetFaceDetector.detect() o detect_training().
        training_mode: Si True, usa umbrales relajados para maximizar muestras
                       del dataset (menor nitidez mínima, menor tamaño mínimo,
                       mayor tolerancia de ángulo). No afecta la inferencia RT.
    """
    # Selección de umbrales según modo
    min_w      = TRN_MIN_FACE_WIDTH_PX      if training_mode else INF_MIN_FACE_WIDTH_PX
    min_h      = TRN_MIN_FACE_HEIGHT_PX     if training_mode else INF_MIN_FACE_HEIGHT_PX
    min_sharp  = TRN_MIN_SHARPNESS_VARIANCE if training_mode else INF_MIN_SHARPNESS_VARIANCE
    max_yaw    = TRN_MAX_YAW_ASYMMETRY      if training_mode else INF_MAX_YAW_ASYMMETRY

    reasons = []

    if not face_result.detected or face_result.bbox is None:
        return FaceQualityReport(is_valid=False, reasons=["sin_rostro_detectado"],
                                  sharpness=0.0, size_ok=False, pose_ok=False)

    if not all(np.isfinite(v) for v in face_result.bbox):
        return FaceQualityReport(is_valid=False, reasons=["bbox_invalido_nan_inf"],
                                  sharpness=0.0, size_ok=False, pose_ok=False)

    x, y, w, h = [int(v) for v in face_result.bbox]
    roi_h, roi_w = roi.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(roi_w, x + w), min(roi_h, y + h)

    size_ok = (x2 - x) >= min_w and (y2 - y) >= min_h
    if not size_ok:
        reasons.append(
            f"rostro_demasiado_pequeno ({x2 - x}x{y2 - y}px, min={min_w}x{min_h})"
        )

    face_crop = roi[y:y2, x:x2] if (x2 > x and y2 > y) else np.zeros((1, 1), dtype=np.uint8)
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
    
    brightness = float(np.mean(gray)) if gray.size > 0 else 0.0
    if brightness <= 35.0:
        reasons.append(f"rostro_muy_oscuro (brillo={brightness:.1f}, min=35.0)")

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size > 1 else 0.0
    if sharpness < min_sharp:
        reasons.append(f"rostro_borroso (var={sharpness:.1f}, min={min_sharp})")

    pose_ok = _check_pose_symmetry(face_result, max_yaw_ratio=max_yaw)
    if not pose_ok:
        reasons.append("pose_muy_lateral_o_landmarks_incoherentes")

    is_valid = size_ok and (brightness > 35.0) and (sharpness >= min_sharp) and pose_ok
    return FaceQualityReport(is_valid=is_valid, reasons=reasons, sharpness=sharpness,
                              size_ok=size_ok, pose_ok=pose_ok)


def _check_pose_symmetry(face_result: FaceDetectionResult,
                          max_yaw_ratio: float = INF_MAX_YAW_ASYMMETRY) -> bool:
    """Descarta rostros en perfil extremo usando la simetría horizontal ojo-nariz."""
    if face_result.landmarks is None:
        return True  # sin landmarks no se puede evaluar; no se penaliza injustamente

    left_eye, right_eye, nose, _, _ = face_result.landmarks
    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_span = abs(right_eye[0] - left_eye[0]) or 1.0

    nose_offset_ratio = abs(nose[0] - eye_center_x) / eye_span
    return nose_offset_ratio <= max_yaw_ratio