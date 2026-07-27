"""Valida la calidad de un rostro detectado antes de usarlo para entrenamiento o inferencia.
Reduce falsos positivos 'técnicamente válidos mas' inútiles (borrosos, diminutos, muy rotados)."""
from dataclasses import dataclass
import cv2
import numpy as np
from feature_extraction.face.yunet_face_detector import FaceDetectionResult

MIN_FACE_WIDTH_PX = 30
MIN_FACE_HEIGHT_PX = 30
MIN_SHARPNESS_VARIANCE = 25.0     # Laplaciano; rostros muy borrosos degradan el HoG.
MAX_YAW_ASYMMETRY_RATIO = 0.6     # Asimetría horizontal ojo-nariz-boca (proxy de perfil extremo)


@dataclass
class FaceQualityReport:
    is_valid: bool
    reasons: list[str]
    sharpness: float
    size_ok: bool
    pose_ok: bool


def validate_face_quality(roi: np.ndarray, face_result: FaceDetectionResult) -> FaceQualityReport:
    """Aplica una batería de chequeos de calidad sobre el rostro detectado."""
    reasons = []

    if not face_result.detected or face_result.bbox is None:
        return FaceQualityReport(is_valid=False, reasons=["sin_rostro_detectado"],
                                  sharpness=0.0, size_ok=False, pose_ok=False)

    x, y, w, h = [int(v) for v in face_result.bbox]
    roi_h, roi_w = roi.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(roi_w, x + w), min(roi_h, y + h)

    size_ok = (x2 - x) >= MIN_FACE_WIDTH_PX and (y2 - y) >= MIN_FACE_HEIGHT_PX
    if not size_ok:
        reasons.append(f"rostro_demasiado_pequeno ({x2 - x}x{y2 - y}px)")

    face_crop = roi[y:y2, x:x2] if (x2 > x and y2 > y) else np.zeros((1, 1), dtype=np.uint8)
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size > 1 else 0.0
    if sharpness < MIN_SHARPNESS_VARIANCE:
        reasons.append(f"rostro_borroso (var={sharpness:.1f})")

    pose_ok = _check_pose_symmetry(face_result)
    if not pose_ok:
        reasons.append("pose_muy_lateral_o_landmarks_incoherentes")

    is_valid = size_ok and (sharpness >= MIN_SHARPNESS_VARIANCE) and pose_ok
    return FaceQualityReport(is_valid=is_valid, reasons=reasons, sharpness=sharpness,
                              size_ok=size_ok, pose_ok=pose_ok)


def _check_pose_symmetry(face_result: FaceDetectionResult) -> bool:
    """Descarta rostros en perfil extremo usando la simetría horizontal ojo-nariz."""
    if face_result.landmarks is None:
        return True  # sin landmarks no se puede evaluar; no se penaliza injustamente

    left_eye, right_eye, nose, _, _ = face_result.landmarks
    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_span = abs(right_eye[0] - left_eye[0]) or 1.0

    nose_offset_ratio = abs(nose[0] - eye_center_x) / eye_span
    return nose_offset_ratio <= MAX_YAW_ASYMMETRY_RATIO