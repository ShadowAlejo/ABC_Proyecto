"""Valida la calidad de un recorte facial/cabeza (brillo, nitidez y dimensiones) sin YuNet."""
from dataclasses import dataclass
import cv2
import numpy as np

MIN_FACE_WIDTH_PX = 32
MIN_FACE_HEIGHT_PX = 32
MIN_SHARPNESS_VARIANCE = 25.0  # Varianza Laplaciana mínima
MIN_BRIGHTNESS = 35.0          # Brillo medio en grises


@dataclass
class FaceQualityReport:
    is_valid: bool
    reasons: list[str]
    sharpness: float
    brightness: float
    size_ok: bool


def validate_face_quality(roi: np.ndarray, training_mode: bool = False) -> FaceQualityReport:
    """Aplica chequeos de calidad sobre la región facial/cabeza (dimensión, nitidez y brillo)."""
    reasons = []

    if roi is None or roi.size == 0:
        return FaceQualityReport(is_valid=False, reasons=["roi_vacio"],
                                  sharpness=0.0, brightness=0.0, size_ok=False)

    h, w = roi.shape[:2]
    size_ok = (w >= MIN_FACE_WIDTH_PX and h >= MIN_FACE_HEIGHT_PX)
    if not size_ok:
        reasons.append(f"rostro_pequeno ({w}x{h}px, min={MIN_FACE_WIDTH_PX}x{MIN_FACE_HEIGHT_PX})")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

    brightness = float(np.mean(gray)) if gray.size > 0 else 0.0
    if brightness < MIN_BRIGHTNESS:
        reasons.append(f"rostro_oscuro (brillo={brightness:.1f}, min={MIN_BRIGHTNESS})")

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size > 1 else 0.0
    if sharpness < MIN_SHARPNESS_VARIANCE:
        reasons.append(f"rostro_borroso (var={sharpness:.1f}, min={MIN_SHARPNESS_VARIANCE})")

    is_valid = size_ok and (brightness >= MIN_BRIGHTNESS) and (sharpness >= MIN_SHARPNESS_VARIANCE)
    return FaceQualityReport(
        is_valid=is_valid,
        reasons=reasons,
        sharpness=sharpness,
        brightness=brightness,
        size_ok=size_ok,
    )