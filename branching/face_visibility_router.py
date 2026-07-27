"""Función de decisión pura: evalúa la confianza de YuNet y determina si conmutar a rama ID o Re-ID."""
from typing import Literal
import numpy as np
from feature_extraction.face.yunet_face_detector import YuNetFaceDetector, DEFAULT_FACE_CONF_THRESHOLD

Branch = Literal["ID", "REID"]

# Umbral mínimo permisivo para intentar la rama facial (evita conmutar a Re-ID prematuramente)
MIN_FACE_ROUTING_CONF = 0.40

_face_detector: YuNetFaceDetector | None = None


def _get_face_detector() -> YuNetFaceDetector:
    global _face_detector
    if _face_detector is None:
        _face_detector = YuNetFaceDetector()
    return _face_detector


def route_branch(body_roi: np.ndarray, conf_threshold: float = MIN_FACE_ROUTING_CONF) -> tuple[Branch, float]:
    """
    Determina la rama activa (ID o REID) según la visibilidad facial [REQ-RID-01].
    Si hay una detección facial con confianza >= 0.40, utiliza la rama ID (dejando que el
    motor de decisión y el gatekeeper validen la certeza).
    """
    detector = _get_face_detector()
    result = detector.detect(body_roi)

    if result.detected and result.confidence >= conf_threshold:
        return "ID", result.confidence
    return "REID", result.confidence