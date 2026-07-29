"""Funcion de decision pura: evalua la confianza de YuNet y determina si conmutar a rama ID o Re-ID."""
from typing import Literal
import numpy as np
from feature_extraction.face.yunet_face_detector import YuNetFaceDetector, FaceDetectionResult, DEFAULT_FACE_CONF_THRESHOLD

Branch = Literal["ID", "REID"]

# Umbral minimo permisivo para intentar la rama facial (evita conmutar a Re-ID prematuramente)
MIN_FACE_ROUTING_CONF = 0.40

_face_detector: YuNetFaceDetector | None = None


def _get_face_detector() -> YuNetFaceDetector:
    global _face_detector
    if _face_detector is None:
        _face_detector = YuNetFaceDetector()
    return _face_detector


def route_branch(body_roi: np.ndarray,
                 conf_threshold: float = MIN_FACE_ROUTING_CONF) -> tuple[Branch, float, FaceDetectionResult]:
    """
    Determina la rama activa (ID o REID) segun la visibilidad facial [REQ-RID-01].
    Retorna tambien el FaceDetectionResult para que id_branch_pipeline pueda reutilizarlo
    sin tener que llamar a YuNet una segunda vez (eliminando trabajo duplicado).

    Returns:
        branch:      "ID" o "REID"
        face_conf:   confianza de la deteccion facial
        face_result: resultado completo de YuNet (bbox, landmarks, etc.)
    """
    detector = _get_face_detector()
    result = detector.detect(body_roi)

    if result.detected and result.confidence >= conf_threshold:
        return "ID", result.confidence, result
    return "REID", result.confidence, result