"""Funcion de decision pura: evalua la deteccion y calidad facial para conmutar a rama ID o Re-ID."""
from typing import Literal
import numpy as np
from feature_extraction.face.yoloface_detector import YoloFaceDetector, FaceDetectionResult
from feature_extraction.face.face_quality_validator import validate_face_quality
from utils.logger import get_logger
from utils.config_loader import ConfigLoader

logger = get_logger(__name__)

Branch = Literal["ID", "REID"]

# Umbral mínimo de confianza para detección facial en inferencia
MIN_FACE_ROUTING_CONF = 0.50

_face_detector: YoloFaceDetector | None = None


def _get_face_detector() -> YoloFaceDetector:
    global _face_detector
    if _face_detector is None:
        model_path = ConfigLoader.get("face.yoloface_model_path", "yolov8n-face.pt")
        conf_threshold = ConfigLoader.get("face.conf_threshold", 0.60)
        _face_detector = YoloFaceDetector(
            model_path=model_path,
            conf_threshold=conf_threshold
        )
    return _face_detector


def route_branch(body_roi: np.ndarray,
                 conf_threshold: float = MIN_FACE_ROUTING_CONF) -> tuple[Branch, float, FaceDetectionResult]:
    """
    Determina la rama activa (ID o REID) según la visibilidad y calidad facial [REQ-RID-01].
    - Si se detecta un rostro con calidad válida (no borroso, simetría frontal) -> Rama ID.
    - Si el sujeto está de espaldas, ocluido o sin rostro -> Rama REID.

    Returns:
        branch:      "ID" o "REID"
        face_conf:   confianza de la deteccion facial
        face_result: resultado completo de YuNet
    """
    detector = _get_face_detector()
    result = detector.detect(body_roi)

    if result.detected and result.confidence >= conf_threshold:
        q_report = validate_face_quality(body_roi, result, training_mode=False)
        if q_report.is_valid:
            return "ID", result.confidence, result

    return "REID", result.confidence, result