"""Funcion de decision pura: evalua la deteccion y calidad facial para conmutar a rama ID o Re-ID."""
from typing import Literal
import numpy as np
from feature_extraction.face.yoloface_detector import YoloFaceDetector, FaceDetectionResult
from feature_extraction.face.face_quality_validator import validate_face_quality
from utils.logger import get_logger
from utils.config_loader import ConfigLoader

logger = get_logger(__name__)

Branch = Literal["ID", "REID", "UNKNOWN"]

# Umbral mínimo de confianza para detección facial en inferencia
MIN_FACE_ROUTING_CONF = 0.50

_face_detector: YoloFaceDetector | None = None


def is_full_body_available(body_roi: np.ndarray, face_result: FaceDetectionResult | None = None) -> bool:
    """Verifica si el recorte contiene una silueta corporal completa apta para Re-ID (Body Completeness Gate).
    
    Descarta planos medios, primeros planos y bustos de cámara web que distorsionarían el descriptor LBP.
    """
    if body_roi is None or body_roi.size == 0:
        return False

    h, w = body_roi.shape[:2]
    aspect_ratio = h / max(1, w)

    # Una persona de cuerpo entero tiene relación de aspecto vertical h/w >= 1.35
    if aspect_ratio < 1.35:
        return False

    # Si hay rostro detectado, verificar que no ocupe más del 30% del área del cuerpo
    if face_result is not None and face_result.bbox is not None:
        _, _, fw, fh = face_result.bbox
        if (fw * fh) > 0.30 * (w * h):
            return False

    return True


def _get_face_detector() -> YoloFaceDetector:
    global _face_detector
    if _face_detector is None:
        model_path = ConfigLoader.get("face.yoloface_model_path", "yolov8n-face.pt")
        conf_threshold = ConfigLoader.get("face.conf_threshold", 0.50)
        _face_detector = YoloFaceDetector(
            model_path=model_path,
            conf_threshold=conf_threshold
        )
    return _face_detector


def route_branch(body_roi: np.ndarray,
                 conf_threshold: float = MIN_FACE_ROUTING_CONF,
                 is_enhanced: bool = False) -> tuple[Branch, float, FaceDetectionResult]:
    """
    Determina la rama activa (ID, REID o UNKNOWN) según la visibilidad, calidad facial y cobertura corporal [REQ-RID-01].
    """
    detector = _get_face_detector()
    result = detector.detect(body_roi, is_enhanced=is_enhanced)

    return route_branch_with_result(body_roi, result, conf_threshold)


def route_branch_with_result(body_roi: np.ndarray,
                             result: FaceDetectionResult,
                             conf_threshold: float = MIN_FACE_ROUTING_CONF) -> tuple[Branch, float, FaceDetectionResult]:
    """
    Toma un resultado de detección facial precalculado y evalúa la compuerta de conmutación.
    - Si rostro válido -> ID.
    - Si no hay rostro pero cuerpo completo disponible -> REID.
    - Si plano corto sin rostro -> UNKNOWN (evita Re-ID erróneo en busto).
    """
    if result.detected and result.confidence >= conf_threshold:
        q_report = validate_face_quality(body_roi, result, training_mode=False)
        if q_report.is_valid:
            return "ID", result.confidence, result

    # Body Completeness Gate: Solo permitir Re-ID si hay cuerpo completo
    if is_full_body_available(body_roi, result):
        return "REID", result.confidence, result

    return "UNKNOWN", 0.0, result