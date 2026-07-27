"""Composición secuencial: normalizador facial -> HoG -> SVM facial (rama de identificación)."""
import numpy as np
from feature_extraction.face.yunet_face_detector import YuNetFaceDetector
from feature_extraction.face.face_normalizer import normalize_face
from feature_extraction.face.hog_extractor import extract_hog_features
from classification.svm_facial_model import SVMFacialModel
from classification.logistic_confidence_converter import decision_margin_to_probability
from feature_extraction.face.face_quality_filter import check_face_quality

_face_detector: YuNetFaceDetector | None = None
_svm_facial: SVMFacialModel | None = None


def _get_face_detector() -> YuNetFaceDetector:
    global _face_detector
    if _face_detector is None:
        _face_detector = YuNetFaceDetector()
    return _face_detector


def _get_svm_facial() -> SVMFacialModel:
    global _svm_facial
    if _svm_facial is None:
        _svm_facial = SVMFacialModel()
        _svm_facial.load()
    return _svm_facial


def run_id_branch(body_roi: np.ndarray) -> tuple[str, float]:
    """Ejecuta: detección facial -> normalización -> HoG -> SVM Facial -> probabilidad [REQ-FAC-01..05]."""
    face_result = _get_face_detector().detect(body_roi)
    face_gray = normalize_face(body_roi, face_result)

    quality = check_face_quality(body_roi, face_result, strict=False)
    if not quality.passed:
        return "Desconocido", 0.0

    if face_gray is None:
        return "Desconocido", 0.0

    hog_vector = extract_hog_features(face_gray)
    svm = _get_svm_facial()

    identity, margin = svm.predict(hog_vector)
    probability = decision_margin_to_probability(margin)
    return identity, probability