"""Composicion secuencial: normalizador facial -> HOG+LBP -> SVM facial (rama de identificacion)."""
import numpy as np
from feature_extraction.face.yunet_face_detector import YuNetFaceDetector, FaceDetectionResult
from feature_extraction.face.face_normalizer import normalize_face
from feature_extraction.face.hog_extractor import extract_hog_features
from classification.svm_facial_model import SVMFacialModel
from feature_extraction.face.face_quality_validator import validate_face_quality

import threading

_face_detector: YuNetFaceDetector | None = None
_svm_facial: SVMFacialModel | None = None
_id_lock = threading.Lock()

def _get_face_detector() -> YuNetFaceDetector:
    global _face_detector
    if _face_detector is None:
        _face_detector = YuNetFaceDetector()
    return _face_detector


def _get_svm_facial() -> SVMFacialModel:
    global _svm_facial
    if _svm_facial is None:
        with _id_lock:
            if _svm_facial is None:
                temp_model = SVMFacialModel()
                temp_model.load()
                _svm_facial = temp_model
    return _svm_facial


def run_id_branch(body_roi: np.ndarray,
                  face_result: FaceDetectionResult | None = None) -> tuple[str, float]:
    """Ejecuta: deteccion facial -> normalizacion+alineacion -> HOG+LBP -> SVM -> probabilidad.

    Pipeline de features identico al entrenamiento:
      normalize_face() [crop + align + CLAHE + unsharp] -> extract_combined_features() [HOG+LBP]

    Args:
        body_roi:    Recorte BGR del cuerpo del track actual.
        face_result: Resultado de YuNet pre-calculado por face_visibility_router (evita
                     segunda llamada redundante). Si es None, se ejecuta internamente.
    """
    if face_result is None:
        face_result = _get_face_detector().detect(body_roi)

    quality = validate_face_quality(body_roi, face_result)
    if not quality.is_valid:
        return "Desconocido", 0.0

    # normalize_face aplica: crop -> align_landmarks -> CLAHE -> unsharp -> resize 64x64
    face_gray, landmarks = normalize_face(body_roi, face_result)
    if face_gray is None:
        return "Desconocido", 0.0

    # Extraer descriptor HOG (Piramidal + Componentes + Opponent)
    feature_vector = extract_hog_features(face_gray, landmarks)
    svm = _get_svm_facial()

    identity, probability = svm.predict(feature_vector)
    return identity, probability