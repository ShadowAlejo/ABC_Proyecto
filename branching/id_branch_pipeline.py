"""Composición secuencial: normalizador de cabeza (96x96 BGR) -> HOG -> SVM facial (rama de identificación)."""
import numpy as np
import threading
from feature_extraction.face.face_normalizer import normalize_face
from feature_extraction.face.hog_extractor import extract_hog_features
from classification.svm_facial_model import SVMFacialModel
from feature_extraction.face.face_quality_validator import validate_face_quality

_svm_facial: SVMFacialModel | None = None
_id_lock = threading.Lock()


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
                  head_crop: np.ndarray | None = None) -> tuple[str, float]:
    """Ejecuta: normalización de cabeza (96x96) -> HOG -> SVM Facial -> (identidad, probabilidad).

    Args:
        body_roi:  Recorte BGR de la persona detectada.
        head_crop: Imagen 96x96 BGR precalculada por face_visibility_router.
    """
    if head_crop is None:
        head_crop = normalize_face(body_roi)

    if head_crop is None:
        return "Desconocido", 0.0

    quality = validate_face_quality(head_crop, training_mode=False)
    if not quality.is_valid:
        return "Desconocido", 0.0

    # Extraer descriptor 100% HOG (Global + Superior + Inferior)
    feature_vector = extract_hog_features(head_crop)
    svm = _get_svm_facial()

    identity, probability = svm.predict(feature_vector)
    return identity, probability