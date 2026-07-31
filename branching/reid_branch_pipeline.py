"""Composición secuencial: aislador de torso -> LBP-U -> SVM Re-ID (rama de re-identificación)."""
import cv2
import numpy as np
from feature_extraction.body.torso_roi_isolator import isolate_torso_roi
from feature_extraction.body.stable_zone_masker import apply_stable_zone_mask
from feature_extraction.body.spatial_grid_histogram import extract_spatial_grid_lbp
from classification.svm_reid_model import SVMReidModel
from classification.logistic_confidence_converter import decision_margin_to_probability
from utils.logger import get_logger

logger = get_logger(__name__)

_svm_reid: SVMReidModel | None = None
# Centinela: True cuando se confirmó que el modelo no existe en disco.
# Evita reintentar la carga (y loggear el warning) en cada frame.
_REID_UNAVAILABLE: bool = False


def _get_svm_reid() -> SVMReidModel | None:
    """Carga el modelo SVM Re-ID de forma perezosa.

    Returns:
        SVMReidModel listo para usar, o None si el archivo .pkl no existe todavía.
        En ese caso se emite un warning UNA SOLA VEZ y la rama retorna 'Desconocido'.
    """
    global _svm_reid, _REID_UNAVAILABLE

    if _REID_UNAVAILABLE:
        return None

    if _svm_reid is None:
        try:
            _svm_reid = SVMReidModel()
            _svm_reid.load()
            logger.info("Modelo SVM Re-ID cargado correctamente.")
        except FileNotFoundError as exc:
            _REID_UNAVAILABLE = True
            logger.warning(
                f"[Re-ID] Modelo no encontrado: {exc}. "
                "La rama Re-ID devolverá 'Desconocido' hasta que se entrene el modelo. "
                "Ejecuta: python scripts/train_reid_svm.py"
            )
            return None

    return _svm_reid


def run_reid_branch(body_roi: np.ndarray) -> tuple[str, float]:
    """Ejecuta: aislamiento de torso -> zona estable -> LBP-U por rejilla -> SVM Re-ID -> probabilidad.

    Si el modelo SVM Re-ID no está disponible, retorna ('Desconocido', 0.0) sin crashear.
    """
    svm = _get_svm_reid()
    if svm is None:
        return "Desconocido", 0.0

    torso_roi = isolate_torso_roi(body_roi, enhance=True)
    gray_roi = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2GRAY)

    _, weight_map = apply_stable_zone_mask(torso_roi)
    lbp_vector = extract_spatial_grid_lbp(gray_roi, weight_map=weight_map)

    identity, margin = svm.predict(lbp_vector)
    probability = decision_margin_to_probability(margin)
    return identity, probability