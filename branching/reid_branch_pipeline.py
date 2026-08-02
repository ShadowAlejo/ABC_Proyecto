"""Composición secuencial: aislador corporal -> LBP-U Hellinger -> SVM Re-ID (rama de re-identificación)."""
import cv2
import numpy as np
from feature_extraction.body.body_roi_isolator import isolate_body_roi
from feature_extraction.body.spatial_grid_histogram import extract_spatial_grid_lbp
from classification.svm_reid_model import SVMReidModel
from utils.logger import get_logger

import threading

logger = get_logger(__name__)

_svm_reid: SVMReidModel | None = None
_REID_UNAVAILABLE: bool = False
_reid_lock = threading.Lock()

def _get_svm_reid() -> SVMReidModel | None:
    """Carga el modelo SVM Re-ID de forma perezosa (Thread-Safe)."""
    global _svm_reid, _REID_UNAVAILABLE

    if _REID_UNAVAILABLE:
        return None

    if _svm_reid is None:
        with _reid_lock:
            # Doble comprobación (Double-Checked Locking)
            if _svm_reid is None:
                try:
                    temp_model = SVMReidModel()
                    temp_model.load()
                    _svm_reid = temp_model
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
    """Ejecuta: aislamiento cuerpo -> LBP-U por rejilla -> SVM Re-ID -> probabilidad."""
    svm = _get_svm_reid()
    if svm is None:
        return "Desconocido", 0.0

    body_128x256 = isolate_body_roi(body_roi)
    gray_roi = cv2.cvtColor(body_128x256, cv2.COLOR_BGR2GRAY)

    lbp_vector = extract_spatial_grid_lbp(gray_roi)

    identity, probability = svm.predict(lbp_vector)
    return identity, probability