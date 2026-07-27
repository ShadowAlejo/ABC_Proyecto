"""Composición secuencial: aislador de torso -> LBP-U -> SVM Re-ID (rama de re-identificación)."""
import cv2
import numpy as np
from feature_extraction.body.torso_roi_isolator import isolate_torso_roi
from feature_extraction.body.stable_zone_masker import apply_stable_zone_mask
from feature_extraction.body.spatial_grid_histogram import extract_spatial_grid_lbp
from classification.svm_reid_model import SVMReidModel
from classification.logistic_confidence_converter import decision_margin_to_probability

_svm_reid: SVMReidModel | None = None


def _get_svm_reid() -> SVMReidModel:
    global _svm_reid
    if _svm_reid is None:
        _svm_reid = SVMReidModel()
        _svm_reid.load()
    return _svm_reid


def run_reid_branch(body_roi: np.ndarray) -> tuple[str, float]:
    """Ejecuta: aislamiento de torso -> zona estable -> LBP-U por rejilla -> SVM Re-ID -> probabilidad."""
    torso_roi = isolate_torso_roi(body_roi)
    gray_roi = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2GRAY)

    _, weight_map = apply_stable_zone_mask(torso_roi)
    lbp_vector = extract_spatial_grid_lbp(gray_roi, weight_map=weight_map)

    svm = _get_svm_reid()
    identity, margin = svm.predict(lbp_vector)
    probability = decision_margin_to_probability(margin)
    return identity, probability