"""Evaluación geométrica de visibilidad de cabeza para conmutar entre rama ID (HOG) o Re-ID (LBP)."""
from typing import Literal
import numpy as np
from feature_extraction.face.face_normalizer import normalize_face
from feature_extraction.face.face_quality_validator import validate_face_quality

Branch = Literal["ID", "REID"]


def route_branch(body_roi: np.ndarray) -> tuple[Branch, np.ndarray | None]:
    """Determina si utilizar la rama ID (HOG en cabeza) o Re-ID (LBP en torso) según la calidad de la región de la cabeza.

    Returns:
        branch:    "ID" o "REID"
        head_crop: Imagen 96x96 BGR de la cabeza si es válida, o None.
    """
    head_crop = normalize_face(body_roi)
    if head_crop is not None:
        q_report = validate_face_quality(head_crop, training_mode=False)
        if q_report.is_valid:
            return "ID", head_crop

    return "REID", None