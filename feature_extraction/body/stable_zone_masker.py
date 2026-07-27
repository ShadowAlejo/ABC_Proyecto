"""Restringe la ROI a cabeza, cuello y hombros, excluyendo/reduciendo prendas inferiores [REQ-RID-03]."""
import numpy as np

# Proporciones sobre la ROI corporal 64x128 (ancho x alto): región superior estable.
STABLE_ZONE_HEIGHT_RATIO = 0.45  # Top ~45% de la altura: cabeza, cuello, hombros, torso superior.
LOWER_ZONE_WEIGHT = 0.15         # Peso residual asignado a la zona inferior (no se descarta del todo).


def apply_stable_zone_mask(body_roi_64x128: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (roi_completa, mapa_de_pesos) donde el mapa de pesos privilegia
    cabeza/cuello/hombros y reduce el peso de prendas inferiores.
    """
    if body_roi_64x128 is None or body_roi_64x128.shape[:2] != (128, 64):
        raise ValueError("Se requiere una ROI corporal canónica de 64x128 (alto x ancho = 128x64).")

    h, w = body_roi_64x128.shape[:2]
    stable_rows = int(h * STABLE_ZONE_HEIGHT_RATIO)

    weight_map = np.full((h, w), LOWER_ZONE_WEIGHT, dtype=np.float32)
    weight_map[:stable_rows, :] = 1.0

    return body_roi_64x128, weight_map