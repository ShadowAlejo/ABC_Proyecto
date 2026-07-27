"""Restringe la ROI a cabeza, cuello y hombros, excluyendo/reduciendo prendas inferiores con gradiente suave [REQ-RID-03]."""
import numpy as np

MIN_LOWER_WEIGHT = 0.15  # Peso residual asignado a la zona inferior (pies/pantalón)
MAX_TOP_WEIGHT = 1.0     # Peso máximo para la zona superior (cabeza/hombros/torso)


def apply_stable_zone_mask(body_roi_64x128: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (roi_completa, mapa_de_pesos) donde el mapa de pesos privilegia
    cabeza/cuello/hombros mediante una función sigmoidea suave que decae de 1.0 a 0.15.
    """
    if body_roi_64x128 is None or body_roi_64x128.shape[:2] != (128, 64):
        raise ValueError("Se requiere una ROI corporal canónica de 64x128 (alto x ancho = 128x64).")

    h, w = body_roi_64x128.shape[:2]

    # Gradiente sigmoideo vertical suave: transición centrada cerca del 45% de la altura
    y_coords = np.linspace(0, 1, h, dtype=np.float32)
    # k=10 controla la pendiente de la transición suave alrededor de y=0.45
    sigmoid = 1.0 / (1.0 + np.exp(10.0 * (y_coords - 0.45)))

    # Mapear rango sigmoideo [0, 1] a [MIN_LOWER_WEIGHT, MAX_TOP_WEIGHT]
    weights_1d = MIN_LOWER_WEIGHT + (MAX_TOP_WEIGHT - MIN_LOWER_WEIGHT) * sigmoid

    # Replicar horizontalmente para cubrir todo el ancho w=64
    weight_map = np.tile(weights_1d[:, np.newaxis], (1, w)).astype(np.float32)

    return body_roi_64x128, weight_map