"""Convierte el margin_gap del SVM en probabilidad mediante la funcion logistica escalada [REQ-DEC-03].

El margin_gap = max(decision) - second_max(decision) tiene escala diferente al margen individual.
El parametro 'scale' estira la curva sigmoid para que se ajuste al rango de gaps tipicos del modelo
(calibrado empiricamente sobre el dataset de 15 clases con LinearSVC + PCA(512)):

  Calibracion (scale=4.0, gaps de rostros reales: mean=0.27, max=0.89):
    gap = 0.00 (empate total)     -> P = 0.50  (maximo rechazo)
    gap = 0.15 (borde de umbral)  -> P = 0.65  (justo en el limite de aceptacion)
    gap = 0.27 (promedio dataset) -> P = 0.75  (aceptado con confianza)
    gap = 0.50 (confianza buena)  -> P = 0.88  (aceptado con certeza)
    gap = 0.89 (maximo observado) -> P = 0.97  (identidad dominante)
"""
import numpy as np

# Factor de escala calibrado empiricamente para LinearSVC + PCA(512) con 15 clases.
# Con C=0.01, los gaps tipicos de rostros reales son 0.0-0.89 (media 0.27).
# scale=4.0 hace que el promedio (gap=0.27) produzca P~0.75 (por encima del threshold 0.65).
_SIGMOID_SCALE = 4.0


def decision_margin_to_probability(margin_gap: float, scale: float = _SIGMOID_SCALE) -> float:
    """Logistic(margin_gap * scale) = 1 / (1 + exp(-margin_gap * scale)).

    Args:
        margin_gap: Diferencia max(decision) - second_max(decision) del SVM.
        scale:      Factor de escala calibrado al rango del modelo. Default: 4.0.
    """
    clipped = float(np.clip(margin_gap * scale, -500, 500))
    return 1.0 / (1.0 + np.exp(-clipped))