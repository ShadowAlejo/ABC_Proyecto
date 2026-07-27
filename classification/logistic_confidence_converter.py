"""Convierte el margen de decisión w^T x en probabilidad mediante la función logística [REQ-DEC-03]."""
import numpy as np


def decision_margin_to_probability(margin: float) -> float:
    """Logistic(w^T x) = 1 / (1 + e^(-w^T x))."""
    # Recorte numérico para evitar overflow en np.exp con márgenes extremos.
    clipped_margin = float(np.clip(margin, -500, 500))
    return 1.0 / (1.0 + np.exp(-clipped_margin))