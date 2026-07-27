"""Aplica el umbral crítico T_aceptación en [0.5, 1.0] para aceptar o rechazar la clase ganadora [REQ-DEC-04]."""


class ThresholdAcceptanceGate:
    """Compuerta de aceptación basada en el umbral crítico T_aceptación."""

    def __init__(self, t_aceptacion: float = 0.65):
        if not (0.5 <= t_aceptacion <= 1.0):
            raise ValueError("T_aceptacion debe estar en el rango [0.5, 1.0] según [REQ-DEC-04].")
        self.t_aceptacion = t_aceptacion

    def accept(self, probability: float) -> bool:
        """Devuelve True si la probabilidad máxima supera o iguala el umbral crítico."""
        return probability >= self.t_aceptacion

    def update_threshold(self, new_threshold: float) -> None:
        if not (0.5 <= new_threshold <= 1.0):
            raise ValueError("El nuevo umbral debe estar en el rango [0.5, 1.0].")
        self.t_aceptacion = new_threshold