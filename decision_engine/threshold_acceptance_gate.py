"""Aplica el umbral critico T_aceptacion en [0.5, 1.0] para aceptar o rechazar la clase ganadora [REQ-DEC-04]."""


class ThresholdAcceptanceGate:
    """Compuerta de aceptacion basada en el umbral critico T_aceptacion.

    Con la nueva metrica margin_gap + sigmoid escalado (scale=4.0, calibrado sobre el dataset):

    Equivalencias threshold -> gap minimo requerido (con scale=4.0):
      threshold=0.55 -> gap > 0.05  (muy laxo)
      threshold=0.65 -> gap > 0.15  (moderado, DEFAULT — acepta la mayoria de rostros conocidos)
      threshold=0.75 -> gap > 0.27  (estricto — solo acepta rostros con ventaja clara)
      threshold=0.85 -> gap > 0.49  (muy estricto)

    Por que 0.65:
      El gap promedio de rostros conocidos del dataset es 0.27 (prob~0.75).
      Un threshold de 0.65 acepta gaps >= 0.15, cubriendo ~80% de los rostros
      del dataset y rechazando los casos ambiguos (gap < 0.15).
    """

    def __init__(self, t_aceptacion: float = 0.65):
        if not (0.5 <= t_aceptacion <= 1.0):
            raise ValueError("T_aceptacion debe estar en el rango [0.5, 1.0] segun [REQ-DEC-04].")
        self.t_aceptacion = t_aceptacion

    def accept(self, probability: float) -> bool:
        """Devuelve True si la probabilidad supera o iguala el umbral critico."""
        return probability >= self.t_aceptacion

    def update_threshold(self, new_threshold: float) -> None:
        if not (0.5 <= new_threshold <= 1.0):
            raise ValueError("El nuevo umbral debe estar en el rango [0.5, 1.0].")
        self.t_aceptacion = new_threshold