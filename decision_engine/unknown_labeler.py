"""Etiqueta obligatoriamente como 'Desconocido' cuando la probabilidad máxima cae por debajo del umbral [REQ-DEC-05]."""

UNKNOWN_LABEL = "Desconocido"


def label_unknown() -> str:
    """Devuelve la etiqueta estándar de sujeto no reconocido."""
    return UNKNOWN_LABEL


def is_unknown(identity: str) -> bool:
    return identity == UNKNOWN_LABEL