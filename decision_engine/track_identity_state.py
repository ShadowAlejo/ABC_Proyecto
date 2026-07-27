"""Mantiene la identidad confirmada previa de un Track-ID cuando el sujeto deja de mostrar el rostro,
usando Re-ID solo como validación de continuidad [REQ-DEC-02]."""
from dataclasses import dataclass, field
from typing import Dict
from decision_engine.unknown_labeler import is_unknown, label_unknown


@dataclass
class _IdentityMemory:
    confirmed_identity: str = ""
    confirmed_via_face: bool = False


class TrackIdentityState:
    """Persiste identidad confirmada por rostro y usa Re-ID únicamente para sostener el seguimiento."""

    def __init__(self):
        self._memory: Dict[int, _IdentityMemory] = {}

    def resolve(self, track_id: int, branch: str, candidate_identity: str, accepted: bool) -> str:
        """
        Devuelve la identidad final aplicando la lógica de persistencia:
        - Si la rama es ID y se acepta: se confirma y memoriza la nueva identidad.
        - Si la rama es REID: se usa solo para validar continuidad; si hay identidad
          previa confirmada por rostro, se mantiene esa identidad en lugar de sobrescribirla.
        """
        memory = self._memory.setdefault(track_id, _IdentityMemory())

        if branch == "ID":
            if accepted and not is_unknown(candidate_identity):
                memory.confirmed_identity = candidate_identity
                memory.confirmed_via_face = True
                return candidate_identity
            return candidate_identity if not memory.confirmed_via_face else memory.confirmed_identity

        # Rama REID: mantiene identidad confirmada previa por rostro si existe.
        if memory.confirmed_via_face and memory.confirmed_identity:
            return memory.confirmed_identity

        return candidate_identity if accepted else label_unknown()

    def reset_track(self, track_id: int) -> None:
        self._memory.pop(track_id, None)