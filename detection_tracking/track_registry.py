"""Registro volátil para almacenar los estados de identidad en un fotograma aislado."""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrackState:
    track_id: str
    last_identity: str = "Desconocido"


class TrackRegistry:
    """Registro estático y volátil que se reinicia en cada frame."""

    def __init__(self):
        self._tracks: Dict[str, TrackState] = {}

    def update(self, track_id: str, identity: str, confidence: float, frame_index: int) -> None:
        state = self._tracks.setdefault(track_id, TrackState(track_id=track_id))
        state.last_identity = identity

    def get(self, track_id: str) -> TrackState | None:
        return self._tracks.get(track_id)

    def get_all(self) -> Dict[str, TrackState]:
        return self._tracks

    def clear_all(self) -> None:
        """Limpia el registro completamente al inicio de un nuevo fotograma."""
        self._tracks.clear()