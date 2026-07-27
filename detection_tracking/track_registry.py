"""Estructura de datos que mantiene el estado de cada Track-ID activo."""
from dataclasses import dataclass, field
from typing import Dict, List
import time


@dataclass
class TrackState:
    track_id: int
    last_identity: str = "Desconocido"
    vote_history: List[tuple] = field(default_factory=list)  # (identity, confidence, timestamp)
    last_update_frame: int = -1
    last_update_time: float = field(default_factory=time.time)


class TrackRegistry:
    """Registro central de estado por Track-ID activo."""

    def __init__(self):
        self._tracks: Dict[int, TrackState] = {}

    def update(self, track_id: int, identity: str, confidence: float, frame_index: int) -> None:
        state = self._tracks.setdefault(track_id, TrackState(track_id=track_id))
        state.last_identity = identity
        state.vote_history.append((identity, confidence, time.time()))
        state.last_update_frame = frame_index
        state.last_update_time = time.time()

    def get(self, track_id: int) -> TrackState | None:
        return self._tracks.get(track_id)

    def get_all(self) -> Dict[int, TrackState]:
        return self._tracks

    def purge_stale(self, current_frame: int, max_frame_gap: int = 150) -> None:
        """Elimina tracks que no se actualizan hace más de max_frame_gap fotogramas."""
        stale_ids = [
            tid for tid, st in self._tracks.items()
            if current_frame - st.last_update_frame > max_frame_gap
        ]
        for tid in stale_ids:
            del self._tracks[tid]