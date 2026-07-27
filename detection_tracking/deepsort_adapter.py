"""Envoltorio concreto alternativo que adapta detecciones a DeepSORT (intercambiable por configuración)."""
from dataclasses import dataclass
from typing import List
import numpy as np

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    _DEEPSORT_AVAILABLE = True
except ImportError:
    _DEEPSORT_AVAILABLE = False

from detection_tracking.yolov8n_detector import Detection
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Track:
    track_id: int
    bbox: tuple
    confidence: float


class DeepSORTAdapter:
    """Adaptador concreto de DeepSORT [REQ-TRK-01, REQ-TRK-02]."""

    def __init__(self, max_age: int = 30, n_init: int = 3):
        if not _DEEPSORT_AVAILABLE:
            raise ImportError(
                "El paquete 'deep-sort-realtime' no está instalado. "
                "Instálelo o use ByteTrackAdapter como alternativa."
            )
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        if not detections:
            self.tracker.update_tracks([], frame=frame)
            return []

        raw_dets = [
            ([d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]], d.confidence, "person")
            for d in detections
        ]
        tracks_out = self.tracker.update_tracks(raw_dets, frame=frame)

        tracks: List[Track] = []
        for t in tracks_out:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = t.to_ltrb()
            tracks.append(Track(track_id=int(t.track_id), bbox=(x1, y1, x2, y2), confidence=1.0))
        return tracks