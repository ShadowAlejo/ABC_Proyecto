"""Envoltorio concreto que adapta detecciones a ByteTrack, asignando Track-ID persistente."""
from dataclasses import dataclass
from typing import List
import numpy as np

try:
    from yolox.tracker.byte_tracker import BYTETracker
    _BYTETRACK_AVAILABLE = True
except ImportError:
    _BYTETRACK_AVAILABLE = False

from detection_tracking.yolov8n_detector import Detection
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Track:
    track_id: int
    bbox: tuple
    confidence: float


class _ByteTrackArgs:
    """Argumentos requeridos por BYTETracker."""
    track_thresh = 0.6
    track_buffer = 90
    match_thresh = 0.8
    mot20 = False


class ByteTrackAdapter:
    """Adaptador concreto de ByteTrack [REQ-TRK-01, REQ-TRK-02]."""

    def __init__(self, frame_rate: int = 30):
        if not _BYTETRACK_AVAILABLE:
            raise ImportError(
                "El paquete 'yolox' (ByteTrack) no está instalado. "
                "Instálelo o use DeepSORTAdapter como alternativa."
            )
        self.tracker = BYTETracker(_ByteTrackArgs(), frame_rate=frame_rate)

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        if not detections:
            return []

        dets_array = np.array(
            [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence] for d in detections],
            dtype=np.float32,
        )
        img_h, img_w = frame.shape[:2]
        online_targets = self.tracker.update(dets_array, [img_h, img_w], [img_h, img_w])

        tracks: List[Track] = []
        for t in online_targets:
            x1, y1, w, h = t.tlwh
            tracks.append(Track(track_id=int(t.track_id), bbox=(x1, y1, x1 + w, y1 + h), confidence=float(t.score)))
        return tracks