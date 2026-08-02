"""Garantiza un intervalo mínimo de N fotogramas entre capturas consecutivas del mismo Track-ID [REQ-CAP-04]."""
from typing import Dict

DEFAULT_MIN_FRAME_INTERVAL = 5


class TemporalSamplingFilter:
    def __init__(self, min_frame_interval: int = DEFAULT_MIN_FRAME_INTERVAL):
        self.min_frame_interval = min_frame_interval
        self._last_capture_frame: Dict[int, int] = {}

    def passes(self, track_id: int, current_frame: int) -> bool:
        last_frame = self._last_capture_frame.get(track_id, -self.min_frame_interval - 1)
        return (current_frame - last_frame) >= self.min_frame_interval

    def register_capture(self, track_id: int, current_frame: int) -> None:
        self._last_capture_frame[track_id] = current_frame

    def reset_track(self, track_id: int) -> None:
        self._last_capture_frame.pop(track_id, None)