"""Controla el ritmo de procesamiento y aplica submuestreo temporal si el hardware no alcanza el framerate objetivo."""
import time
from dataclasses import dataclass
from typing import Callable, Optional
from core.video_stream_reader import FrameData
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SchedulerStats:
    processed: int = 0
    skipped: int = 0


class FrameScheduler:
    """Despacha cada frame válido al orquestador, submuestreando si es necesario."""

    def __init__(self, target_fps: float = 15.0, adaptive_skip: bool = True):
        self.target_fps = max(target_fps, 1e-3)
        self.min_interval = 1.0 / self.target_fps
        self.adaptive_skip = adaptive_skip
        self._last_dispatch_time: Optional[float] = None
        self.stats = SchedulerStats()

    def should_dispatch(self) -> bool:
        if not self.adaptive_skip:
            return True
        now = time.perf_counter()
        if self._last_dispatch_time is None:
            self._last_dispatch_time = now
            return True
        elapsed = now - self._last_dispatch_time
        if elapsed >= self.min_interval:
            self._last_dispatch_time = now
            return True
        return False

    def dispatch(self, frame_data: FrameData, callback: Callable[[FrameData], None]) -> bool:
        """Aplica submuestreo temporal y despacha el frame al callback (orquestador) si corresponde."""
        if self.should_dispatch():
            callback(frame_data)
            self.stats.processed += 1
            return True
        self.stats.skipped += 1
        return False