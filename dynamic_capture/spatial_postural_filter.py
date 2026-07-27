"""Exige variación significativa de desplazamiento o escala respecto a la última captura [REQ-CAP-05]."""
from typing import Dict
import math

DEFAULT_MIN_DISPLACEMENT_RATIO = 0.10  # 10% del tamaño de la ROI
DEFAULT_MIN_SCALE_CHANGE_RATIO = 0.10


class SpatialPosturalFilter:
    def __init__(self, min_displacement_ratio: float = DEFAULT_MIN_DISPLACEMENT_RATIO,
                 min_scale_change_ratio: float = DEFAULT_MIN_SCALE_CHANGE_RATIO):
        self.min_displacement_ratio = min_displacement_ratio
        self.min_scale_change_ratio = min_scale_change_ratio
        self._last_bbox: Dict[int, tuple] = {}

    def passes(self, track_id: int, bbox: tuple) -> bool:
        last_bbox = self._last_bbox.get(track_id)
        if last_bbox is None:
            return True

        x1, y1, x2, y2 = bbox
        lx1, ly1, lx2, ly2 = last_bbox

        w, h = (x2 - x1), (y2 - y1)
        lw, lh = (lx2 - lx1), (ly2 - ly1)

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2

        displacement = math.hypot(cx - lcx, cy - lcy)
        diagonal = math.hypot(w, h) or 1.0
        displacement_ratio = displacement / diagonal

        scale_change_ratio = abs((w * h) - (lw * lh)) / max((lw * lh), 1.0)

        return (displacement_ratio >= self.min_displacement_ratio) or \
               (scale_change_ratio >= self.min_scale_change_ratio)

    def register_capture(self, track_id: int, bbox: tuple) -> None:
        self._last_bbox[track_id] = bbox

    def reset_track(self, track_id: int) -> None:
        self._last_bbox.pop(track_id, None)