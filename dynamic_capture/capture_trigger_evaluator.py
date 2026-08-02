"""Verifica si la rama facial confirmó identidad con score >= T_aceptación para activar la recolección [REQ-CAP-01]."""
import numpy as np
from dynamic_capture.resolution_min_filter import passes_resolution_filter
from dynamic_capture.laplacian_sharpness_filter import passes_sharpness_filter
from dynamic_capture.temporal_sampling_filter import TemporalSamplingFilter
from dynamic_capture.spatial_postural_filter import SpatialPosturalFilter
from dynamic_capture.capture_writer import CaptureWriter
from decision_engine.unknown_labeler import is_unknown


class CaptureTriggerEvaluator:
    """Compone todos los filtros de captura dinámica y decide si se almacena la imagen."""

    def __init__(self, capture_writer: CaptureWriter | None = None):
        self.capture_writer = capture_writer or CaptureWriter()
        self.temporal_filter = TemporalSamplingFilter()
        self.spatial_filter = SpatialPosturalFilter()

    def maybe_capture(self, track_id: str, roi: np.ndarray, identity: str, frame_index: int,
                       bbox: tuple | None = None) -> bool:
        """Evalúa todos los filtros en cascada y escribe la captura si todos aprueban."""
        if is_unknown(identity):
            return False

        if not self.capture_writer.has_quota_available(identity):
            return False

        if not passes_resolution_filter(roi):
            return False

        if not passes_sharpness_filter(roi):
            return False

        if not self.temporal_filter.passes(track_id, frame_index):
            return False

        if bbox is not None and not self.spatial_filter.passes(track_id, bbox):
            return False

        success = self.capture_writer.write_capture(roi, identity, frame_index)
        if success:
            self.temporal_filter.register_capture(track_id, frame_index)
            if bbox is not None:
                self.spatial_filter.register_capture(track_id, bbox)
        return success