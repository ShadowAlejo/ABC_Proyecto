"""Función de composición central: encadena detección, ruteo ID/Re-ID,
motor de decisión y captura dinámica. Opera de forma Stateless (sin tracker)."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import concurrent.futures

from detection_tracking.yolov8n_detector import YOLOv8nDetector, Detection
from detection_tracking.track_registry import TrackRegistry
from branching.face_visibility_router import route_branch
from branching.id_branch_pipeline import run_id_branch
from branching.reid_branch_pipeline import run_reid_branch
from decision_engine.threshold_acceptance_gate import ThresholdAcceptanceGate
from decision_engine.unknown_labeler import label_unknown
from dynamic_capture.capture_trigger_evaluator import CaptureTriggerEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EphemeralTrack:
    track_id: str
    bbox: tuple


@dataclass
class TrackResult:
    track_id: str
    bbox: tuple
    identity: str
    confidence: float
    branch_used: str
    captured: bool = False


@dataclass
class PipelineOrchestrator:
    """Orquesta el flujo completo por fotograma sobre las detecciones crudas."""

    detector: YOLOv8nDetector
    registry: TrackRegistry = field(default_factory=TrackRegistry)
    gate: ThresholdAcceptanceGate = field(default_factory=ThresholdAcceptanceGate)
    capture_evaluator: CaptureTriggerEvaluator = field(default_factory=CaptureTriggerEvaluator)
    executor: concurrent.futures.ThreadPoolExecutor = field(init=False)

    def __post_init__(self):
        # i9-14900HX tiene 24 cores (8P+16E), 32 hilos. 16 workers es un buen balance para inferencia concurrente.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> List[TrackResult]:
        # 1. Detección cruda
        detections: List[Detection] = self.detector.detect(frame)
        
        # 2. Asignación de IDs efímeros
        ephemeral_tracks = [
            EphemeralTrack(track_id=f"F{frame_index}_D{i}", bbox=det.bbox)
            for i, det in enumerate(detections)
        ]

        # Limpiar registro en cada frame (Stateless)
        self.registry.clear_all()

        def _process_track(track: EphemeralTrack):
            roi = self._crop_roi(frame, track.bbox)
            if roi is None or roi.size == 0:
                return None

            branch, face_conf, face_result = route_branch(roi)

            if branch == "ID":
                identity, raw_confidence = run_id_branch(roi, face_result=face_result)
                # Fallback a Re-ID
                if raw_confidence < self.gate.t_aceptacion:
                    reid_identity, reid_confidence = run_reid_branch(roi)
                    if reid_confidence > raw_confidence and reid_identity != "Desconocido":
                        identity = reid_identity
                        raw_confidence = reid_confidence
                        branch = "REID"
            else:
                identity, raw_confidence = run_reid_branch(roi)

            # Clasificación instantánea sin inercia
            accepted = self.gate.accept(raw_confidence)
            final_identity = identity if accepted else label_unknown()

            self.registry.update(track.track_id, final_identity, raw_confidence, frame_index)

            captured = False
            if branch == "ID" and accepted:
                captured = self.capture_evaluator.maybe_capture(
                    track_id=track.track_id,
                    roi=roi,
                    identity=final_identity,
                    frame_index=frame_index,
                )

            return TrackResult(
                track_id=track.track_id,
                bbox=track.bbox,
                identity=final_identity,
                confidence=raw_confidence,
                branch_used=branch,
                captured=captured,
            )

        raw_results: List[TrackResult] = []
        for res in self.executor.map(_process_track, ephemeral_tracks):
            if res is not None:
                raw_results.append(res)
                
        # --- EXCLUSION MUTUA ---
        identity_to_track: Dict[str, TrackResult] = {}
        for res in raw_results:
            if res.identity == "Desconocido":
                continue
                
            if res.identity not in identity_to_track:
                identity_to_track[res.identity] = res
            else:
                existing = identity_to_track[res.identity]
                if res.confidence > existing.confidence:
                    existing.identity = label_unknown()
                    identity_to_track[res.identity] = res
                else:
                    res.identity = label_unknown()

        return raw_results

    @staticmethod
    def _crop_roi(frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()