"""Función de composición central: encadena detección, tracking, ruteo ID/Re-ID,
motor de decisión y captura dinámica. No hereda de clases base; compone funciones concretas."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import concurrent.futures

from detection_tracking.yolov8n_detector import YOLOv8nDetector, Detection
from detection_tracking.bytetrack_adapter import ByteTrackAdapter
from detection_tracking.deepsort_adapter import DeepSORTAdapter
from detection_tracking.track_registry import TrackRegistry
from branching.face_visibility_router import route_branch
from branching.id_branch_pipeline import run_id_branch
from branching.reid_branch_pipeline import run_reid_branch
from decision_engine.weighted_voting_inertia import WeightedVotingInertia
from decision_engine.threshold_acceptance_gate import ThresholdAcceptanceGate
from decision_engine.unknown_labeler import label_unknown
from decision_engine.track_identity_state import TrackIdentityState
from dynamic_capture.capture_trigger_evaluator import CaptureTriggerEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrackResult:
    track_id: int
    bbox: tuple
    identity: str
    confidence: float
    branch_used: str
    captured: bool = False


@dataclass
class PipelineOrchestrator:
    """Orquesta el flujo completo por fotograma sobre un conjunto de Track-IDs."""

    detector: YOLOv8nDetector
    tracker: ByteTrackAdapter | DeepSORTAdapter
    registry: TrackRegistry = field(default_factory=TrackRegistry)
    identity_state: TrackIdentityState = field(default_factory=TrackIdentityState)
    voting: WeightedVotingInertia = field(default_factory=WeightedVotingInertia)
    gate: ThresholdAcceptanceGate = field(default_factory=ThresholdAcceptanceGate)
    capture_evaluator: CaptureTriggerEvaluator = field(default_factory=CaptureTriggerEvaluator)
    executor: concurrent.futures.ThreadPoolExecutor = field(init=False)

    def __post_init__(self):
        # i9-14900HX tiene 24 cores (8P+16E), 32 hilos. 16 workers es un buen balance para inferencia concurrente.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> List[TrackResult]:
        detections: List[Detection] = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)

        def _process_track(track):
            roi = self._crop_roi(frame, track.bbox)
            if roi is None or roi.size == 0:
                return None

            branch, face_conf, face_result = route_branch(roi)

            if branch == "ID":
                identity, raw_confidence = run_id_branch(roi, face_result=face_result)
            else:
                identity, raw_confidence = run_reid_branch(roi)

            self.voting.accumulate(track.track_id, branch, identity, raw_confidence)
            winner_identity, winner_confidence = self.voting.get_winner(track.track_id)

            accepted = self.gate.accept(winner_confidence)
            final_identity = winner_identity if accepted else label_unknown()

            final_identity = self.identity_state.resolve(
                track_id=track.track_id,
                branch=branch,
                candidate_identity=final_identity,
                accepted=accepted,
            )

            self.registry.update(track.track_id, final_identity, winner_confidence, frame_index)

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
                confidence=winner_confidence,
                branch_used=branch,
                captured=captured,
            )

        results: List[TrackResult] = []
        for res in self.executor.map(_process_track, tracks):
            if res is not None:
                results.append(res)
                
        return results

    @staticmethod
    def _crop_roi(frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()