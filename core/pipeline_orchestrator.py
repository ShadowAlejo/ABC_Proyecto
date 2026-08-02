"""Función de composición central: encadena detección (YOLOv8n), ruteo ID/Re-ID sin YuNet/Tracker,
motor de decisión y captura dinámica automática en dataset/captures/."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import concurrent.futures

from detection_tracking.yolov8n_detector import YOLOv8nDetector, Detection
from branching.face_visibility_router import route_branch
from branching.id_branch_pipeline import run_id_branch
from branching.reid_branch_pipeline import run_reid_branch
from decision_engine.threshold_acceptance_gate import ThresholdAcceptanceGate
from decision_engine.unknown_labeler import label_unknown
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
    """Orquesta el flujo completo por fotograma procesando detecciones y activando capturas dinámicas."""

    detector: YOLOv8nDetector
    gate: ThresholdAcceptanceGate = field(default_factory=ThresholdAcceptanceGate)
    capture_evaluator: CaptureTriggerEvaluator = field(default_factory=CaptureTriggerEvaluator)
    executor: concurrent.futures.ThreadPoolExecutor = field(init=False)

    def __post_init__(self):
        from utils.config_loader import ConfigLoader
        config = ConfigLoader.load("config.yaml")
        t_ac = config.get("decision", {}).get("t_aceptacion", 0.80)
        self.gate = ThresholdAcceptanceGate(t_aceptacion=t_ac)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> List[TrackResult]:
        detections: List[Detection] = self.detector.detect(frame)

        def _process_det(item):
            idx, det = item
            roi = self._crop_roi(frame, det.bbox)
            if roi is None or roi.size == 0:
                return None

            branch, head_crop = route_branch(roi)

            if branch == "ID":
                identity, raw_confidence = run_id_branch(roi, head_crop=head_crop)
                if raw_confidence < self.gate.t_aceptacion:
                    reid_identity, reid_confidence = run_reid_branch(roi)
                    if reid_confidence > raw_confidence and reid_identity != "Desconocido":
                        identity = reid_identity
                        raw_confidence = reid_confidence
                        branch = "REID"
            else:
                identity, raw_confidence = run_reid_branch(roi)

            accepted = self.gate.accept(raw_confidence)
            final_identity = identity if accepted else label_unknown()

            captured = False
            if accepted and final_identity != "Desconocido":
                captured = self.capture_evaluator.maybe_capture(
                    track_id=idx + 1,
                    roi=roi,
                    identity=final_identity,
                    frame_index=frame_index,
                    bbox=det.bbox,
                )

            return TrackResult(
                track_id=idx + 1,
                bbox=det.bbox,
                identity=final_identity,
                confidence=raw_confidence,
                branch_used=branch,
                captured=captured,
            )

        items = list(enumerate(detections))
        raw_results: List[TrackResult] = []
        for res in self.executor.map(_process_det, items):
            if res is not None:
                raw_results.append(res)

        # Exclusión Mutua en el mismo frame (dos detecciones no pueden reclamar la misma persona)
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