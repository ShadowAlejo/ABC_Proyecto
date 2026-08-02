"""Función de composición central: encadena detección, ruteo ID/Re-ID,
motor de decisión y captura dinámica. Opera de forma Stateless (sin tracker)."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from detection_tracking.yolov8n_detector import YOLOv8nDetector, Detection
from detection_tracking.track_registry import TrackRegistry
from branching.face_visibility_router import route_branch_with_result, _get_face_detector
from branching.id_branch_pipeline import run_id_branch
from branching.reid_branch_pipeline import run_reid_branch
from decision_engine.threshold_acceptance_gate import ThresholdAcceptanceGate
from decision_engine.unknown_labeler import label_unknown
from dynamic_capture.capture_trigger_evaluator import CaptureTriggerEvaluator
from preprocessing.far_distance_enhancer import enhance_far_distance_roi
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

    def __post_init__(self):
        pass

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

        # 1. Extracción secuencial de ROIs y Padding
        rois = []
        is_enhanced_list = []
        for track in ephemeral_tracks:
            roi, is_enhanced = enhance_far_distance_roi(frame, track.bbox)
            rois.append(roi)
            is_enhanced_list.append(is_enhanced)
            
        # 2. Inferencia Facial Batch (GPU/CPU Unificado)
        face_detector = _get_face_detector()
        face_results = face_detector.detect_batch(rois, is_enhanced_list)
        
        # 3. Procesamiento y Clasificación Secuencial (CPU rápida)
        raw_results: List[TrackResult] = []
        
        for track, roi, is_enhanced, face_res in zip(ephemeral_tracks, rois, is_enhanced_list, face_results):
            if roi is None or roi.size == 0:
                continue
                
            branch, face_conf, face_result = route_branch_with_result(roi, face_res)

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

            raw_results.append(
                TrackResult(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    identity=final_identity,
                    confidence=raw_confidence,
                    branch_used=branch,
                    captured=captured,
                )
            )
                
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