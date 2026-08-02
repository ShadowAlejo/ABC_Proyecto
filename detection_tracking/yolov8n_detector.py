"""Carga YOLOv8n, filtra exclusivamente la clase 'person' (clase 0 COCO), conf >= 0.40."""
from dataclasses import dataclass
from typing import List
import numpy as np
from ultralytics import YOLO
from detection_tracking.bbox_utils import validate_min_dimensions
from utils.logger import get_logger

logger = get_logger(__name__)

COCO_PERSON_CLASS_ID = 0
DEFAULT_CONF_THRESHOLD = 0.40  # [REQ-DET-02]


@dataclass
class Detection:
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = COCO_PERSON_CLASS_ID


class YOLOv8nDetector:
    """Detector de personas basado en YOLOv8n [REQ-DET-01, REQ-DET-02, REQ-DET-03]."""

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                 device: str = "cpu"):
        import torch
        self.conf_threshold = conf_threshold
        
        # Auto-fallback si se pide GPU pero no hay CUDA
        if str(device).lower() in ["gpu", "cuda", "0"]:
            if torch.cuda.is_available():
                self.device = "0"
            else:
                logger.warning(f"Device '{device}' solicitado pero CUDA no está disponible. Cayendo a 'cpu'.")
                self.device = "cpu"
        else:
            self.device = "cpu"
            
        logger.info(f"Cargando modelo YOLOv8n desde: {model_path} (Device: {self.device})")
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Devuelve únicamente detecciones de la clase 'person' con confianza >= umbral."""
        # Se restaura la resolución original para priorizar precisión sobre la velocidad en CPU.
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=[COCO_PERSON_CLASS_ID],
            device=self.device,
            half=False,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            if cls_id != COCO_PERSON_CLASS_ID:
                continue
            conf = float(box.conf.item())
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = (x1, y1, x2, y2)
            if not validate_min_dimensions(bbox):
                continue

            detections.append(Detection(bbox=bbox, confidence=conf))

        return detections