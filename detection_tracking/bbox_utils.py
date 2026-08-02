"""Funciones puras para conversión de coordenadas, cálculo de IoU y validación de dimensiones mínimas de ROI."""
from typing import Tuple
import numpy as np

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


def xywh_to_xyxy(bbox: BBox) -> BBox:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def xyxy_to_xywh(bbox: BBox) -> BBox:
    x1, y1, x2, y2 = bbox
    return (x1, y1, x2 - x1, y2 - y1)


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    """Calcula Intersection over Union entre dos cajas en formato (x1, y1, x2, y2)."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def validate_min_dimensions(bbox: BBox, min_width: int = 20, min_height: int = 40) -> bool:
    """Valida que la ROI cumpla dimensiones mínimas para evitar degradación de detalle en LBP/HoG."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) >= min_width and (y2 - y1) >= min_height


def clip_bbox_to_frame(bbox: BBox, frame_shape: Tuple[int, int]) -> BBox:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return (
        float(np.clip(x1, 0, w)),
        float(np.clip(y1, 0, h)),
        float(np.clip(x2, 0, w)),
        float(np.clip(y2, 0, h)),
    )