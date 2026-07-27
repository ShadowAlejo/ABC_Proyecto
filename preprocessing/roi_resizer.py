"""Redimensiona por interpolación cualquier ROI a su tamaño canónico (64x64 rostro, 64x128 cuerpo)."""
import cv2
import numpy as np
from preprocessing.antialiasing_filter import apply_antialiasing

FACE_CANONICAL_SIZE = (64, 64)   # (width, height)
BODY_CANONICAL_SIZE = (64, 128)  # (width, height)


def resize_roi(roi: np.ndarray, target_size: tuple[int, int], apply_smoothing: bool = True) -> np.ndarray:
    """Redimensiona una ROI a target_size=(w, h), aplicando anti-aliasing previo si se reduce la imagen."""
    if roi is None or roi.size == 0:
        raise ValueError("ROI vacía o nula recibida en roi_resizer.")

    src_h, src_w = roi.shape[:2]
    dst_w, dst_h = target_size

    if apply_smoothing and (dst_w < src_w or dst_h < src_h):
        roi = apply_antialiasing(roi)

    interp = cv2.INTER_AREA if (dst_w <= src_w and dst_h <= src_h) else cv2.INTER_LINEAR
    return cv2.resize(roi, (dst_w, dst_h), interpolation=interp)


def resize_face_roi(roi: np.ndarray) -> np.ndarray:
    return resize_roi(roi, FACE_CANONICAL_SIZE)


def resize_body_roi(roi: np.ndarray) -> np.ndarray:
    return resize_roi(roi, BODY_CANONICAL_SIZE)