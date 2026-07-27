"""Aísla la ROI de cuerpo/torso eliminando fondo y redimensiona a la ventana canónica 64x128 [REQ-RID-02]."""
import cv2
import numpy as np
from preprocessing.roi_resizer import resize_body_roi


def isolate_torso_roi(body_roi: np.ndarray, use_grabcut: bool = False) -> np.ndarray:
    """Aísla el torso/cuerpo de la ROI detectada y la redimensiona a 64x128."""
    if body_roi is None or body_roi.size == 0:
        raise ValueError("ROI corporal vacía recibida en torso_roi_isolator.")

    processed = body_roi
    if use_grabcut and body_roi.shape[0] > 10 and body_roi.shape[1] > 10:
        processed = _apply_grabcut_background_removal(body_roi)

    return resize_body_roi(processed)


def _apply_grabcut_background_removal(image: np.ndarray) -> np.ndarray:
    """Aplica GrabCut asumiendo que el sujeto ocupa el centro de la ROI para eliminar fondo."""
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    h, w = image.shape[:2]
    margin_w, margin_h = int(w * 0.08), int(h * 0.05)
    rect = (margin_w, margin_h, w - 2 * margin_w, h - 2 * margin_h)

    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        return image * mask2[:, :, np.newaxis]
    except cv2.error:
        return image