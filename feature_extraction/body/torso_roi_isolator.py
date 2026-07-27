"""Aísla la ROI de cuerpo/torso eliminando fondo y redimensiona a la ventana canónica 64x128 [REQ-RID-02]."""
import cv2
import numpy as np
from preprocessing.roi_resizer import resize_body_roi
from preprocessing.white_patch_normalizer import apply_white_patch


def isolate_torso_roi(body_roi: np.ndarray, use_grabcut: bool = False, enhance: bool = False) -> np.ndarray:
    """Aísla el torso/cuerpo de la ROI detectada y la redimensiona a 64x128.

    Args:
        body_roi: Imagen BGR del sujeto.
        use_grabcut: Si True, aplica remoción de fondo mediante GrabCut.
        enhance: Si True, aplica White-Patch + CLAHE para normalizar iluminación (recomendado en entrenamiento).
    """
    if body_roi is None or body_roi.size == 0:
        raise ValueError("ROI corporal vacía recibida en torso_roi_isolator.")

    processed = body_roi
    if enhance:
        processed = _enhance_body_illumination(processed)

    if use_grabcut and processed.shape[0] > 10 and processed.shape[1] > 10:
        processed = _apply_grabcut_background_removal(processed)

    return resize_body_roi(processed)


def _enhance_body_illumination(image: np.ndarray) -> np.ndarray:
    """Normaliza la temperatura cromática (White-Patch) y la luminancia local (CLAHE) del cuerpo."""
    try:
        wp_img = apply_white_patch(image)
        lab = cv2.cvtColor(wp_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        merged = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return image


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