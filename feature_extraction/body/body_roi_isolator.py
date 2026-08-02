"""Prepara la ROI de cuerpo completo redimensionándola a la ventana canónica 128x256 [REQ-RID-02]."""
import cv2
import numpy as np
from preprocessing.roi_resizer import resize_body_roi

def isolate_body_roi(body_roi: np.ndarray) -> np.ndarray:
    """Prepara el cuerpo completo (Bounding Box de YOLO) y lo redimensiona a 128x256.
    Se eliminan preprocesamientos fotométricos destructivos para LBP (CLAHE/White-Patch)
    y no se recorta artificialmente el torso.
    """
    if body_roi is None or body_roi.size == 0:
        raise ValueError("ROI corporal vacía recibida en body_roi_isolator.")
        
    return resize_body_roi(body_roi)
