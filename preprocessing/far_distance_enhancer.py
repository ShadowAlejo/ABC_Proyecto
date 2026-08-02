"""Filtro de refinamiento intermedio para detecciones de YOLOv8 a gran distancia.
Restaura recortes lejanos aplicando expansión contextual (padding adaptativo),
interpolación espacial de alto grado (Lanczos-4), desenfoque de máscara (Unsharp Masking)
y normalización fotométrica regional en el espacio LAB mediante CLAHE."""

from typing import Optional, Tuple
import numpy as np
import cv2

# Umbrales de Histéresis para detección a distancia
FAR_THRESHOLD_LOW = 100    # Por debajo de 100px: Activa restauración profunda
FAR_THRESHOLD_HIGH = 140   # Por encima de 140px: Desactiva restauración profunda

_last_enhancement_state: dict[int, bool] = {}


def enhance_far_distance_roi(
    frame: np.ndarray,
    bbox: tuple,
    track_id: Optional[int] = None,
    threshold_low: int = FAR_THRESHOLD_LOW,
    threshold_high: int = FAR_THRESHOLD_HIGH,
) -> Tuple[Optional[np.ndarray], bool, Tuple[int, int]]:
    """Extrae un ROI de la imagen original con padding asimétrico uniforme.
    
    Aplica restauración algorítmica proporcional si la persona está a gran distancia,
    usando histéresis (100-140px) para evitar oscilaciones de confianza entre fotogramas.

    Args:
        frame: Imagen BGR original de alta resolución.
        bbox: Coordenadas de la detección de YOLO (x1, y1, x2, y2).
        track_id: Identificador del track para mantener la memoria de histéresis.
        threshold_low: Umbral inferior de activación.
        threshold_high: Umbral superior de desactivación.

    Returns:
        Tupla (roi_extraido, is_enhanced, (pad_x, pad_top))
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    frame_h, frame_w = frame.shape[:2]

    # Prevenir cajas invertidas o fuera de rango
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return None, False, (0, 0)

    # 1. Expansión Contextual Adaptativa Homogénea (Padding Asimétrico)
    # +25% superior para preservar frente/cabeza, +15% lateral, +10% inferior
    pad_x = int(w * 0.15)
    pad_top = int(h * 0.25)
    pad_bottom = int(h * 0.10)

    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_top)
    x2_pad = min(frame_w, x2 + pad_x)
    y2_pad = min(frame_h, y2 + pad_bottom)

    roi_padded = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()
    if roi_padded.size == 0:
        return None, False, (0, 0)

    actual_pad_x = x1 - x1_pad
    actual_pad_top = y1 - y1_pad
    offsets = (actual_pad_x, actual_pad_top)

    # 2. Discriminador con Histéresis de Escala
    prev_state = _last_enhancement_state.get(track_id, False) if track_id is not None else False
    if h < threshold_low:
        should_enhance = True
    elif h > threshold_high:
        should_enhance = False
    else:
        should_enhance = prev_state

    if track_id is not None:
        _last_enhancement_state[track_id] = should_enhance

    if not should_enhance:
        return roi_padded, False, offsets

    # 3. Restauración Proporcional (Preservando relación de aspecto natural)
    # 3.1. Reenfoque de Frecuencias (Unsharp Masking Adaptativo)
    blurred = cv2.GaussianBlur(roi_padded, (0, 0), 1.5)
    unsharp = cv2.addWeighted(roi_padded, 1.4, blurred, -0.4, 0)

    # 4. Normalización Fotométrica Regional (CLAHE en Espacio LAB)
    lab = cv2.cvtColor(unsharp, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_clahe = clahe.apply(l_channel)

    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    enhanced_roi = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return enhanced_roi, True, offsets
