"""Filtro de refinamiento intermedio para detecciones de YOLOv8 a gran distancia.
Restaura recortes lejanos aplicando expansión contextual (padding adaptativo),
interpolación espacial de alto grado (Lanczos-4), desenfoque de máscara (Unsharp Masking)
y normalización fotométrica regional en el espacio LAB mediante CLAHE."""

from typing import Optional
import numpy as np
import cv2

# Umbral por defecto: si el recorte original (YOLO) mide menos de 120 px de alto,
# el sujeto está lo suficientemente lejos como para ameritar el pipeline de restauración profunda.
FAR_DISTANCE_HEIGHT_THRESHOLD = 120

# Tamaño canónico esperado por el extractor corporal (Re-ID LBP)
CANONICAL_SIZE = (128, 256)

def enhance_far_distance_roi(frame: np.ndarray, bbox: tuple, threshold: int = FAR_DISTANCE_HEIGHT_THRESHOLD) -> Optional[np.ndarray]:
    """Extrae un ROI de la imagen original. Si la persona está lejos, aplica
    una restauración algorítmica profunda antes de retornar la matriz.
    
    Args:
        frame: Imagen BGR original de alta resolución.
        bbox: Coordenadas de la detección de YOLO (x1, y1, x2, y2).
        threshold: Umbral en píxeles de altura para activar la restauración.
        
    Returns:
        np.ndarray BGR con la región extraída (y potencialmente mejorada) o None.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    frame_h, frame_w = frame.shape[:2]
    
    # Prevenir cajas invertidas o fuera de rango
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    
    w = x2 - x1
    h = y2 - y1
    
    if w <= 0 or h <= 0:
        return None
        
    # 1. Discriminador de Escala y Distancia
    if h >= threshold:
        # Bypass: Si la persona está cerca, pasa directo sin latencia
        return frame[y1:y2, x1:x2].copy()
        
    # 2. Expansión Contextual Adaptativa (Padding Dinámico del 15%)
    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)
    
    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(frame_w, x2 + pad_x)
    y2_pad = min(frame_h, y2 + pad_y)
    
    roi_padded = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()
    if roi_padded.size == 0:
        return None
        
    # 3. Reconstrucción Espacial por Interpolación Lanczos-4
    # Escala el recorte expandido de baja resolución directo a la rejilla del clasificador Re-ID
    resized_roi = cv2.resize(roi_padded, CANONICAL_SIZE, interpolation=cv2.INTER_LANCZOS4)
    
    # 3.1. Reenfoque de Frecuencias (Unsharp Masking Adaptativo)
    # Suavizamos y restamos al original para extraer los bordes (costuras, texturas) de alta frecuencia
    blurred = cv2.GaussianBlur(resized_roi, (0, 0), 2.0)
    unsharp = cv2.addWeighted(resized_roi, 1.5, blurred, -0.5, 0)
    
    # 4. Normalización Fotométrica Regional (CLAHE en Espacio LAB)
    lab = cv2.cvtColor(unsharp, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Ecualización sobre la luminancia localizada
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_clahe = clahe.apply(l_channel)
    
    # Fusionamos de vuelta y pasamos a BGR
    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    enhanced_roi = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # 5. Re-Inyección al Pipeline
    return enhanced_roi
