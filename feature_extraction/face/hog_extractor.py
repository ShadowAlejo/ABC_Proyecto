"""Extrae el descriptor facial HOG maximizado (Piramidal + Componentes locales).

De acuerdo a las reglas del sistema, para identificacion facial SOLO se usa HOG.
Para maximizar su precision, esta implementacion aplica:
  1. Transformacion Gamma (Raiz cuadrada) previa para resistencia a iluminacion.
  2. 12 orientaciones de gradiente (en lugar de 9) para mayor detalle.
  3. HOG Piramidal (escalas 4x4, 8x8, 16x16).
  4. HOG Basado en Componentes: recortes locales de 16x16 sobre ojos y nariz guiados
     por landmarks, forzando atencion estructural a las micro-zonas clave.

Dimension total: 14,880 caracteristicas (comprimidas luego por PCA a 512).
"""
import cv2
import numpy as np

# ── HOG Piramidal Global (64x64) con 12 orientaciones ──────────────────────────
_HOG_FINE = cv2.HOGDescriptor(
    _winSize=(64, 64), _blockSize=(8, 8), _blockStride=(4, 4), _cellSize=(4, 4), _nbins=12
)
_HOG_MID = cv2.HOGDescriptor(
    _winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=12
)
_HOG_COARSE = cv2.HOGDescriptor(
    _winSize=(64, 64), _blockSize=(32, 32), _blockStride=(16, 16), _cellSize=(16, 16), _nbins=12
)

# ── HOG de Componentes Locales (16x16) con 12 orientaciones ────────────────────
_HOG_COMPONENT = cv2.HOGDescriptor(
    _winSize=(16, 16), _blockSize=(8, 8), _blockStride=(4, 4), _cellSize=(4, 4), _nbins=12
)


def _extract_patch(gray: np.ndarray, x: float, y: float, size: int = 16) -> np.ndarray:
    """Extrae un parche centrado en (x, y). Si sale de los bordes, hace padding (reflect)."""
    h, w = gray.shape
    half = size // 2
    ix, iy = int(round(x)), int(round(y))

    y1, y2 = iy - half, iy + half
    x1, x2 = ix - half, ix + half

    # Si el parche se sale de la imagen, usamos pad
    if y1 < 0 or y2 > h or x1 < 0 or x2 > w:
        pad_y1 = max(0, -y1)
        pad_y2 = max(0, y2 - h)
        pad_x1 = max(0, -x1)
        pad_x2 = max(0, x2 - w)
        
        y1 = max(0, y1)
        y2 = min(h, y2)
        x1 = max(0, x1)
        x2 = min(w, x2)
        
        crop = gray[y1:y2, x1:x2]
        padded = np.pad(crop, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='reflect')
        return padded
    else:
        return gray[y1:y2, x1:x2]


def extract_hog_features(face_gray_64x64: np.ndarray, landmarks: list = None) -> np.ndarray:
    """Extrae el vector HOG maximizado de una imagen facial 64x64.

    Args:
        face_gray_64x64: Imagen normalizada en escala de grises.
        landmarks: Lista de 5 puntos (x, y) en coordenadas relativas a la imagen 64x64.
                   [ojo_izq, ojo_der, nariz, boca_izq, boca_der].
                   Si es None, los componentes locales se rellenan con ceros.

    Returns:
        Vector HOG combinado de 14,880 dimensiones (L2 normalizado por partes).
    """
    if face_gray_64x64 is None or face_gray_64x64.shape[:2] != (64, 64):
        raise ValueError("Se requiere imagen facial normalizada de 64x64.")

    # 1. Transformacion Gamma (Raiz cuadrada) para invarianza a iluminacion
    # cv2.HOG no tiene transform_sqrt integrado facil, lo aplicamos manualmente
    float_img = face_gray_64x64.astype(np.float32) / 255.0
    gamma_img = np.sqrt(float_img)
    gamma_8u = (gamma_img * 255.0).astype(np.uint8)

    # 2. HOG Piramidal Global (13,584 dims)
    feat_fine   = _HOG_FINE.compute(gamma_8u).flatten().astype(np.float32)
    feat_mid    = _HOG_MID.compute(gamma_8u).flatten().astype(np.float32)
    feat_coarse = _HOG_COARSE.compute(gamma_8u).flatten().astype(np.float32)
    
    global_hog = np.concatenate([feat_fine, feat_mid, feat_coarse], axis=0)
    norm_g = np.linalg.norm(global_hog)
    if norm_g > 1e-6:
        global_hog = global_hog / norm_g

    # 3. HOG Basado en Componentes Locales (1,296 dims)
    component_features = []
    
    if landmarks is not None and len(landmarks) >= 3:
        # Puntos: 0=Ojo_Izq, 1=Ojo_Der, 2=Nariz
        for i in range(3):
            lx, ly = landmarks[i]
            patch = _extract_patch(gamma_8u, lx, ly, size=16)
            patch_hog = _HOG_COMPONENT.compute(patch).flatten().astype(np.float32)
            component_features.append(patch_hog)
            
        local_hog = np.concatenate(component_features, axis=0)
        norm_l = np.linalg.norm(local_hog)
        if norm_l > 1e-6:
            local_hog = local_hog / norm_l
    else:
        # Fallback si por alguna razon no hay landmarks (1296 dims en cero)
        local_hog = np.zeros((1296,), dtype=np.float32)

    # 4. Concatenacion Final (13584 + 1296 = 14880 dims)
    return np.concatenate([global_hog, local_hog], axis=0)