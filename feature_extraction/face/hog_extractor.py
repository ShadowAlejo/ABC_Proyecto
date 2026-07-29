"""Extrae el descriptor facial HOG maximizado (Piramidal + Componentes locales + Opponent-HOG).

Para maximizar la precision, esta implementacion aplica:
  1. Filtro Tan & Triggs (Gamma + DoG + Truncado) a la Intensidad.
  2. Gradientes con signo (0-360) (signedGradient=True).
  3. HOG Piramidal (escalas 4x4, 8x8, 16x16).
  4. HOG Basado en Componentes Locales (6 parches: ojos, nariz, comisuras, entrecejo).
  5. Opponent-HOG: extraccion en canales de Intensidad, O1 (R-G) y O2 (R+G-2B).
"""
import cv2
import numpy as np

# ── HOG Piramidal Global (64x64) con 12 orientaciones y signo ──────────────────
_HOG_FINE = cv2.HOGDescriptor(
    _winSize=(64, 64), _blockSize=(8, 8), _blockStride=(4, 4), _cellSize=(4, 4), _nbins=12, signedGradient=True
)
_HOG_MID = cv2.HOGDescriptor(
    _winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=12, signedGradient=True
)
_HOG_COARSE = cv2.HOGDescriptor(
    _winSize=(64, 64), _blockSize=(32, 32), _blockStride=(16, 16), _cellSize=(16, 16), _nbins=12, signedGradient=True
)

# ── HOG de Componentes Locales (16x16) con 12 orientaciones y signo ────────────
_HOG_COMPONENT = cv2.HOGDescriptor(
    _winSize=(16, 16), _blockSize=(8, 8), _blockStride=(4, 4), _cellSize=(4, 4), _nbins=12, signedGradient=True
)


def _apply_tan_triggs(gray: np.ndarray, alpha=0.1, tau=10.0, gamma=0.2, sigma0=1, sigma1=2) -> np.ndarray:
    """Filtro de Iluminacion Tan & Triggs: Gamma -> DoG -> Truncado de contraste."""
    # 1. Gamma Correction
    img = np.power(gray.astype(np.float32) / 255.0, gamma)

    # 2. Difference of Gaussians (DoG)
    g1 = cv2.GaussianBlur(img, (0, 0), sigma0)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma1)
    dog = g1 - g2

    # 3. Contrast Truncation
    dog = dog / (np.mean(np.abs(dog)) ** alpha)
    dog = dog / (np.mean(np.minimum(np.abs(dog), tau)) ** alpha)
    dog = tau * np.tanh(dog / tau)
    
    return cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def _extract_patch(gray: np.ndarray, x: float, y: float, size: int = 16) -> np.ndarray:
    """Extrae un parche centrado en (x, y). Si sale de los bordes, hace padding."""
    h, w = gray.shape
    half = size // 2
    ix, iy = int(round(x)), int(round(y))

    y1, y2 = iy - half, iy + half
    x1, x2 = ix - half, ix + half

    if x2 <= 0 or x1 >= w or y2 <= 0 or y1 >= h:
        return np.zeros((size, size), dtype=gray.dtype)

    pad_y1 = max(0, -y1)
    pad_y2 = max(0, y2 - h)
    pad_x1 = max(0, -x1)
    pad_x2 = max(0, x2 - w)
    
    safe_y1 = max(0, min(h, y1))
    safe_y2 = max(0, min(h, y2))
    safe_x1 = max(0, min(w, x1))
    safe_x2 = max(0, min(w, x2))
    
    crop = gray[safe_y1:safe_y2, safe_x1:safe_x2]
    
    if crop.size > 0:
        return np.pad(crop, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='reflect')
    else:
        return np.zeros((size, size), dtype=gray.dtype)


def _compute_channel_hog(channel_8u: np.ndarray, landmarks: list = None) -> np.ndarray:
    """Calcula HOG Piramidal + Local para un solo canal de 8 bits."""
    # Piramidal Global
    feat_fine   = _HOG_FINE.compute(channel_8u).flatten().astype(np.float32)
    feat_mid    = _HOG_MID.compute(channel_8u).flatten().astype(np.float32)
    feat_coarse = _HOG_COARSE.compute(channel_8u).flatten().astype(np.float32)
    
    global_hog = np.concatenate([feat_fine, feat_mid, feat_coarse], axis=0)
    norm_g = np.linalg.norm(global_hog)
    if norm_g > 1e-6:
        global_hog = global_hog / norm_g

    # Local Componentes (6 parches)
    component_features = []
    if landmarks is not None and len(landmarks) >= 5:
        lx_eyeL, ly_eyeL = landmarks[0]
        lx_eyeR, ly_eyeR = landmarks[1]
        lx_nose, ly_nose = landmarks[2]
        lx_mouthL, ly_mouthL = landmarks[3]
        lx_mouthR, ly_mouthR = landmarks[4]
        
        # 6 Puntos: OjoIzq, OjoDer, Nariz, BocaIzq, BocaDer, Entrecejo
        lx_inter, ly_inter = (lx_eyeL + lx_eyeR) / 2.0, (ly_eyeL + ly_eyeR) / 2.0
        
        points = [
            (lx_eyeL, ly_eyeL), (lx_eyeR, ly_eyeR), (lx_nose, ly_nose),
            (lx_mouthL, ly_mouthL), (lx_mouthR, ly_mouthR), (lx_inter, ly_inter)
        ]
        
        for px, py in points:
            patch = _extract_patch(channel_8u, px, py, size=16)
            patch_hog = _HOG_COMPONENT.compute(patch).flatten().astype(np.float32)
            component_features.append(patch_hog)
            
        local_hog = np.concatenate(component_features, axis=0)
        norm_l = np.linalg.norm(local_hog)
        if norm_l > 1e-6:
            local_hog = local_hog / norm_l
    else:
        local_hog = np.zeros((1296 * 2,), dtype=np.float32) # Fallback (6 parches * dims)

    return np.concatenate([global_hog, local_hog], axis=0)


def extract_hog_features(face_bgr_64x64: np.ndarray, landmarks: list = None) -> np.ndarray:
    """Extrae HOG en Espacio Oponente de Color y lo concatena.

    Args:
        face_bgr_64x64: Imagen BGR de 64x64 alineada afinalmente.
        landmarks: Lista de 5 puntos (x, y) de la cara alineada.

    Returns:
        Vector resultante concatenado.
    """
    if face_bgr_64x64 is None or face_bgr_64x64.shape[:2] != (64, 64):
        raise ValueError("Se requiere imagen facial alineada de 64x64 BGR.")

    b_float = face_bgr_64x64[:, :, 0].astype(np.float32)
    g_float = face_bgr_64x64[:, :, 1].astype(np.float32)
    r_float = face_bgr_64x64[:, :, 2].astype(np.float32)

    # Canal 1: Intensidad (Grayscale) con Tan & Triggs
    gray = cv2.cvtColor(face_bgr_64x64, cv2.COLOR_BGR2GRAY)
    channel_intensity = _apply_tan_triggs(gray)

    # Canal 2: Opponent 1 (R - G)
    o1 = r_float - g_float
    o1_norm = cv2.normalize(o1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Canal 3: Opponent 2 (R + G - 2B)
    o2 = r_float + g_float - 2.0 * b_float
    o2_norm = cv2.normalize(o2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    hog_int = _compute_channel_hog(channel_intensity, landmarks)
    hog_o1  = _compute_channel_hog(o1_norm, landmarks)
    hog_o2  = _compute_channel_hog(o2_norm, landmarks)

    # Vector gigante final
    return np.concatenate([hog_int, hog_o1, hog_o2], axis=0)