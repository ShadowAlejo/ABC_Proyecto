"""Calcula el descriptor HoG piramidal multi-escala sobre la imagen facial normalizada [REQ-FAC-03].

Mejora de invarianza a distancia:
  El mismo descriptor se extrae con celdas de 4×4, 8×8 y 16×16 píxeles sobre la imagen 64×64.
  - Celdas 4×4: captura textura fina (poros, cejas, vello) → útil cuando el rostro es grande/cercano.
  - Celdas 8×8: equilibrio entre textura y forma (ojos, nariz, boca) → escala media.
  - Celdas 16×16: contorno y forma global del rostro → útil cuando el rostro es pequeño/lejano.
  Los tres vectores se concatenan y se normalizan L2 globalmente. El SVM recibe la misma
  "firma piramidal" independientemente de la distancia a cámara.
"""
import cv2
import numpy as np

# ── Descriptores HOG a tres granularidades ──────────────────────────────────
_HOG_FINE = cv2.HOGDescriptor(      # Celda 4×4: textura fina
    _winSize=(64, 64),
    _blockSize=(8, 8),
    _blockStride=(4, 4),
    _cellSize=(4, 4),
    _nbins=9,
)

_HOG_MID = cv2.HOGDescriptor(       # Celda 8×8: escala media (descriptor original)
    _winSize=(64, 64),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)

_HOG_COARSE = cv2.HOGDescriptor(    # Celda 16×16: forma global / rostro lejano
    _winSize=(64, 64),
    _blockSize=(32, 32),
    _blockStride=(16, 16),
    _cellSize=(16, 16),
    _nbins=9,
)


def extract_hog_features(face_gray_64x64: np.ndarray) -> np.ndarray:
    """Extrae el vector HOG piramidal (fine + mid + coarse) de una imagen facial 64×64.

    El vector resultante combina información de textura fina, estructura media y
    forma global, dando invarianza a si el rostro fue capturado de cerca (grande)
    o de lejos (pequeño). Se aplica normalización L2 al vector concatenado para que
    el SVM lineal pese cada nivel de la pirámide de forma equitativa.

    Dimensiones de salida:
      - Fine  (4×4 cells): 3969 elementos
      - Mid   (8×8 cells):  1764 elementos  (nivel original)
      - Coarse (16×16 cells):  81 elementos
      Total: 5814 elementos antes de L2.
    """
    if face_gray_64x64 is None or face_gray_64x64.shape[:2] != (64, 64):
        raise ValueError("Se requiere una imagen facial normalizada de 64x64 en escala de grises.")

    feat_fine   = _HOG_FINE.compute(face_gray_64x64).flatten().astype(np.float32)
    feat_mid    = _HOG_MID.compute(face_gray_64x64).flatten().astype(np.float32)
    feat_coarse = _HOG_COARSE.compute(face_gray_64x64).flatten().astype(np.float32)

    combined = np.concatenate([feat_fine, feat_mid, feat_coarse], axis=0)

    # Normalización L2 global: todos los niveles de la pirámide pesan igual
    norm = np.linalg.norm(combined)
    if norm > 1e-6:
        combined = combined / norm

    return combined