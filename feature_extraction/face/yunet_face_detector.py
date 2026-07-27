"""Detector facial YuNet robusto: multi-escala, recuperación de casos difíciles y
descarte temprano de falsos positivos evidentes mediante landmarks [REQ-FAC-01]."""
from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FACE_CONF_THRESHOLD = 0.6

# Umbral interno más laxo para NO perder candidatos (evita falsos negativos);
# el filtrado fino real ocurre después, vía landmarks + confianza combinada.
INTERNAL_DETECTOR_THRESHOLD = 0.3
NMS_IOU_THRESHOLD = 0.3


@dataclass
class FaceDetectionResult:
    bbox: Optional[tuple]          # (x, y, w, h)
    confidence: float
    detected: bool
    landmarks: Optional[np.ndarray] = None  # 5 puntos: ojo_izq, ojo_der, nariz, boca_izq, boca_der
    all_candidates: List[tuple] = field(default_factory=list)  # (bbox, confidence, landmarks) sin filtrar


class YuNetFaceDetector:
    """
    Detector facial YuNet con estrategia de máxima cobertura:
    - Detecta con umbral interno bajo (recall alto).
    - Prueba múltiples escalas de entrada si la imagen es muy grande o muy pequeña.
    - Prueba variantes con ecualización de contraste si la detección inicial falla (recupera falsos negativos
      causados por iluminación pobre).
    - Filtra candidatos falsos-positivos mediante geometría de landmarks antes de aceptar el mejor resultado.
    """

    def __init__(self, model_path: str = "models/face_detection_yunet.onnx",
                 conf_threshold: float = DEFAULT_FACE_CONF_THRESHOLD,
                 input_size: tuple[int, int] = (320, 320),
                 enable_multiscale_retry: bool = True,
                 enable_contrast_retry: bool = True):
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.enable_multiscale_retry = enable_multiscale_retry
        self.enable_contrast_retry = enable_contrast_retry

        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=INTERNAL_DETECTOR_THRESHOLD,
            nms_threshold=NMS_IOU_THRESHOLD,
            top_k=20,
        )

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def detect(self, roi: np.ndarray) -> FaceDetectionResult:
        """Ejecuta la estrategia completa de detección robusta y devuelve el mejor candidato validado."""
        if roi is None or roi.size == 0:
            return FaceDetectionResult(bbox=None, confidence=0.0, detected=False)

        candidates = self._run_raw_detection(roi)

        # Recuperación de falsos negativos: si no hay candidatos, reintenta con variantes de la imagen.
        if not candidates and self.enable_contrast_retry:
            candidates = self._run_raw_detection(self._apply_clahe(roi))

        if not candidates and self.enable_multiscale_retry:
            candidates = self._retry_multiscale(roi)

        if not candidates:
            return FaceDetectionResult(bbox=None, confidence=0.0, detected=False)

        # Filtrado de falsos positivos mediante validación geométrica de landmarks.
        valid_candidates = [c for c in candidates if self._passes_geometric_sanity(c)]
        pool = valid_candidates if valid_candidates else candidates

        best_bbox, best_conf, best_landmarks = max(pool, key=lambda c: c[1])
        detected = best_conf >= self.conf_threshold

        return FaceDetectionResult(
            bbox=best_bbox,
            confidence=best_conf,
            detected=detected,
            landmarks=best_landmarks,
            all_candidates=candidates,
        )

    # ------------------------------------------------------------------ #
    # Detección base
    # ------------------------------------------------------------------ #
    def _run_raw_detection(self, image: np.ndarray) -> List[tuple]:
        h, w = image.shape[:2]
        if h < 10 or w < 10:
            return []

        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            return []

        candidates = []
        for f in faces:
            x, y, fw, fh = f[0:4]
            landmarks = f[4:14].reshape(5, 2)
            confidence = float(f[-1])
            candidates.append(((x, y, fw, fh), confidence, landmarks))
        return candidates

    def _retry_multiscale(self, roi: np.ndarray) -> List[tuple]:
        """Reintenta la detección escalando la imagen (recupera rostros muy pequeños o muy grandes)."""
        scales = [1.5, 2.0, 0.75]
        h, w = roi.shape[:2]

        for scale in scales:
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w < 10 or new_h < 10 or new_w > 4000 or new_h > 4000:
                continue

            resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            candidates = self._run_raw_detection(resized)
            if candidates:
                # Reescala las coordenadas de vuelta al espacio original.
                rescaled = []
                for bbox, conf, landmarks in candidates:
                    x, y, fw, fh = bbox
                    rescaled_bbox = (x / scale, y / scale, fw / scale, fh / scale)
                    rescaled_landmarks = landmarks / scale
                    rescaled.append((rescaled_bbox, conf, rescaled_landmarks))
                logger.debug(f"Rostro recuperado con reintento multi-escala (factor={scale}).")
                return rescaled
        return []

    @staticmethod
    def _apply_clahe(image: np.ndarray) -> np.ndarray:
        """Ecualización adaptativa de contraste (CLAHE) para recuperar rostros en baja/alta iluminación."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        merged = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------ #
    # Filtro anti-falsos-positivos basado en geometría facial
    # ------------------------------------------------------------------ #
    @staticmethod
    def _passes_geometric_sanity(candidate: tuple) -> bool:
        """
        Valida que los 5 landmarks (ojos, nariz, boca) tengan una disposición anatómicamente
        plausible. Descarta detecciones espurias sobre texturas, ropa o fondos que YuNet
        a veces confunde con rostros de baja confianza.
        """
        bbox, _, landmarks = candidate
        _, _, fw, fh = bbox

        if fw <= 0 or fh <= 0 or landmarks is None or landmarks.shape != (5, 2):
            return False

        left_eye, right_eye, nose, mouth_left, mouth_right = landmarks

        eye_distance = np.linalg.norm(right_eye - left_eye)
        face_width = fw
        # La distancia interocular debe ser una fracción razonable del ancho del rostro.
        if not (0.20 * face_width <= eye_distance <= 0.75 * face_width):
            return False

        # La nariz debe estar verticalmente entre ojos y boca (no invertida).
        eyes_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        mouth_mid_y = (mouth_left[1] + mouth_right[1]) / 2.0
        if not (eyes_mid_y < nose[1] < mouth_mid_y):
            return False

        # El ojo izquierdo debe estar a la izquierda del derecho (orientación coherente).
        if left_eye[0] >= right_eye[0]:
            return False

        # Relación de aspecto plausible para un rostro (ni demasiado ancho ni demasiado alto).
        aspect_ratio = fw / fh
        if not (0.55 <= aspect_ratio <= 1.6):
            return False

        return True