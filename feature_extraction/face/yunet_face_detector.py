"""Detector facial YuNet robusto: multi-escala, recuperación de casos difíciles y
descarte temprano de falsos positivos evidentes mediante landmarks [REQ-FAC-01].

Mejoras para entrenamiento:
  - Cascada de 8 variantes de preprocesamiento (CLAHE, gamma oscuro/claro,
    unsharp mask, bilateral filter, CLAHE+multiscale, multiscale puro, gamma+unsharp).
  - detect_training(): acepta candidatos con confianza >= MIN_TRAINING_CONF (0.25)
    para maximizar el número de muestras recuperadas sin degradar la inferencia RT.
  - Umbral interno bajado a 0.20 para mayor sensibilidad en la detección inicial.
"""
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple
import cv2
import numpy as np
import threading
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FACE_CONF_THRESHOLD = 0.6

# Umbral interno ajustado a 0.40 para purgar detecciones de baja calidad tempranamente;
# el filtrado fino real ocurre después, vía landmarks + confianza combinada.
INTERNAL_DETECTOR_THRESHOLD = 0.40
NMS_IOU_THRESHOLD = 0.3

# Umbral mínimo aceptable en modo entrenamiento (más bajo que inferencia).
MIN_TRAINING_CONF = 0.25


@dataclass
class FaceDetectionResult:
    bbox: Optional[tuple]          # (x, y, w, h)
    confidence: float
    detected: bool
    landmarks: Optional[np.ndarray] = None  # 5 puntos: ojo_izq, ojo_der, nariz, boca_izq, boca_der
    all_candidates: List[tuple] = field(default_factory=list)  # (bbox, confidence, landmarks)
    was_recovered: bool = False    # True si el rostro se encontró tras preprocesamiento adicional


class YuNetFaceDetector:
    """
    Detector facial YuNet con estrategia de máxima cobertura.

    Modo inferencia (detect):
      - Detecta con umbral interno bajo (recall alto).
      - Prueba múltiples escalas si la imagen es muy grande/pequeña.
      - Aplica CLAHE como primer retry si la detección inicial falla.
      - Filtra falsos positivos via validación geométrica de landmarks.

    Modo entrenamiento (detect_training):
      - Cascada ampliada de 8 variantes de preprocesamiento (CLAHE, gamma
        oscuro/claro, unsharp mask, bilateral filter, CLAHE+multiscale, etc.).
      - Acepta candidatos con confianza >= MIN_TRAINING_CONF aunque no alcancen
        el umbral de inferencia normal, maximizando muestras del dataset.
      - Marca was_recovered=True si fue necesario preprocesamiento adicional.
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
        self._lock = threading.Lock()

        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=INTERNAL_DETECTOR_THRESHOLD,
            nms_threshold=NMS_IOU_THRESHOLD,
            top_k=20,
        )

    # ------------------------------------------------------------------ #
    # API pública — inferencia en tiempo real (comportamiento original)
    # ------------------------------------------------------------------ #
    def detect(self, roi: np.ndarray) -> FaceDetectionResult:
        """Ejecuta la estrategia robusta de detección y devuelve el mejor candidato
        validado. Comportamiento idéntico al original para no afectar la inferencia RT."""
        if roi is None or roi.size == 0:
            return FaceDetectionResult(bbox=None, confidence=0.0, detected=False)

        candidates = self._run_raw_detection(roi)

        # Recuperación de falsos negativos: reintenta con variantes de la imagen.
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
    # API pública — entrenamiento (cascada extendida + umbral relajado)
    # ------------------------------------------------------------------ #
    def detect_training(self, roi: np.ndarray) -> FaceDetectionResult:
        """Variante para entrenamiento: maximiza recall probando 8 variantes de
        preprocesamiento antes de rendirse.

        Acepta candidatos con confianza >= MIN_TRAINING_CONF aunque no alcancen
        el umbral de inferencia normal. Devuelve was_recovered=True si fue
        necesario algún preprocesamiento adicional para encontrar el rostro.
        """
        if roi is None or roi.size == 0:
            return FaceDetectionResult(bbox=None, confidence=0.0, detected=False)

        # ── Etapa 1: intento base (sin preprocesamiento adicional) ──────────
        candidates = self._run_raw_detection(roi)
        was_recovered = False
        variant_used = "base"

        # ── Etapa 2: cascada extendida de 8 variantes (early-exit) ─────────
        if not candidates:
            for variant_name, variant_img in self._preprocessing_cascade(roi):
                candidates = self._run_raw_detection(variant_img)
                if candidates:
                    logger.debug(f"Rostro recuperado con variante '{variant_name}'.")
                    was_recovered = True
                    variant_used = variant_name
                    break

        if not candidates:
            return FaceDetectionResult(bbox=None, confidence=0.0, detected=False,
                                       was_recovered=False)

        # ── Etapa 3: filtrado geométrico (igual que en inferencia) ──────────
        valid_candidates = [c for c in candidates if self._passes_geometric_sanity(c)]
        pool = valid_candidates if valid_candidates else candidates

        best_bbox, best_conf, best_landmarks = max(pool, key=lambda c: c[1])

        # En entrenamiento aceptar si supera el umbral mínimo de entrenamiento.
        detected = best_conf >= MIN_TRAINING_CONF
        if detected and was_recovered:
            logger.debug(
                f"[training] Rostro recuperado vía '{variant_used}' "
                f"(conf={best_conf:.3f} >= {MIN_TRAINING_CONF})."
            )

        return FaceDetectionResult(
            bbox=best_bbox,
            confidence=best_conf,
            detected=detected,
            landmarks=best_landmarks,
            all_candidates=candidates,
            was_recovered=was_recovered,
        )

    # ------------------------------------------------------------------ #
    # Cascada extendida de preprocesamiento (solo para entrenamiento)
    # ------------------------------------------------------------------ #
    def _preprocessing_cascade(self, roi: np.ndarray) -> Iterator[Tuple[str, np.ndarray]]:
        """Genera variantes del ROI en orden de invasividad creciente.
        detect_training() para al primer intento que produzca candidatos (early-exit).
        """
        # 1. CLAHE — iluminación desigual / bajo contraste local
        yield "clahe", self._apply_clahe(roi)

        # 2. Gamma oscuro (γ=0.45) — sobreexposición / imagen lavada
        yield "gamma_dark", self._apply_gamma(roi, gamma=0.45)

        # 3. Gamma claro (γ=1.9) — subexposición / imagen muy oscura
        yield "gamma_bright", self._apply_gamma(roi, gamma=1.9)

        # 4. Unsharp mask — desenfoque / baja nitidez
        yield "unsharp", self._apply_unsharp_mask(roi)

        # 5. Bilateral filter — imagen con ruido, preservando bordes faciales
        yield "bilateral", self._apply_bilateral(roi)

        # 6. CLAHE + multiscale — iluminación deficiente Y rostro pequeño
        clahe_img = self._apply_clahe(roi)
        for scale in [1.5, 2.0, 0.75]:
            h, w = clahe_img.shape[:2]
            nw, nh = int(w * scale), int(h * scale)
            if 10 <= nw <= 4000 and 10 <= nh <= 4000:
                yield f"clahe_scale_{scale}", cv2.resize(
                    clahe_img, (nw, nh), interpolation=cv2.INTER_LINEAR
                )

        # 7. Multiscale puro — rostro pequeño sin problema de iluminación
        for scale in [1.5, 2.0, 3.0, 0.75]:
            h, w = roi.shape[:2]
            nw, nh = int(w * scale), int(h * scale)
            if 10 <= nw <= 4000 and 10 <= nh <= 4000:
                yield f"scale_{scale}", cv2.resize(
                    roi, (nw, nh), interpolation=cv2.INTER_LINEAR
                )

        # 8. Gamma claro + unsharp — caso extremo: imagen oscura Y borrosa
        yield "gamma_bright_unsharp", self._apply_unsharp_mask(
            self._apply_gamma(roi, gamma=1.9)
        )

    # ------------------------------------------------------------------ #
    # Detección base
    # ------------------------------------------------------------------ #
    def _run_raw_detection(self, image: np.ndarray) -> List[tuple]:
        h, w = image.shape[:2]
        if h < 10 or w < 10:
            return []

        with self._lock:
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

    # ------------------------------------------------------------------ #
    # Variantes de preprocesamiento
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_clahe(image: np.ndarray) -> np.ndarray:
        """Ecualización adaptativa de contraste (CLAHE) — iluminación deficiente/excesiva."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        merged = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
        """Corrección gamma: <1 oscurece (sobreexpuestos), >1 aclara (subexpuestos)."""
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
        )
        return cv2.LUT(image, table)

    @staticmethod
    def _apply_unsharp_mask(image: np.ndarray,
                             sigma: float = 1.5,
                             alpha: float = 1.5) -> np.ndarray:
        """Unsharp masking: realza bordes para mejorar detección en imágenes borrosas."""
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, alpha, blurred, -(alpha - 1.0), 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_bilateral(image: np.ndarray) -> np.ndarray:
        """Filtro bilateral: elimina ruido preservando bordes (imágenes granuladas)."""
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

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