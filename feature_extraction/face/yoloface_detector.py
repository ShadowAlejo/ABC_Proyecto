"""Detector facial YoloFace robusto.
Reemplaza a YuNet conservando los contratos de inferencia y entrenamiento (Cascada de preprocesamiento).
"""
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple
import cv2
import numpy as np
import threading
from ultralytics import YOLO
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FACE_CONF_THRESHOLD = 0.50
INTERNAL_DETECTOR_THRESHOLD = 0.40
MIN_TRAINING_CONF = 0.25


@dataclass
class FaceDetectionResult:
    bbox: Optional[tuple]          # (x, y, w, h)
    confidence: float
    detected: bool
    landmarks: Optional[np.ndarray] = None  # 5 puntos: ojo_izq, ojo_der, nariz, boca_izq, boca_der
    all_candidates: List[tuple] = field(default_factory=list)  # (bbox, confidence, landmarks)
    was_recovered: bool = False


class YoloFaceDetector:
    """Wrapper para modelo YOLOv8-face conservando el contrato de la API YuNet."""

    def __init__(self, model_path: str = "yolov8n-face.pt",
                 conf_threshold: float = DEFAULT_FACE_CONF_THRESHOLD,
                 enable_multiscale_retry: bool = True,
                 enable_contrast_retry: bool = True):
        self.conf_threshold = conf_threshold
        self.enable_multiscale_retry = enable_multiscale_retry
        self.enable_contrast_retry = enable_contrast_retry
        self._lock = threading.Lock()

        # Cargar modelo YOLO
        self.model = YOLO(model_path)

    def detect(self, roi: np.ndarray, is_enhanced: bool = False) -> FaceDetectionResult:
        """Inferencia tiempo real."""
        if roi is None or roi.size == 0:
            return FaceDetectionResult(bbox=None, confidence=0.0, detected=False)

        candidates = self._run_raw_detection(roi)

        if not candidates and self.enable_contrast_retry and not is_enhanced:
            candidates = self._run_raw_detection(self._apply_clahe(roi))

        if not candidates and self.enable_multiscale_retry:
            candidates = self._retry_multiscale(roi)

        if not candidates:
            return FaceDetectionResult(bbox=None, confidence=0.0, detected=False)

        valid_candidates = [c for c in candidates if self._passes_geometric_sanity(c, c[1])]
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

    def detect_batch(self, rois: List[np.ndarray], is_enhanced_list: List[bool]) -> List[FaceDetectionResult]:
        """Inferencia en lote (Batch Inference)."""
        if not rois:
            return []
            
        results = []
        # TODO: A simple loop that calls detect for now, or true batching if needed.
        # Since YOLOv8 supports list of images directly:
        raw_results = self.model(rois, verbose=False)
        
        for i, (roi, r, is_enhanced) in enumerate(zip(rois, raw_results, is_enhanced_list)):
            if roi is None or roi.size == 0:
                results.append(FaceDetectionResult(bbox=None, confidence=0.0, detected=False))
                continue
                
            candidates = self._parse_yolo_result(r, 1.0)
            
            if not candidates and self.enable_contrast_retry and not is_enhanced:
                candidates = self._run_raw_detection(self._apply_clahe(roi))
            
            if not candidates and self.enable_multiscale_retry:
                candidates = self._retry_multiscale(roi)
                
            if not candidates:
                results.append(FaceDetectionResult(bbox=None, confidence=0.0, detected=False))
                continue
                
            valid_candidates = [c for c in candidates if self._passes_geometric_sanity(c, c[1])]
            pool = valid_candidates if valid_candidates else candidates
            
            best_bbox, best_conf, best_landmarks = max(pool, key=lambda c: c[1])
            detected = best_conf >= self.conf_threshold
            
            results.append(FaceDetectionResult(
                bbox=best_bbox,
                confidence=best_conf,
                detected=detected,
                landmarks=best_landmarks,
                all_candidates=candidates
            ))
            
        return results

    def detect_training(self, roi: np.ndarray) -> FaceDetectionResult:
        """Cascada extendida de preprocesamiento para maximizar recall en entrenamiento."""
        if roi is None or roi.size == 0:
            return FaceDetectionResult(bbox=None, confidence=0.0, detected=False)

        candidates = self._run_raw_detection(roi)
        was_recovered = False
        variant_used = "base"

        if not candidates:
            for variant_name, variant_img, scale_factor in self._preprocessing_cascade(roi):
                raw_cands = self._run_raw_detection(variant_img)
                if raw_cands:
                    if scale_factor != 1.0:
                        inv_s = 1.0 / scale_factor
                        rescaled = []
                        for bbox, conf, landmarks in raw_cands:
                            x, y, fw, fh = bbox
                            rescaled_bbox = (x * inv_s, y * inv_s, fw * inv_s, fh * inv_s)
                            rescaled_landmarks = (landmarks * inv_s).astype(np.float32) if landmarks is not None else None
                            rescaled.append((rescaled_bbox, conf, rescaled_landmarks))
                        candidates = rescaled
                    else:
                        candidates = raw_cands

                    logger.debug(f"Rostro recuperado con variante '{variant_name}'.")
                    was_recovered = True
                    variant_used = variant_name
                    break

        if not candidates:
            return FaceDetectionResult(bbox=None, confidence=0.0, detected=False, was_recovered=False)

        valid_candidates = [c for c in candidates if self._passes_geometric_sanity(c, c[1])]
        pool = valid_candidates if valid_candidates else candidates

        best_bbox, best_conf, best_landmarks = max(pool, key=lambda c: c[1])
        detected = best_conf >= MIN_TRAINING_CONF

        if detected and was_recovered:
            logger.debug(f"[training] Rostro recuperado vía '{variant_used}' (conf={best_conf:.3f}).")

        return FaceDetectionResult(
            bbox=best_bbox,
            confidence=best_conf,
            detected=detected,
            landmarks=best_landmarks,
            all_candidates=candidates,
            was_recovered=was_recovered,
        )

    def _preprocessing_cascade(self, roi: np.ndarray) -> Iterator[Tuple[str, np.ndarray, float]]:
        h, w = roi.shape[:2]
        is_small = min(h, w) < 128
        yield "clahe", self._apply_clahe(roi), 1.0
        yield "gamma_dark", self._apply_gamma(roi, gamma=0.45), 1.0
        yield "gamma_bright", self._apply_gamma(roi, gamma=1.9), 1.0
        yield "unsharp", self._apply_unsharp_mask(roi), 1.0
        yield "bilateral", self._apply_bilateral(roi), 1.0

        clahe_img = self._apply_clahe(roi)
        scales_to_try = [1.25, 1.5, 0.75] if is_small else [0.75]
        for scale in scales_to_try:
            nw, nh = int(w * scale), int(h * scale)
            if 16 <= nw <= 2048 and 16 <= nh <= 2048:
                yield f"clahe_scale_{scale}", cv2.resize(clahe_img, (nw, nh), interpolation=cv2.INTER_LINEAR), scale

        for scale in scales_to_try:
            nw, nh = int(w * scale), int(h * scale)
            if 16 <= nw <= 2048 and 16 <= nh <= 2048:
                yield f"scale_{scale}", cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_LINEAR), scale

        yield "gamma_bright_unsharp", self._apply_unsharp_mask(self._apply_gamma(roi, gamma=1.9)), 1.0

    def _run_raw_detection(self, image: np.ndarray) -> List[tuple]:
        h, w = image.shape[:2]
        if h < 10 or w < 10:
            return []

        scale = 1.0
        max_dim = 1280
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            proc_w, proc_h = int(w * scale), int(h * scale)
            proc_image = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            proc_image = image

        with self._lock:
            results = self.model(proc_image, verbose=False, conf=INTERNAL_DETECTOR_THRESHOLD)
        
        if len(results) == 0:
            return []
            
        return self._parse_yolo_result(results[0], 1.0 / scale)

    def _parse_yolo_result(self, result, inv_scale: float = 1.0) -> List[tuple]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
            
        keypoints = result.keypoints

        candidates = []
        for i in range(len(boxes)):
            box = boxes[i].xyxy[0].cpu().numpy()
            conf = float(boxes[i].conf[0].cpu().numpy())
            
            x1, y1, x2, y2 = box
            fw = x2 - x1
            fh = y2 - y1
            
            rx = float(x1 * inv_scale)
            ry = float(y1 * inv_scale)
            rfw = float(fw * inv_scale)
            rfh = float(fh * inv_scale)
            
            kpts = None
            if keypoints is not None and hasattr(keypoints, 'xy'):
                kp_array = keypoints.xy[i].cpu().numpy()
                if len(kp_array) >= 5:
                    kpts = (kp_array[:5] * inv_scale).astype(np.float32)

            candidates.append(((rx, ry, rfw, rfh), conf, kpts))
            
        return candidates

    def _retry_multiscale(self, roi: np.ndarray) -> List[tuple]:
        h, w = roi.shape[:2]
        scales = [1.25, 1.5, 0.75] if min(h, w) < 128 else [0.75]

        for scale in scales:
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w < 16 or new_h < 16 or new_w > 2048 or new_h > 2048:
                continue

            resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            candidates = self._run_raw_detection(resized)
            if candidates:
                rescaled = []
                for bbox, conf, landmarks in candidates:
                    x, y, fw, fh = bbox
                    rescaled_bbox = (x / scale, y / scale, fw / scale, fh / scale)
                    rescaled_landmarks = landmarks / scale if landmarks is not None else None
                    rescaled.append((rescaled_bbox, conf, rescaled_landmarks))
                return rescaled
        return []

    @staticmethod
    def _apply_clahe(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        merged = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image, table)

    @staticmethod
    def _apply_unsharp_mask(image: np.ndarray, sigma: float = 1.5, alpha: float = 1.5) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, alpha, blurred, -(alpha - 1.0), 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_bilateral(image: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    @staticmethod
    def _passes_geometric_sanity(candidate: tuple, conf: float) -> bool:
        if conf >= 0.75:
            return True
            
        bbox, _, landmarks = candidate
        _, _, fw, fh = bbox

        if fw <= 0 or fh <= 0 or landmarks is None or landmarks.shape != (5, 2):
            return False

        left_eye, right_eye, nose, mouth_left, mouth_right = landmarks

        eye_distance = np.linalg.norm(right_eye - left_eye)
        if not (0.15 * fw <= eye_distance <= 0.75 * fw):
            return False

        eyes_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        mouth_mid_y = (mouth_left[1] + mouth_right[1]) / 2.0
        if not (eyes_mid_y < nose[1] < mouth_mid_y):
            return False

        if left_eye[0] >= right_eye[0]:
            return False

        aspect_ratio = fw / fh
        if not (0.50 <= aspect_ratio <= 2.0):
            return False

        return True
