"""Genera muestras sintéticas (rotaciones, desplazamientos, escalas, variaciones White-Patch) [REQ-ENT-02]."""
from typing import List
import cv2
import numpy as np
from preprocessing.white_patch_normalizer import apply_white_patch


class DataAugmentationEngine:
    def __init__(self, rotation_range_deg: float = 8.0, translation_ratio: float = 0.05,
                 scale_range: tuple[float, float] = (0.92, 1.08)):
        self.rotation_range_deg = rotation_range_deg
        self.translation_ratio = translation_ratio
        self.scale_range = scale_range

    def _random_affine(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        angle = np.random.uniform(-self.rotation_range_deg, self.rotation_range_deg)
        scale = np.random.uniform(*self.scale_range)
        tx = np.random.uniform(-self.translation_ratio, self.translation_ratio) * w
        ty = np.random.uniform(-self.translation_ratio, self.translation_ratio) * h

        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def _random_white_patch_variation(self, image: np.ndarray) -> np.ndarray:
        noisy = image.astype(np.float32)
        gain = np.random.uniform(0.85, 1.15, size=3)
        for c in range(3):
            noisy[..., c] *= gain[c]
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return apply_white_patch(noisy)

    def generate_synthetic_samples(self, image: np.ndarray, n_samples: int = 5) -> List[np.ndarray]:
        """Genera n_samples variaciones sintéticas de una imagen de clase minoritaria."""
        synthetic = []
        for _ in range(n_samples):
            augmented = self._random_affine(image)
            augmented = self._random_white_patch_variation(augmented)
            synthetic.append(augmented)
        return synthetic