"""Genera muestras sintéticas (Jittering de landmarks, Random Cutout, Variaciones Fotométricas y White-Patch) [REQ-ENT-02]."""
from typing import List, Sequence
import cv2
import numpy as np
from preprocessing.white_patch_normalizer import apply_white_patch


class DataAugmentationEngine:
    def __init__(self, rotation_range_deg: float = 8.0, translation_ratio: float = 0.05,
                 scale_range: tuple[float, float] = (0.92, 1.08)):
        self.rotation_range_deg = rotation_range_deg
        self.translation_ratio = translation_ratio
        self.scale_range = scale_range

    def generate_landmark_jitter(self, landmarks: np.ndarray | Sequence, n_samples: int = 2,
                                 max_offset: float = 2.0) -> List[np.ndarray]:
        """Aplica perturbaciones aleatorias a las coordenadas de los landmarks en la imagen original.
        
        Evita artefactos de borde y desalineaciones de parches locales al realizar la alineación afín.
        """
        pts = np.array(landmarks, dtype=np.float32)
        jittered_list = []
        for _ in range(n_samples):
            noise = np.random.uniform(-max_offset, max_offset, size=pts.shape).astype(np.float32)
            jittered_list.append(pts + noise)
        return jittered_list

    def jitter_landmarks(self, landmarks: np.ndarray | Sequence, scale: float = 0.02) -> np.ndarray:
        """Alias para perturbación individual de landmarks."""
        pts = np.array(landmarks, dtype=np.float32)
        offset = float(np.max(np.ptp(pts, axis=0))) * scale if len(pts) > 0 else 2.0
        noise = np.random.uniform(-offset, offset, size=pts.shape).astype(np.float32)
        return pts + noise

    def apply_random_erasing(self, face_bgr_96x96: np.ndarray, **kwargs) -> np.ndarray:
        """Alias para apply_random_cutout."""
        return self.apply_random_cutout(face_bgr_96x96)

    def apply_random_cutout(self, face_bgr_96x96: np.ndarray, mode: str = "random") -> np.ndarray:
        """Aplica oclusión sintética (Random Cutout / Erasing) sobre un rostro normalizado 96x96.
        
        Permite entrenar al StackingClassifier (Nivel 2) a desconfiar de subespacios ocluidos.
        """
        img = face_bgr_96x96.copy()
        h, w = img.shape[:2]
        fill_val = [128, 128, 128]

        chosen_mode = mode
        if chosen_mode == "random":
            chosen_mode = np.random.choice(["eyes", "mouth", "patch"], p=[0.4, 0.4, 0.2])

        if chosen_mode == "eyes":
            # Oclusión en región ocular / cejas
            img[24:46, 18:78] = fill_val
        elif chosen_mode == "mouth":
            # Oclusión en región bucal / nasal inferior
            img[56:84, 24:72] = fill_val
        else:
            # Parche rectangular aleatorio
            ph = np.random.randint(14, 24)
            pw = np.random.randint(14, 24)
            y = np.random.randint(10, max(11, h - ph - 10))
            x = np.random.randint(10, max(11, w - pw - 10))
            img[y:y+ph, x:x+pw] = fill_val

        return img

    def _random_white_patch_variation(self, image: np.ndarray) -> np.ndarray:
        """Aplica variaciones de ganancia por canal seguidas de White-Patch."""
        noisy = image.astype(np.float32)
        gain = np.random.uniform(0.85, 1.15, size=3)
        for c in range(3):
            noisy[..., c] *= gain[c]
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return apply_white_patch(noisy)

    def generate_reid_photometric_samples(self, image: np.ndarray, n_samples: int = 5) -> List[np.ndarray]:
        """Genera muestras sintéticas para Re-ID corporal SIN rotaciones para preservar bandas LBP.
        
        Aplica variaciones cromáticas (White-Patch), ajustes de contraste adaptativo (CLAHE) y
        escalados/traslaciones sutiles estrictamente verticales.
        """
        synthetic = []
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        for _ in range(n_samples):
            # Escala y traslación sutil (SIN rotación)
            scale = np.random.uniform(0.97, 1.03)
            tx = np.random.uniform(-0.02, 0.02) * w
            ty = np.random.uniform(-0.02, 0.02) * h

            matrix = cv2.getRotationMatrix2D(center, 0.0, scale)
            matrix[0, 2] += tx
            matrix[1, 2] += ty

            aug = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

            # Perturbación fotométrica (White-Patch + CLAHE LAB aleatorio)
            aug_wp = self._random_white_patch_variation(aug)
            lab = cv2.cvtColor(aug_wp, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clip_limit = float(np.random.uniform(1.5, 3.0))
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            
            # Ganancia sutil de brillo en luminancia
            l_gain = float(np.random.uniform(0.92, 1.08))
            l_final = np.clip(l_clahe.astype(np.float32) * l_gain, 0, 255).astype(np.uint8)
            
            aug_final = cv2.cvtColor(cv2.merge((l_final, a, b)), cv2.COLOR_LAB2BGR)
            synthetic.append(aug_final)

        return synthetic

    def generate_synthetic_samples(self, image: np.ndarray, n_samples: int = 5) -> List[np.ndarray]:
        """Alias retrocompatible: redirige a augmentación fotométrica sin rotación para Re-ID."""
        return self.generate_reid_photometric_samples(image, n_samples=n_samples)