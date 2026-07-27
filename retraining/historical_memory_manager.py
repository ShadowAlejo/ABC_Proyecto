"""Conserva una proporción fija del dataset original junto a capturas recientes [REQ-ENT-04]."""
from pathlib import Path
from typing import List
import numpy as np
from utils.file_io_helpers import list_files, load_image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class HistoricalMemoryManager:
    def __init__(self, raw_images_dir: str = "dataset/raw_images",
                 captures_dir: str = "dataset/captures",
                 historical_ratio: float = 0.3):
        """
        historical_ratio: proporción del dataset original a conservar respecto al total combinado,
        para evitar el desaprendizaje progresivo de las características estructurales del sujeto.
        """
        self.raw_images_dir = Path(raw_images_dir)
        self.captures_dir = Path(captures_dir)
        self.historical_ratio = historical_ratio

    def build_training_set(self, subject_id: str) -> List[np.ndarray]:
        """Combina imágenes históricas + capturas recientes manteniendo la proporción configurada."""
        historical_paths = list_files(self.raw_images_dir / subject_id, IMAGE_EXTENSIONS)
        recent_paths = list_files(self.captures_dir / subject_id, IMAGE_EXTENSIONS)

        n_recent = len(recent_paths)
        n_historical_target = int((self.historical_ratio * n_recent) / max(1 - self.historical_ratio, 1e-6))
        n_historical_target = max(n_historical_target, 1) if historical_paths else 0

        selected_historical = historical_paths[:n_historical_target] if historical_paths else []

        combined_paths = selected_historical + recent_paths
        images = [img for p in combined_paths if (img := load_image(p)) is not None]
        return images