"""Escribe físicamente las imágenes válidas (hasta 75 por Track-ID) en dataset/captures/<id>/ [REQ-CAP-02, REQ-CAP-06]."""
from pathlib import Path
from typing import Dict
import numpy as np
from utils.file_io_helpers import save_image, count_files_in_subdir
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_CAPTURES_PER_TRACK = 75  # [REQ-CAP-02]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class CaptureWriter:
    def __init__(self, base_dir: str = "dataset/captures", max_captures: int = MAX_CAPTURES_PER_TRACK):
        self.base_dir = Path(base_dir)
        self.max_captures = max_captures
        self._capture_counts: Dict[str, int] = {}

    def get_capture_count(self, identity: str) -> int:
        if identity not in self._capture_counts:
            self._capture_counts[identity] = count_files_in_subdir(self.base_dir, identity, IMAGE_EXTENSIONS)
        return self._capture_counts[identity]

    def has_quota_available(self, identity: str) -> bool:
        return self.get_capture_count(identity) < self.max_captures

    def write_capture(self, roi: np.ndarray, identity: str, frame_index: int) -> bool:
        """Escribe la imagen etiquetada con la identidad confirmada, respetando el cupo de 75."""
        if not self.has_quota_available(identity):
            return False

        current_count = self.get_capture_count(identity)
        filename = f"{identity}_{current_count:04d}_f{frame_index}.jpg"
        output_path = self.base_dir / identity / filename

        success = save_image(roi, output_path)
        if success:
            self._capture_counts[identity] = current_count + 1
            logger.debug(f"Captura guardada: {output_path}")
        return success