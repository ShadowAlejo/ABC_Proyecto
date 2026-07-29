"""Escribe físicamente las imágenes válidas (hasta 75 por Track-ID) en dataset/captures/<id>/ [REQ-CAP-02, REQ-CAP-06]."""
from pathlib import Path
from typing import Dict
import numpy as np
import concurrent.futures
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
        # Hilo dedicado a guardar imágenes para evitar bloqueo de I/O
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def get_capture_count(self, identity: str) -> int:
        if identity not in self._capture_counts:
            self._capture_counts[identity] = count_files_in_subdir(self.base_dir, identity, IMAGE_EXTENSIONS)
        return self._capture_counts[identity]

    def has_quota_available(self, identity: str) -> bool:
        return self.get_capture_count(identity) < self.max_captures

    def write_capture(self, roi: np.ndarray, identity: str, frame_index: int) -> bool:
        """Escribe la imagen de forma asíncrona, respetando el cupo de 75."""
        if not self.has_quota_available(identity):
            return False

        current_count = self.get_capture_count(identity)
        filename = f"{identity}_{current_count:04d}_f{frame_index}.jpg"
        output_path = self.base_dir / identity / filename

        # Incremento optimista para evitar sobre-escrituras si se encolan muchas tareas rápido
        self._capture_counts[identity] = current_count + 1

        def _async_write():
            success = save_image(roi, output_path)
            if success:
                logger.debug(f"Captura guardada asíncronamente: {output_path}")
            else:
                self._capture_counts[identity] -= 1  # Revertir si falla

        self.executor.submit(_async_write)
        return True