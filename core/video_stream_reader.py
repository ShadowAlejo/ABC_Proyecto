"""Función pura para abrir y leer streams de video (archivo o cámara)."""
from dataclasses import dataclass
from typing import Iterator, Union
import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameData:
    frame: np.ndarray
    frame_index: int
    timestamp_ms: float


class VideoStreamReader:
    """Abre un stream de video (archivo o índice de cámara) y entrega fotogramas normalizados."""

    def __init__(self, source: Union[str, int]):
        self.source = source
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            logger.error(f"No se pudo abrir el stream de video: {self.source}")
            return False
        logger.info(f"Stream de video abierto: {self.source}")
        return True

    def read_frames(self) -> Iterator[FrameData]:
        """Generador que entrega fotogramas secuenciales normalizados hasta el fin del flujo."""
        if self._cap is None and not self.open():
            return

        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                logger.info("Fin del flujo de video alcanzado.")
                break

            frame = frame.astype(np.uint8)
            timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            yield FrameData(frame=frame, frame_index=self._frame_index, timestamp_ms=timestamp_ms)
            self._frame_index += 1

    def get_fps(self) -> float:
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 0.0

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            logger.info("Stream de video liberado.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()