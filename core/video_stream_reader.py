"""Función pura para abrir y leer streams de video (archivo o cámara)."""
from dataclasses import dataclass
from typing import Iterator, Union, Optional
import cv2
import numpy as np
import threading
import queue
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameData:
    frame: np.ndarray
    frame_index: int
    timestamp_ms: float


class VideoStreamReader:
    """Abre un stream de video (archivo o índice de cámara) y entrega fotogramas normalizados.
    Utiliza un hilo en segundo plano para leer los fotogramas y evitar cuellos de botella de I/O."""

    def __init__(self, source: Union[str, int], queue_size: int = 128):
        self.source = source
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0
        self.queue_size = queue_size
        self.queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stopped = False

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            logger.error(f"No se pudo abrir el stream de video: {self.source}")
            return False
        logger.info(f"Stream de video abierto: {self.source}")
        return True

    def _read_loop(self):
        while not self._stopped:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                logger.info("Fin del flujo de video alcanzado (Hilo Lector).")
                # Put a Sentinel to indicate end of stream, use timeout to exit if stopped
                try:
                    self.queue.put(None, timeout=1.0)
                except queue.Full:
                    pass
                break

            frame = frame.astype(np.uint8)
            timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_data = FrameData(frame=frame, frame_index=self._frame_index, timestamp_ms=timestamp_ms)
            
            # Bloquea si la cola está llena, regulando la lectura (backpressure)
            while not self._stopped:
                try:
                    self.queue.put(frame_data, timeout=0.5)
                    break
                except queue.Full:
                    continue
            self._frame_index += 1

    def start_thread(self):
        if self._cap is None and not self.open():
            return False
            
        self._stopped = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def read_frames(self) -> Iterator[FrameData]:
        """Generador que entrega fotogramas secuenciales normalizados extraídos de la cola."""
        if self._thread is None:
            if not self.start_thread():
                return
                
        while not self._stopped:
            try:
                frame_data = self.queue.get(timeout=0.5)
                if frame_data is None: # Sentinel
                    break
                yield frame_data
            except queue.Empty:
                if not self._thread.is_alive():
                    break

    def get_fps(self) -> float:
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 0.0

    def release(self) -> None:
        self._stopped = True
        if self._cap is not None:
            # Desatascar el hilo si está bloqueado en put
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
                    
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.0)
                
            self._cap.release()
            logger.info("Stream de video liberado.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()