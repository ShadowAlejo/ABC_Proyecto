"""Worker thread que ejecuta el pipeline de video en segundo plano y emite señales Qt."""
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)


class PipelineWorker(QThread):
    """Ejecuta detección + tracking + ID/Re-ID en un hilo separado.

    Señales emitidas:
        frame_ready   — frame BGR anotado + lista de TrackResult
        stats_update  — fps real, procesados, saltados, total_id, total_reid
        event_logged  — string de evento para el log de la UI
        error_occurred — mensaje de error crítico
        finished_ok   — pipeline terminó limpiamente
    """

    frame_ready = pyqtSignal(object, list)          # (np.ndarray BGR, List[TrackResult])
    stats_update = pyqtSignal(float, int, int, int, int)  # fps, proc, skipped, id_count, reid_count
    event_logged = pyqtSignal(str)                  # mensaje de evento
    error_occurred = pyqtSignal(str)                # mensaje de error
    finished_ok = pyqtSignal()

    def __init__(self, source, config: dict, parent=None):
        super().__init__(parent)
        self.source = source
        self.config = config
        self._paused = False
        self._stop_requested = False

        # Contadores de rama acumulados
        self.id_count = 0
        self.reid_count = 0

    # ------------------------------------------------------------------ control
    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stop_requested = True
        self._paused = False

    # ------------------------------------------------------------------ helpers
    def _build_tracker(self):
        algorithm = self.config.get("tracking", {}).get("algorithm", "deepsort")
        frame_rate = self.config.get("tracking", {}).get("frame_rate", 30)
        if algorithm == "bytetrack":
            from detection_tracking.bytetrack_adapter import ByteTrackAdapter
            return ByteTrackAdapter(frame_rate=frame_rate)
        from detection_tracking.deepsort_adapter import DeepSORTAdapter
        return DeepSORTAdapter()

    @staticmethod
    def _draw_overlay(frame: np.ndarray, results) -> np.ndarray:
        """Dibuja bboxes y etiquetas enriquecidas sobre el frame."""
        overlay = frame.copy()
        for r in results:
            x1, y1, x2, y2 = [int(v) for v in r.bbox]
            is_unknown = r.identity == "Desconocido"
            is_id_branch = r.branch_used == "ID"

            # Color por rama: ID=verde vivo, Re-ID=naranja, Desconocido=rojo
            if is_unknown:
                color = (60, 60, 220)          # rojo
                border_thick = 1
            elif is_id_branch:
                color = (50, 205, 50)          # verde
                border_thick = 2
            else:
                color = (30, 165, 255)         # naranja
                border_thick = 2

            # Caja con esquinas redondeadas (simuladas con líneas)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, border_thick)

            # Fondo semitransparente para la etiqueta
            label = f"ID:{r.track_id}  {r.identity}  {r.confidence:.2f}  [{r.branch_used}]"
            if r.captured:
                label += "  \U0001f4f8"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
            label_y = max(y1 - 4, th + 4)
            cv2.rectangle(overlay, (x1, label_y - th - 4), (x1 + tw + 6, label_y + 2), color, -1)
            cv2.putText(overlay, label, (x1 + 3, label_y - 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Fusión semitransparente
        result = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        return result

    # ------------------------------------------------------------------ main loop
    def run(self):
        from detection_tracking.yolov8n_detector import YOLOv8nDetector
        from core.pipeline_orchestrator import PipelineOrchestrator
        from core.frame_scheduler import FrameScheduler
        from core.video_stream_reader import VideoStreamReader

        try:
            detector = YOLOv8nDetector(
                model_path=self.config["detection"]["yolo_model_path"],
                conf_threshold=self.config["detection"]["conf_threshold"],
                device=self.config["detection"]["device"],
            )
            tracker = self._build_tracker()
            orchestrator = PipelineOrchestrator(detector=detector, tracker=tracker)
            scheduler = FrameScheduler(target_fps=self.config["video"]["target_fps"])
        except Exception as exc:
            self.error_occurred.emit(f"Error al inicializar el pipeline: {exc}")
            return

        fps_timer = time.perf_counter()
        fps_frame_count = 0
        current_fps = 0.0

        try:
            with VideoStreamReader(self.source) as reader:
                self.event_logged.emit("▶ Pipeline iniciado")
                for frame_data in reader.read_frames():
                    if self._stop_requested:
                        break

                    while self._paused and not self._stop_requested:
                        time.sleep(0.05)

                    if self._stop_requested:
                        break

                    def _process(fd):
                        nonlocal current_fps, fps_frame_count
                        results = orchestrator.process_frame(fd.frame, fd.frame_index)

                        # Conteo por rama
                        for r in results:
                            if r.branch_used == "ID":
                                self.id_count += 1
                            else:
                                self.reid_count += 1
                            if r.captured:
                                self.event_logged.emit(
                                    f"\U0001f4f8 Captura guardada — Track {r.track_id} ({r.identity})"
                                )

                        annotated = self._draw_overlay(fd.frame.copy(), results)
                        self.frame_ready.emit(annotated, results)

                        # FPS real
                        fps_frame_count += 1
                        elapsed = time.perf_counter() - fps_timer
                        if elapsed >= 1.0:
                            current_fps = fps_frame_count / elapsed
                            self.stats_update.emit(
                                current_fps,
                                scheduler.stats.processed,
                                scheduler.stats.skipped,
                                self.id_count,
                                self.reid_count,
                            )
                            fps_frame_count = 0

                    scheduler.dispatch(frame_data, _process)

            self.event_logged.emit("⏹ Video finalizado")
            self.finished_ok.emit()

        except Exception as exc:
            logger.exception("Error en el worker thread")
            self.error_occurred.emit(str(exc))
