"""Punto de entrada del Sistema de Identificación (ID) y Re-identificación (Re-ID).

Modos de ejecución:
    python main.py           → Ventana OpenCV clásica (comportamiento original)
    python main.py --ui      → Dashboard visual PyQt6 (interfaz enriquecida)
"""
import argparse
import sys
from pathlib import Path

import cv2

from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from core.video_stream_reader import VideoStreamReader
from core.frame_scheduler import FrameScheduler
from core.pipeline_orchestrator import PipelineOrchestrator
from detection_tracking.yolov8n_detector import YOLOv8nDetector

logger = get_logger("main")


def build_tracker(config: dict):
    algorithm = config.get("tracking", {}).get("algorithm", "bytetrack")
    frame_rate = config.get("tracking", {}).get("frame_rate", 30)

    if algorithm == "bytetrack":
        from detection_tracking.bytetrack_adapter import ByteTrackAdapter
        return ByteTrackAdapter(frame_rate=frame_rate)
    elif algorithm == "deepsort":
        from detection_tracking.deepsort_adapter import DeepSORTAdapter
        return DeepSORTAdapter()
    raise ValueError(f"Algoritmo de tracking no soportado: {algorithm}")


def draw_overlay(frame, results):
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r.bbox]
        color = (0, 200, 0) if r.identity != "Desconocido" else (0, 0, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{r.track_id} {r.identity} ({r.confidence:.2f}) [{r.branch_used}]"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def run_classic(config: dict):
    """Modo clásico: ventana OpenCV simple (comportamiento original)."""
    detector = YOLOv8nDetector(
        model_path=config["detection"]["yolo_model_path"],
        conf_threshold=config["detection"]["conf_threshold"],
        device=config["detection"]["device"],
    )
    tracker = build_tracker(config)
    orchestrator = PipelineOrchestrator(detector=detector, tracker=tracker)

    scheduler = FrameScheduler(target_fps=config["video"]["target_fps"])
    source = config["video"]["source"]
    last_results = []
    fps_delay = int(1000 / config["video"]["target_fps"])
    
    import threading
    import time

    # Variables compartidas
    current_frame_data = None
    lock = threading.Lock()

    def ai_worker():
        nonlocal last_results, current_frame_data
        last_processed_idx = -1
        
        while True:
            fd = None
            with lock:
                fd = current_frame_data
            
            # Solo procesar si hay un frame nuevo que no hayamos analizado antes
            if fd is not None and fd.frame_index > last_processed_idx:
                # Ejecutar YOLO + Re-ID (libera el GIL en funciones OpenCV/PyTorch)
                results = orchestrator.process_frame(fd.frame, fd.frame_index)
                
                with lock:
                    last_results = results
                last_processed_idx = fd.frame_index
            else:
                time.sleep(0.005)

    # Iniciar hilo de IA
    worker_thread = threading.Thread(target=ai_worker, daemon=True)
    worker_thread.start()

    with VideoStreamReader(source) as reader:
        for frame_data in reader.read_frames():
            # Actualizar el frame actual para que el hilo de IA lo tome cuando esté libre
            with lock:
                current_frame_data = frame_data
            
            # Dibujar siempre los últimos resultados conocidos (tracking visual instantáneo)
            with lock:
                annotated = draw_overlay(frame_data.frame.copy(), last_results)
                
            cv2.imshow("ID/Re-ID System", annotated)

            # Sincronizar con el framerate original del video (ej. 33ms para 30fps)
            if cv2.waitKey(fps_delay) & 0xFF == ord("q"):
                logger.info("Interrupción manual por el usuario.")
                break

    cv2.destroyAllWindows()


def run_dashboard(config: dict):
    """Modo dashboard: interfaz visual PyQt6 con stats en tiempo real."""
    from ui.dashboard import launch_dashboard
    return launch_dashboard(config)


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Identificación y Re-identificación de Personas"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Lanza el dashboard visual PyQt6 en lugar de la ventana OpenCV clásica.",
    )
    args = parser.parse_args()

    config = ConfigLoader.load("config.yaml")

    if args.ui:
        logger.info("Iniciando en modo Dashboard PyQt6.")
        return run_dashboard(config)
    else:
        logger.info("Iniciando en modo OpenCV clásico.")
        return run_classic(config)


if __name__ == "__main__":
    sys.exit(main())