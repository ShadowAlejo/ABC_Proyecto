"""Dashboard principal del sistema ID/Re-ID — PyQt6."""
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import (
    QColor, QFont, QIcon, QImage, QPainter, QPalette,
    QPixmap, QBrush, QLinearGradient,
)
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSplitter, QStatusBar, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QFrame,
    QHeaderView, QGraphicsDropShadowEffect,
)

from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from ui.worker_thread import PipelineWorker

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  PALETA & CONSTANTES DE DISEÑO
# ═══════════════════════════════════════════════════════════════════════════════
BG_DARK       = "#0D1117"
BG_PANEL      = "#161B22"
BG_CARD       = "#1C2128"
BG_HOVER      = "#21262D"
ACCENT_GREEN  = "#39D353"
ACCENT_ORANGE = "#FF8C00"
ACCENT_BLUE   = "#58A6FF"
ACCENT_RED    = "#F85149"
ACCENT_PURPLE = "#BC8CFF"
TEXT_PRIMARY  = "#E6EDF3"
TEXT_MUTED    = "#7D8590"
BORDER_COLOR  = "#30363D"


def _apply_dark_palette(app: QApplication):
    """Aplica la paleta oscura global a la aplicación."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(BG_DARK))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base,            QColor(BG_PANEL))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.Text,            QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button,          QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(ACCENT_BLUE))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(BG_DARK))
    palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.ToolTipText,     QColor(TEXT_PRIMARY))
    app.setPalette(palette)


# ═══════════════════════════════════════════════════════════════════════════════
#  WIDGETS AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════════
class StyledCard(QFrame):
    """Panel con fondo semitransparente, borde y radio de esquinas."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            StyledCard {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
            }}
        """)


class StyledButton(QPushButton):
    def __init__(self, text: str, accent: str = ACCENT_BLUE, parent=None):
        super().__init__(text, parent)
        self._accent = accent
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_CARD};
                color: {accent};
                border: 1px solid {accent};
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {accent};
                color: {BG_DARK};
            }}
            QPushButton:pressed {{
                background-color: {accent};
                opacity: 0.8;
            }}
            QPushButton:disabled {{
                color: {TEXT_MUTED};
                border-color: {BORDER_COLOR};
            }}
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class BranchBar(QWidget):
    """Barras de progreso personalizadas para ID vs Re-ID."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._id_label   = QLabel("ID Branch      0")
        self._reid_label = QLabel("Re-ID Branch   0")
        self._id_bar     = QProgressBar()
        self._reid_bar   = QProgressBar()

        for lbl in (self._id_label, self._reid_label):
            lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")

        bar_style = """
            QProgressBar {{
                background: {bg};
                border: none;
                border-radius: 4px;
                height: 12px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                border-radius: 4px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {c1}, stop:1 {c2});
            }}
        """
        self._id_bar.setStyleSheet(
            bar_style.format(bg=BG_PANEL, c1="#2EA043", c2=ACCENT_GREEN)
        )
        self._reid_bar.setStyleSheet(
            bar_style.format(bg=BG_PANEL, c1="#B45309", c2=ACCENT_ORANGE)
        )
        for bar in (self._id_bar, self._reid_bar):
            bar.setMaximum(100)
            bar.setTextVisible(False)
            bar.setValue(0)

        layout.addWidget(self._id_label)
        layout.addWidget(self._id_bar)
        layout.addWidget(self._reid_label)
        layout.addWidget(self._reid_bar)

    def update_counts(self, id_count: int, reid_count: int):
        total = id_count + reid_count
        if total == 0:
            self._id_bar.setValue(0)
            self._reid_bar.setValue(0)
            self._id_label.setText(f"ID Branch      0  (0%)")
            self._reid_label.setText(f"Re-ID Branch   0  (0%)")
            return
        id_pct   = int(id_count   / total * 100)
        reid_pct = int(reid_count / total * 100)
        self._id_bar.setValue(id_pct)
        self._reid_bar.setValue(reid_pct)
        self._id_label.setText(f"ID Branch      {id_count:,}  ({id_pct}%)")
        self._reid_label.setText(f"Re-ID Branch   {reid_count:,}  ({reid_pct}%)")


class EventLog(QScrollArea):
    """Log scrollable de eventos con timestamp."""
    MAX_EVENTS = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet(f"""
            QScrollArea {{ border: none; background: {BG_PANEL}; }}
            QScrollBar:vertical {{
                background: {BG_PANEL}; width: 6px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_COLOR}; border-radius: 3px; min-height: 20px;
            }}
        """)

        container = QWidget()
        container.setStyleSheet(f"background: {BG_PANEL};")
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(6, 4, 6, 4)
        self._layout.setSpacing(2)
        self._layout.addStretch()
        self.setWidget(container)
        self._entries: List[QLabel] = []

    def add_event(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        full = f"<span style='color:{TEXT_MUTED};'>[{ts}]</span> {msg}"
        lbl = QLabel(full)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px; padding: 1px 0;")
        lbl.setTextFormat(Qt.TextFormat.RichText)

        # Inserta antes del stretch
        self._layout.insertWidget(self._layout.count() - 1, lbl)
        self._entries.append(lbl)

        # Truncar si hay demasiados
        if len(self._entries) > self.MAX_EVENTS:
            old = self._entries.pop(0)
            self._layout.removeWidget(old)
            old.deleteLater()

        # Auto-scroll al final
        QTimer.singleShot(30, lambda: self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()
        ))


# ═══════════════════════════════════════════════════════════════════════════════
#  VENTANA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
class DashboardWindow(QMainWindow):
    """Dashboard principal del sistema ID/Re-ID en tiempo real."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._worker: Optional[PipelineWorker] = None
        self._is_paused = False
        self._frame_count = 0
        self._last_results = []

        self.setWindowTitle("ID / Re-ID System — Dashboard")
        self.setMinimumSize(1280, 760)
        self.resize(1440, 840)

        self._build_ui()
        self._apply_styles()
        self._check_models()

    # ------------------------------------------------------------------ UI build
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_topbar())

        # Splitter principal
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER_COLOR}; }}")

        splitter.addWidget(self._build_video_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([900, 380])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, 1)
        root.addWidget(self._build_statusbar_widget())

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(52)
        bar.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #0D1117, stop:0.5 #161B22, stop:1 #0D1117);
            border-bottom: 1px solid {BORDER_COLOR};
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 20, 0)

        # Ícono + título
        title = QLabel("🎯  ID / Re-ID System")
        title.setStyleSheet(f"""
            color: {TEXT_PRIMARY};
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)
        layout.addWidget(title)
        layout.addStretch()

        # Indicadores de modelo
        self._facial_badge = QLabel("⬤  SVM Facial")
        self._reid_badge   = QLabel("⬤  SVM Re-ID")
        for badge in (self._facial_badge, self._reid_badge):
            badge.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; margin-left: 16px;")
        layout.addWidget(self._facial_badge)
        layout.addWidget(self._reid_badge)

        # Live indicator
        self._live_badge = QLabel("◉  IDLE")
        self._live_badge.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px; margin-left: 24px; font-weight:600;")
        layout.addWidget(self._live_badge)

        return bar

    def _build_video_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {BG_DARK};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 6, 12)
        layout.setSpacing(8)

        # Display de video
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._video_label.setMinimumSize(640, 400)
        self._video_label.setStyleSheet(f"""
            background: #000000;
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
        """)
        self._show_placeholder()
        layout.addWidget(self._video_label, 1)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {BG_DARK};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 12, 12, 12)
        layout.setSpacing(10)

        # ── Controles ──────────────────────────────────────────────────
        ctrl_card = StyledCard()
        ctrl_layout = QVBoxLayout(ctrl_card)
        ctrl_layout.setContentsMargins(12, 10, 12, 10)
        ctrl_layout.setSpacing(8)

        ctrl_title = QLabel("⚙  Controles")
        ctrl_title.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight:600; letter-spacing:1px;")
        ctrl_layout.addWidget(ctrl_title)

        btn_row1 = QHBoxLayout()
        self._btn_open  = StyledButton("📂  Abrir Video", ACCENT_BLUE)
        self._btn_cam   = StyledButton("📷  Cámara",      ACCENT_GREEN)
        btn_row1.addWidget(self._btn_open)
        btn_row1.addWidget(self._btn_cam)
        ctrl_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self._btn_pause = StyledButton("⏸  Pausar",    ACCENT_ORANGE)
        self._btn_stop  = StyledButton("⏹  Detener",   ACCENT_RED)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        btn_row2.addWidget(self._btn_pause)
        btn_row2.addWidget(self._btn_stop)
        ctrl_layout.addLayout(btn_row2)

        layout.addWidget(ctrl_card)

        # ── Personas activas ────────────────────────────────────────────
        tracks_card = StyledCard()
        tracks_layout = QVBoxLayout(tracks_card)
        tracks_layout.setContentsMargins(12, 10, 12, 10)
        tracks_layout.setSpacing(6)

        tracks_title = QLabel("👥  Personas Activas")
        tracks_title.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight:600; letter-spacing:1px;")
        tracks_layout.addWidget(tracks_title)

        self._tracks_table = QTableWidget(0, 4)
        self._tracks_table.setHorizontalHeaderLabels(["Track", "Identidad", "Conf.", "Rama"])
        self._tracks_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tracks_table.verticalHeader().setVisible(False)
        self._tracks_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._tracks_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._tracks_table.setMaximumHeight(180)
        self._tracks_table.setStyleSheet(f"""
            QTableWidget {{
                background: {BG_PANEL};
                gridline-color: {BORDER_COLOR};
                color: {TEXT_PRIMARY};
                border: none;
                border-radius: 4px;
                font-size: 12px;
            }}
            QHeaderView::section {{
                background: {BG_CARD};
                color: {TEXT_MUTED};
                border: none;
                border-bottom: 1px solid {BORDER_COLOR};
                padding: 4px;
                font-size: 11px;
                font-weight: 600;
            }}
            QTableWidget::item {{ padding: 3px 6px; }}
        """)
        tracks_layout.addWidget(self._tracks_table)
        layout.addWidget(tracks_card)

        # ── Distribución de Ramas ───────────────────────────────────────
        branch_card = StyledCard()
        branch_layout = QVBoxLayout(branch_card)
        branch_layout.setContentsMargins(12, 10, 12, 10)
        branch_layout.setSpacing(6)

        branch_title = QLabel("📊  Distribución de Ramas")
        branch_title.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight:600; letter-spacing:1px;")
        branch_layout.addWidget(branch_title)

        self._branch_bar = BranchBar()
        branch_layout.addWidget(self._branch_bar)
        layout.addWidget(branch_card)

        # ── Log de Eventos ──────────────────────────────────────────────
        log_card = StyledCard()
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(12, 10, 12, 10)
        log_layout.setSpacing(6)

        log_title = QLabel("📝  Log de Eventos")
        log_title.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight:600; letter-spacing:1px;")
        log_layout.addWidget(log_title)

        self._event_log = EventLog()
        self._event_log.setMinimumHeight(160)
        log_layout.addWidget(self._event_log)
        layout.addWidget(log_card, 1)

        # Conexión de botones
        self._btn_open.clicked.connect(self._on_open_video)
        self._btn_cam.clicked.connect(self._on_use_camera)
        self._btn_pause.clicked.connect(self._on_pause_resume)
        self._btn_stop.clicked.connect(self._on_stop)

        return panel

    def _build_statusbar_widget(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(36)
        bar.setStyleSheet(f"""
            background: {BG_PANEL};
            border-top: 1px solid {BORDER_COLOR};
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(0)

        style = f"color: {TEXT_MUTED}; font-size: 11px; padding-right: 20px;"
        self._fps_label       = QLabel("FPS: —")
        self._proc_label      = QLabel("Procesados: 0")
        self._skipped_label   = QLabel("Saltados: 0")
        self._source_label    = QLabel("Fuente: —")

        for lbl in (self._fps_label, self._proc_label,
                    self._skipped_label, self._source_label):
            lbl.setStyleSheet(style)
            layout.addWidget(lbl)

        layout.addStretch()

        self._version_label = QLabel("ID/Re-ID System  v1.0")
        self._version_label.setStyleSheet(f"color: {BORDER_COLOR}; font-size: 10px;")
        layout.addWidget(self._version_label)

        return bar

    # ------------------------------------------------------------------ styles
    def _apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background: {BG_DARK}; }}
            QSplitter {{ background: {BG_DARK}; }}
            QScrollBar:vertical {{
                background: {BG_PANEL}; width: 8px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_COLOR}; border-radius: 4px; min-height: 20px;
            }}
            QToolTip {{
                background: {BG_CARD};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                padding: 4px;
                font-size: 11px;
            }}
        """)

    # ------------------------------------------------------------------ model check
    def _check_models(self):
        facial_path = Path(self.config.get("models", {}).get("svm_facial_path", ""))
        reid_path   = Path(self.config.get("models", {}).get("svm_reid_path", ""))

        if facial_path.exists():
            self._facial_badge.setStyleSheet(
                f"color: {ACCENT_GREEN}; font-size: 12px; margin-left: 16px; font-weight:600;"
            )
            self._facial_badge.setText(f"✅  SVM Facial")
        else:
            self._facial_badge.setStyleSheet(
                f"color: {ACCENT_RED}; font-size: 12px; margin-left: 16px;"
            )
            self._facial_badge.setText("❌  SVM Facial")

        if reid_path.exists():
            self._reid_badge.setStyleSheet(
                f"color: {ACCENT_GREEN}; font-size: 12px; margin-left: 16px; font-weight:600;"
            )
            self._reid_badge.setText("✅  SVM Re-ID")
        else:
            self._reid_badge.setStyleSheet(
                f"color: {ACCENT_RED}; font-size: 12px; margin-left: 16px;"
            )
            self._reid_badge.setText("❌  SVM Re-ID")

    # ------------------------------------------------------------------ placeholder
    def _show_placeholder(self):
        pw = self._video_label.width()  or 640
        ph = self._video_label.height() or 400
        img = QImage(pw, ph, QImage.Format.Format_RGB888)
        img.fill(QColor("#000000"))
        painter = QPainter(img)
        painter.setPen(QColor(BORDER_COLOR))
        painter.setFont(QFont("Segoe UI", 14))
        painter.drawText(img.rect(), Qt.AlignmentFlag.AlignCenter,
                         "Sin señal de video\n\nAbre un video o activa la cámara")
        painter.end()
        self._video_label.setPixmap(QPixmap.fromImage(img))

    # ------------------------------------------------------------------ worker mgmt
    def _start_pipeline(self, source):
        self._stop_worker()

        src_str = str(source)
        self._source_label.setText(
            f"Fuente: {'Cámara' if src_str == '0' else Path(src_str).name}"
        )

        self._worker = PipelineWorker(source=source, config=self.config)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.stats_update.connect(self._on_stats_update)
        self._worker.event_logged.connect(self._on_event_logged)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished_ok.connect(self._on_pipeline_finished)
        self._worker.start()

        self._btn_pause.setEnabled(True)
        self._btn_stop.setEnabled(True)
        self._btn_open.setEnabled(False)
        self._btn_cam.setEnabled(False)
        self._btn_pause.setText("⏸  Pausar")
        self._is_paused = False

        self._live_badge.setText("◉  LIVE")
        self._live_badge.setStyleSheet(
            f"color: {ACCENT_GREEN}; font-size: 13px; margin-left: 24px; font-weight:600;"
        )

    def _stop_worker(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        self._worker = None

    # ------------------------------------------------------------------ slots
    @pyqtSlot(object, list)
    def _on_frame_ready(self, bgr_frame: np.ndarray, results: list):
        self._last_results = results
        self._frame_count += 1

        # Convertir BGR → RGB → QImage → QPixmap
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)

        lw = self._video_label.width()
        lh = self._video_label.height()
        pixmap = QPixmap.fromImage(qimg).scaled(
            lw, lh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(pixmap)

        # Actualizar tabla de tracks
        self._update_tracks_table(results)

    @pyqtSlot(float, int, int, int, int)
    def _on_stats_update(self, fps: float, processed: int, skipped: int,
                         id_count: int, reid_count: int):
        self._fps_label.setText(f"FPS: {fps:.1f}")
        self._proc_label.setText(f"Procesados: {processed:,}")
        self._skipped_label.setText(f"Saltados: {skipped:,}")
        self._branch_bar.update_counts(id_count, reid_count)

    @pyqtSlot(str)
    def _on_event_logged(self, msg: str):
        self._event_log.add_event(msg)

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self._event_log.add_event(f"<span style='color:{ACCENT_RED};'>⚠ {msg}</span>")
        self._live_badge.setText("◉  ERROR")
        self._live_badge.setStyleSheet(
            f"color: {ACCENT_RED}; font-size: 13px; margin-left: 24px; font-weight:600;"
        )
        self._reset_controls()

    @pyqtSlot()
    def _on_pipeline_finished(self):
        self._live_badge.setText("◉  IDLE")
        self._live_badge.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 13px; margin-left: 24px; font-weight:600;"
        )
        self._reset_controls()
        self._show_placeholder()

    def _reset_controls(self):
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_open.setEnabled(True)
        self._btn_cam.setEnabled(True)

    # ------------------------------------------------------------------ table update
    def _update_tracks_table(self, results: list):
        self._tracks_table.setRowCount(len(results))
        for row, r in enumerate(results):
            is_unknown = r.identity == "Desconocido"
            is_id      = r.branch_used == "ID"

            id_item   = QTableWidgetItem(str(r.track_id))
            name_item = QTableWidgetItem(r.identity)
            conf_item = QTableWidgetItem(f"{r.confidence:.2f}")
            branch_item = QTableWidgetItem(r.branch_used)

            # Color por estado
            if is_unknown:
                name_item.setForeground(QColor(ACCENT_RED))
            elif is_id:
                name_item.setForeground(QColor(ACCENT_GREEN))
            else:
                name_item.setForeground(QColor(ACCENT_ORANGE))

            branch_item.setForeground(
                QColor(ACCENT_GREEN) if is_id else QColor(ACCENT_ORANGE)
            )

            for item in (id_item, name_item, conf_item, branch_item):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self._tracks_table.setItem(row, 0, id_item)
            self._tracks_table.setItem(row, 1, name_item)
            self._tracks_table.setItem(row, 2, conf_item)
            self._tracks_table.setItem(row, 3, branch_item)

    # ------------------------------------------------------------------ button handlers
    def _on_open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Video",
            str(Path(self.config.get("video", {}).get("source", ".")).parent),
            "Videos (*.mp4 *.avi *.mkv *.mov *.wmv);;Todos (*)",
        )
        if path:
            self._event_log.add_event(f"📂 Abriendo: {Path(path).name}")
            self._start_pipeline(path)

    def _on_use_camera(self):
        self._event_log.add_event("📷 Activando cámara (índice 0)")
        self._start_pipeline(0)

    def _on_pause_resume(self):
        if not self._worker:
            return
        if self._is_paused:
            self._worker.resume()
            self._btn_pause.setText("⏸  Pausar")
            self._is_paused = False
            self._live_badge.setText("◉  LIVE")
            self._live_badge.setStyleSheet(
                f"color: {ACCENT_GREEN}; font-size: 13px; margin-left: 24px; font-weight:600;"
            )
            self._event_log.add_event("▶ Pipeline reanudado")
        else:
            self._worker.pause()
            self._btn_pause.setText("▶  Reanudar")
            self._is_paused = True
            self._live_badge.setText("◉  PAUSED")
            self._live_badge.setStyleSheet(
                f"color: {ACCENT_ORANGE}; font-size: 13px; margin-left: 24px; font-weight:600;"
            )
            self._event_log.add_event("⏸ Pipeline en pausa")

    def _on_stop(self):
        self._event_log.add_event("⏹ Detención solicitada")
        self._stop_worker()
        self._on_pipeline_finished()

    # ------------------------------------------------------------------ close
    def closeEvent(self, event):
        self._stop_worker()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def launch_dashboard(config: dict):
    """Lanza la aplicación PyQt6 con el dashboard. Bloquea hasta que se cierra."""
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ID/Re-ID System")
    app.setStyle("Fusion")
    _apply_dark_palette(app)

    window = DashboardWindow(config=config)
    window.show()

    # Cargar la fuente por defecto del config al iniciar
    default_source = config.get("video", {}).get("source", "")
    if default_source:
        src = int(default_source) if str(default_source).isdigit() else default_source
        window._event_log.add_event(
            f"💡 Fuente por defecto en config.yaml: "
            f"{'Cámara' if str(default_source) == '0' else Path(str(default_source)).name}"
        )

    return app.exec()
