"""Detecta falsas alarmas de fondo/personas no registradas y las almacena como negativos difíciles (bootstrapping)."""
from pathlib import Path
import numpy as np
from utils.file_io_helpers import save_image
from decision_engine.unknown_labeler import UNKNOWN_LABEL
from utils.logger import get_logger

logger = get_logger(__name__)


class HardNegativeMiner:
    def __init__(self, output_dir: str = "dataset/hard_negatives", false_alarm_threshold: float = 0.5):
        self.output_dir = Path(output_dir)
        self.false_alarm_threshold = false_alarm_threshold
        self._counter = 0

    def evaluate_and_mine(self, roi: np.ndarray, predicted_identity: str, confidence: float,
                           ground_truth_is_registered: bool) -> bool:
        """
        Si el sistema clasifica como identidad registrada (con alta confianza) a un sujeto
        no registrado (ground_truth_is_registered=False), se almacena como Hard Negative.
        """
        is_false_alarm = (
            predicted_identity != UNKNOWN_LABEL
            and confidence >= self.false_alarm_threshold
            and not ground_truth_is_registered
        )
        if not is_false_alarm:
            return False

        filename = f"hard_negative_{self._counter:06d}.jpg"
        success = save_image(roi, self.output_dir / filename)
        if success:
            self._counter += 1
            logger.info(f"Hard negative minado y almacenado: {filename}")
        return success