"""Carga desde disco los modelos SVM serializados según la ruta configurada en config.yaml."""
from pathlib import Path
from typing import Any
from utils.file_io_helpers import load_pickle
from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)


def load_svm_model(model_key: str) -> Any:
    """
    Carga un modelo SVM serializado según la clave de configuración
    (ej: 'models.svm_facial_path' o 'models.svm_reid_path').
    """
    model_path = ConfigLoader.get(model_key)
    if model_path is None:
        raise ValueError(f"No se encontró la ruta de modelo para la clave de configuración: {model_key}")

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo SVM no encontrado en: {path}")

    logger.info(f"Cargando modelo SVM desde: {path}")
    return load_pickle(path)