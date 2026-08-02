"""Carga y validación de configuración global (config.yaml)."""
from pathlib import Path
from typing import Any, Dict
import yaml

_DEFAULT_CONFIG_PATH = "config.yaml"


class ConfigLoader:
    """Carga config.yaml una sola vez (patrón singleton simple) y expone acceso por punto."""

    _instance: Dict[str, Any] | None = None
    _path: str = _DEFAULT_CONFIG_PATH

    @classmethod
    def load(cls, path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
        if cls._instance is None or path != cls._path:
            config_path = Path(path)
            if not config_path.exists():
                raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")
            with open(config_path, "r", encoding="utf-8") as f:
                cls._instance = yaml.safe_load(f)
            cls._path = path
        return cls._instance

    @classmethod
    def get(cls, dotted_key: str, default: Any = None) -> Any:
        """Obtiene un valor anidado usando notación de puntos, ej: 'detection.conf_threshold'."""
        cfg = cls.load()
        node: Any = cfg
        for key in dotted_key.split("."):
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node