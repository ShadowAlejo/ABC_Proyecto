"""Utilidades de E/S de archivos para el dataset y modelos."""
import json
import pickle
from pathlib import Path
from typing import Any, Iterable
import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_image(image: np.ndarray, path: str | Path) -> bool:
    p = Path(path)
    ensure_dir(p.parent)
    return cv2.imwrite(str(p), image)


def load_image(path: str | Path) -> np.ndarray | None:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def save_pickle(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_files(directory: str | Path, extensions: Iterable[str]) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    exts = {e.lower() for e in extensions}
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in exts)


def count_files_in_subdir(base_dir: str | Path, subdir_name: str, extensions: Iterable[str]) -> int:
    return len(list_files(Path(base_dir) / subdir_name, extensions))